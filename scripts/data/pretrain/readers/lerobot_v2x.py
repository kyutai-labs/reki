"""Readers for one upstream LeRobot repo → one leaf: v2.x (OXE v2.1/v2.0,
berkeley_rpt), episodes from the pinned meta, converted to v3.0 (``v30.convert_v2x``); v3.0
(bridge_v2), the selected episodes copied (``v30.select_v30``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..common import dedup, v30
from ..common.download import upstream_dir
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, root_dir

logger = logging.getLogger(__name__)

V2X_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def episode_files(episodes: tuple[int, ...], video_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Allow-patterns of ``meta/**`` and the data and video files of ``episodes`` (chunks of
    1000, the IPEC layout): the ``--subset`` download of a v2.x repo."""
    out = ["meta/**"]
    for ep in episodes:
        out.append(f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
        out += [f"videos/chunk-{ep // 1000:03d}/{k}/episode_{ep:06d}.mp4" for k in video_keys]
    return tuple(out)


@dataclass(frozen=True)
class _UpstreamLeaf(Reader):
    """One upstream LeRobot repo → one leaf ``<registry root>/<leaf>``."""

    leaf: str

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [root_dir(name, root) / self.leaf]

    def upstream(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> Path:
        """The upstream dir; its ``info.json`` totals must be the full pin's."""
        from ..catalog import pins

        if shard != (0, 1):
            raise ValueError(f"{name}: one leaf, no shards")
        (source,) = pin.sources
        assert isinstance(source, HfSnapshot)
        src = upstream_dir(root, source)
        full = pins()[name]
        v30.check_info_totals(src, full.episodes, full.frames)
        return src


@dataclass(frozen=True)
class LeRobotV2x(_UpstreamLeaf):
    """A v2.x repo, converted; ``keep_columns`` are the data columns written (video features are
    always kept); ``transform`` changes state/action per episode; ``key``: the episode key's
    columns (``Reader.key_columns``)."""

    keep_columns: tuple[str, ...]
    transform: v30.Transform | None = None
    key: dedup.Columns | None = None

    def key_columns(self) -> dedup.Columns | None:
        return self.key

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        src = self.upstream(name, root, pin, shard)
        lengths = v30.v2x_lengths(src)
        keep = list(pin.select) if pin.select is not None else list(lengths)
        frames = sum(lengths[i] for i in keep)
        if (len(keep), frames) != (pin.episodes, pin.frames):
            raise ValueError(
                f"{name}: {len(keep)} episodes / {frames} frames, pinned {pin.episodes} / "
                f"{pin.frames}"
            )
        (out,) = self.leaves(name, root, pin)
        v30.convert_v2x(src, out, keep, self.keep_columns, self.transform)
        logger.info("%s: %d episodes → %s", name, len(keep), out)


@dataclass(frozen=True)
class PassthroughV30(_UpstreamLeaf):
    """A v3.0 repo, values unchanged (``v30.select_v30``)."""

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        src = self.upstream(name, root, pin, shard)
        (out,) = self.leaves(name, root, pin)
        totals = v30.select_v30(src, out, pin.select)
        if totals != (pin.episodes, pin.frames):
            raise ValueError(
                f"{name}: {totals} episodes / frames, pinned {pin.episodes} / {pin.frames}"
            )
        logger.info("%s: %d episodes → %s", name, totals[0], out)
