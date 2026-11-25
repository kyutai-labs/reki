"""Checks on built leaves: counts vs the pins, drop keys, stats keys, video frame counts, and a
one-sample load through the training loader."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from ..pins import Pin
from .stats import dropped

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig
from .v30 import data_files, episode_bounds, read_leaf, video_keys
from .video import loader_pts, packet_pts

DECODE_BATCH = 16
"""Frames decoded per call in ``check_decode``."""


def _episodes_table(leaf: Path) -> dict[str, list]:
    files = sorted((leaf / "meta" / "episodes").glob("**/*.parquet"))
    tables = [pq.read_table(f) for f in files]
    cols = [c for c in tables[0].column_names if not c.startswith("stats/")]
    return {c: [v for t in tables for v in t.column(c).to_pylist()] for c in cols}


@dataclass
class _Segment:
    """One episode on one camera: its video file, the frames of its segment
    ``[from_timestamp, to_timestamp)`` and the timestamps the loader queries for its rows."""

    path: Path
    episode: int
    length: int
    frames: int
    queries: np.ndarray


def _segments(
    leaf: Path,
) -> Iterator[tuple[Path, list[_Segment], tuple[np.ndarray, Fraction, float]]]:
    """Per video file of a leaf: its episode segments and its packet pts (``packet_pts``)."""
    info = json.loads((leaf / "meta" / "info.json").read_text())
    half = 0.5 / info["fps"]
    eps = _episodes_table(leaf)
    timestamps = read_leaf(leaf, ["timestamp"])["timestamp"]
    for key in video_keys(info):
        rows = list(
            zip(
                eps["episode_index"],
                eps["length"],
                eps["dataset_from_index"],
                *(eps[f"videos/{key}/{k}"] for k in ("chunk_index", "file_index")),
                *(eps[f"videos/{key}/{k}"] for k in ("from_timestamp", "to_timestamp")),
            )
        )
        paths = [
            leaf / info["video_path"].format(video_key=key, chunk_index=c, file_index=f)
            for _, _, _, c, f, _, _ in rows
        ]
        unique = list(dict.fromkeys(paths))
        with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
            times = dict(zip(unique, pool.map(packet_pts, unique)))
        files: dict[Path, list[_Segment]] = {}
        for path, (ep, n, i0, _, _, t0, t1) in zip(paths, rows):
            pts, time_base, _ = times[path]
            start, stop = np.searchsorted(pts * float(time_base), [t0 - half, t1 - half])
            frames = int(stop - start)
            queries = t0 + timestamps[i0 : i0 + min(n, frames)].astype(np.float64)
            files.setdefault(path, []).append(_Segment(path, ep, n, frames, queries))
        for path, segments in files.items():
            yield path, segments, times[path]


def _aligned(seg: _Segment, timing: tuple[np.ndarray, Fraction, float]) -> bool:
    """Whether the training loader decodes the frame of each queried row of ``seg``
    (``video.loader_pts`` within the loader's tolerance)."""
    from reki.data.robotics.factory import VIDEO_TOLERANCE_S

    pts, time_base, fps = timing
    got = loader_pts(pts, time_base, fps, seg.queries)
    return bool(np.all(np.abs(got - seg.queries) < VIDEO_TOLERANCE_S))


def episode_video_frames(leaf: Path) -> Iterator[tuple[Path, int, int, int, bool]]:
    """Per camera and episode: the video file, the episode, its length, the frames of its
    segment ``[from_timestamp, to_timestamp)`` and whether the training loader decodes the
    frame of each of its first ``length`` rows."""
    for path, segments, timing in _segments(leaf):
        for seg in segments:
            yield path, seg.episode, seg.length, seg.frames, _aligned(seg, timing)


def _decode_starts(path: Path, segments: list[_Segment]) -> None:
    """Decode the first frame of each segment through the training loader's decode."""
    from reki.data.robotics.factory import VIDEO_TOLERANCE_S
    from reki.data.robotics.lerobot_patches import (
        _BoundedDecoderCache,
        _decode_video_frames_torchcodec_uint8,
    )

    cache = _BoundedDecoderCache(1)
    starts = [(s.episode, float(s.queries[0])) for s in segments if len(s.queries)]
    for i in range(0, len(starts), DECODE_BATCH):
        batch = starts[i : i + DECODE_BATCH]
        try:
            _decode_video_frames_torchcodec_uint8(
                path, [t for _, t in batch], VIDEO_TOLERANCE_S, decoder_cache=cache
            )
        except Exception as e:
            raise ValueError(f"{path}: episodes {batch[0][0]}-{batch[-1][0]}: {e}") from e


def check_videos(leaf: Path) -> None:
    """Each episode's segment ``[from_timestamp, to_timestamp)`` of its video file holds as many
    frames as the episode, and the loader decodes the frame at each row's timestamp; trailing
    extra frames (an upstream mp4 longer than its episode) are allowed, training never reads
    them. A dropped episode may hold fewer frames (``detect.short_video``). Other segments of
    the file may belong to no episode. Then the loader decodes the first frame of every
    episode on every camera: an episode starts a segment of a joined file, so this sees each
    stream format a file holds (a size change fails, the decoder allocates the header's)."""
    drops = dropped(leaf)
    files = []
    for path, segments, timing in _segments(leaf):
        for seg in segments:
            aligned = _aligned(seg, timing)
            if seg.frames < seg.length and seg.episode in drops and aligned:
                continue
            if seg.frames < seg.length or not aligned:
                raise ValueError(
                    f"{path}: episode {seg.episode} has {seg.frames} frames, length "
                    f"{seg.length}{'' if aligned else ', decoded off its timestamps'}"
                )
        files.append((path, segments))
    with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        list(pool.map(lambda f: _decode_starts(*f), files))


def check_leaf(leaf: Path, stats_command: str) -> None:
    """Meta totals, data rows, drop keys and stats keys of one leaf agree. A missing
    ``stats.json`` or band names ``stats_command``, which writes them."""
    info = json.loads((leaf / "meta" / "info.json").read_text())
    from_idx, to_idx, eps = episode_bounds(leaf)
    rows = sum(pq.read_metadata(f).num_rows for f in data_files(leaf))
    expected = (info["total_episodes"], info["total_frames"], info["total_frames"])
    if (len(eps), to_idx[-1], rows) != expected:
        raise ValueError(
            f"{leaf}: {len(eps)} episodes / {to_idx[-1]} frames / {rows} data rows, info.json "
            f"{info['total_episodes']} / {info['total_frames']}"
        )
    if eps != list(range(len(eps))) or from_idx != [0, *to_idx[:-1]]:
        raise ValueError(f"{leaf}: meta/episodes is not contiguous from episode 0")
    drops_path = leaf / "meta" / "dropped_episodes.json"
    if drops_path.exists():
        unknown = set(json.loads(drops_path.read_text())) - set(eps)
        if unknown:
            raise KeyError(f"{drops_path}: unknown episodes {sorted(unknown)[:5]}")
    path = leaf / "meta" / "stats.json"
    if not path.exists():
        raise FileNotFoundError(f"{leaf}: no meta/stats.json. Run:\n  {stats_command}")
    stats = json.loads(path.read_text())
    for key in ("observation.state", "action"):
        missing = {"qk01", "qk99", "max"} - set(stats[key])
        if missing:
            raise KeyError(
                f"{leaf}: stats.json {key} lacks {sorted(missing)}. Run:\n  {stats_command}"
            )
    check_videos(leaf)


def check_counts(name: str, leaves: list[Path], pin: Pin, drops: list[list[int]]) -> None:
    """Episodes and frames of the dataset match the pin; each leaf drops what its list of
    excluded episodes drops (``drops``, per leaf)."""
    infos = [json.loads((leaf / "meta" / "info.json").read_text()) for leaf in leaves]
    episodes = sum(i["total_episodes"] for i in infos)
    frames = sum(i["total_frames"] for i in infos)
    if episodes != pin.episodes or pin.frames not in (None, frames):
        raise ValueError(
            f"{name}: {episodes} episodes / {frames} frames, pinned {pin.episodes} / {pin.frames}"
        )
    for leaf, eps in zip(leaves, drops):
        if dropped(leaf) != set(eps):
            raise ValueError(
                f"{leaf}: meta/dropped_episodes.json differs from the list of excluded episodes"
            )


def check_euler(name: str, leaves: list[Path]) -> None:
    """An axis-angle dataset has its euler sidecar in every leaf."""
    from reki.data.robotics.registry import AXIS_ANGLE, DATASETS

    spec = DATASETS[name]
    if AXIS_ANGLE not in (spec.action_rot_rep, spec.state_rot_rep):
        return
    sidecar = "norm_stats_euler_roll.json" if spec.recenter_state_roll else "norm_stats_euler.json"
    for leaf in leaves:
        if not (leaf / "meta" / sidecar).exists():
            raise FileNotFoundError(f"{leaf}: no meta/{sidecar}")


def dry_load(name: str, root: Path, train: TrainConfig) -> None:
    """Build the training dataset of ``name`` under ``root`` at the horizon of ``train`` and
    load its first kept sample."""
    from reki.data.robotics.factory import VALIDATION_SETTINGS, make_robotics_dataset
    from reki.policy.train_config import TrainConfig

    os.environ["REKI_DATASETS_ROOT"] = str(root)
    cfg = TrainConfig(
        dataset=name,
        vlm_model_id=train.vlm_model_id,
        chunk_duration_s=train.chunk_duration_s,
        chunk_size=train.chunk_size,
        action_pace={k: v for k, v in train.action_pace.items() if k == name},
        **VALIDATION_SETTINGS,
    )
    ds, _, _ = make_robotics_dataset(cfg, [name])
    first = next(i for i in range(len(ds)) if ds.is_kept(i))
    sample = ds[first]
    for item in sample.get("group_items", [sample]):
        if not np.isfinite(np.asarray(item["action"])).all():
            raise ValueError(f"{name}: non-finite action in sample {first}")
