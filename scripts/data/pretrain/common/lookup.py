"""Subtask lookups from upstream instruction segments (AgiBot, Galaxea), and the reader base of
those two (``TarballLeaf``).

A leaf's ``meta/instruction_segments.jsonl`` holds, per episode, the upstream segments
``{"instruction", "start", "end"}`` (frames of the episode, end exclusive). ``from_segments``
turns one episode's segments into the sorted, non-overlapping ``[start, end, text]`` intervals
of ``meta/subtask_lookup.jsonl`` (the loader's format, ``reki.data.robotics.subtasks``):

- segments spanning >= 90% of the episode (the last segment end) are episode-wide labels and
  left out;
- the rest sort by (start, length); a segment that overlaps the previous interval replaces it
  when shorter, else it is left out.
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from ..pins import HfSnapshot, Pin
from ..recipe import Reader
from . import v30
from .download import upstream_dir
from .tasks import TextRules, text_rules

logger = logging.getLogger(__name__)

SEGMENTS = "instruction_segments.jsonl"
LOOKUP = "subtask_lookup.jsonl"
EPISODE_WIDE = 0.9


def from_segments(segments: list[dict]) -> list[list]:
    """``[start, end, instruction]`` intervals of one episode's segments."""
    if not segments:
        return []
    length = max(s["end"] for s in segments)
    steps = sorted(
        (s for s in segments if s["end"] - s["start"] < EPISODE_WIDE * length),
        key=lambda s: (s["start"], s["end"] - s["start"]),
    )
    intervals: list[list] = []
    for s in steps:
        if intervals and s["start"] < intervals[-1][1]:
            if s["end"] - s["start"] < intervals[-1][1] - intervals[-1][0]:
                intervals[-1] = [s["start"], s["end"], s["instruction"]]
        else:
            intervals.append([s["start"], s["end"], s["instruction"]])
    return intervals


def write_segments(leaf: Path, segments: list[list[dict]]) -> None:
    """``meta/instruction_segments.jsonl``: one line per episode, in episode order."""
    lines = [
        json.dumps({"episode_index": ep, "instruction_segments": segs}, ensure_ascii=False)
        for ep, segs in enumerate(segments)
    ]
    (leaf / "meta" / SEGMENTS).write_text("".join(f"{line}\n" for line in lines))


def write_lookup(leaf: Path, text: Callable[[str], str] = str) -> int:
    """``meta/subtask_lookup.jsonl`` from the leaf's segments, each instruction passed through
    ``text`` first (the dataset's subtask text rules). Episodes without an interval get no line.
    Returns the number of episodes written."""
    lines = []
    for line in (leaf / "meta" / SEGMENTS).read_text().splitlines():
        entry = json.loads(line)
        segs = [{**s, "instruction": text(s["instruction"])} for s in entry["instruction_segments"]]
        intervals = from_segments(segs)
        if intervals:
            row = {"episode_index": entry["episode_index"], "intervals": intervals}
            lines.append(json.dumps(row, ensure_ascii=False))
    (leaf / "meta" / LOOKUP).write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def runs(labels: Sequence[str | None]) -> list[tuple[int, int, str]]:
    """``(start, end, label)`` per run of equal consecutive labels (end exclusive); runs of None
    are left out."""
    out = []
    for label, run in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
        frames = [k for k, _ in run]
        if label is not None:
            out.append((frames[0], frames[-1] + 1, label))
    return out


class TarballLeaf(Reader):
    """One leaf ``<root>/<leaf>`` from the pinned tarballs of one HF source, each a LeRobot v2.1
    dataset, read in pin order (no shards). The instruction segments of the episodes go to
    ``meta/instruction_segments.jsonl``, the lookup is written at curate."""

    leaf: str
    fps: int
    robot_type: str

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [root / self.leaf]

    def tarball_episodes(
        self, name: str, archive: Path, work: Path, rules: TextRules
    ) -> Iterator[tuple[v30.Episode, list[dict]]]:
        """The episodes of one tarball and their instruction segments; ``work`` is the
        tarball's scratch dir, kept until the leaf is written."""
        raise NotImplementedError

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        if shard != (0, 1):
            raise ValueError(f"{name}: one leaf, no shards")
        (src,) = pin.sources
        assert isinstance(src, HfSnapshot) and src.allow is not None
        allow = src.allow
        up = upstream_dir(root, src)
        (out,) = self.leaves(name, root, pin)
        work = out.parent / f".{out.name}.work"
        if work.exists():
            shutil.rmtree(work)
        rules = text_rules(name)
        segments: list[list[dict]] = []

        def episodes() -> Iterator[v30.Episode]:
            for i, path in enumerate(allow):
                logger.info("%s: tarball %d/%d %s", name, i + 1, len(allow), path)
                for ep, segs in self.tarball_episodes(name, up / path, work / f"{i:03d}", rules):
                    segments.append(segs)
                    yield ep

        v30.write_leaf(episodes(), out, self.fps, self.robot_type)
        shutil.rmtree(work)
        write_segments(out, segments)
        info = json.loads((out / "meta" / "info.json").read_text())
        if (info["total_episodes"], info["total_frames"]) != (pin.episodes, pin.frames):
            raise ValueError(
                f"{name}: {info['total_episodes']} episodes / {info['total_frames']} frames, "
                f"pinned {pin.episodes} / {pin.frames}"
            )

    def curate(self, name: str, root: Path, pin: Pin) -> None:
        (leaf,) = self.leaves(name, root, pin)
        n = write_lookup(leaf, text_rules(name).subtask)
        logger.info("%s: subtask lookup for %d episodes", name, n)
