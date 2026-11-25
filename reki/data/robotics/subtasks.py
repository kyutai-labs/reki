"""Per-episode subtask labels: loading the lookup file, load-time interval transforms, and
the per-frame caption and success label."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

Intervals = dict[int, list[tuple[int, int, str]]]
"""Episode → start-sorted, non-overlapping ``(start_frame, end_frame, caption)`` intervals."""
DEFAULT_LOOKUP = "subtask_lookup.jsonl"


@dataclass
class SubtaskLookup:
    """One leaf's ``meta/<filename>``; ``filename`` is None when the leaf has no lookup."""

    intervals: Intervals | None = None
    tasks: dict[int, str] | None = None
    """Episode → task prompt that replaces the episode's stored task."""
    keep_ranges: dict[int, list[list[int]]] | None = None
    """Episode → the ``[action_start, action_end]`` frame range (leading/trailing idle cut)."""
    mistake_intervals: dict[int, set[int]] | None = None
    """Episode → indices of the intervals flagged as a failed attempt. A dataset that annotates
    mistakes lists only the episodes with one, so an absent episode is all-clean; None means
    the dataset annotates no mistakes at all."""
    filename: str | None = None


def load_subtask_lookup(ds_root: Path, filename: str = DEFAULT_LOOKUP) -> SubtaskLookup:
    """Load a pre-built subtask lookup jsonl.

    Each line: ``{"episode_index": N, "task": "...", "intervals": [[s, e, text], ...],
    "action_start": int, "action_end": int, "mistake_intervals": [i, ...]}``; every field but
    ``episode_index`` is optional.

    Falls back to ``subtask_lookup.jsonl`` for a leaf without ``filename``. The dataset raises
    if ``predict_subtask`` is on and no dataset in the mix has ``filename``.
    """
    used = next(
        (c for c in (filename, DEFAULT_LOOKUP) if (ds_root / "meta" / c).exists()),
        None,
    )
    if used is None:
        return SubtaskLookup(tasks={})
    intervals: Intervals = {}
    tasks: dict[int, str] = {}
    keep_ranges: dict[int, list[list[int]]] = {}
    mistake_intervals: dict[int, set[int]] = {}
    for line in (ds_root / "meta" / used).read_text().splitlines():
        entry = json.loads(line)
        ep = int(entry["episode_index"])
        intervals[ep] = [tuple(iv) for iv in entry.get("intervals") or []]  # type: ignore[misc]
        task_val = entry.get("task")
        if task_val:
            tasks[ep] = str(task_val)
        if "action_start" in entry and "action_end" in entry:
            keep_ranges[ep] = [[int(entry["action_start"]), int(entry["action_end"])]]
        if "mistake_intervals" in entry:
            mistake_intervals[ep] = {int(i) for i in entry["mistake_intervals"] or []}
    return SubtaskLookup(
        intervals, tasks, keep_ranges or None, mistake_intervals or None, filename=used
    )


def _interval_index(intervals: list[tuple[int, int, str]], frame_index: int) -> int | None:
    """Index of the interval that contains ``frame_index``, or None in a gap."""
    idx = bisect.bisect_right(intervals, frame_index, key=lambda x: x[0]) - 1
    if idx >= 0 and frame_index < intervals[idx][1]:
        return idx
    return None


def interval_text(lookup: Intervals | None, episode_index: int, frame_index: int) -> str | None:
    """The caption of the interval that contains this frame; None in a gap, for an unannotated
    episode, or for a blank caption."""
    intervals = lookup.get(episode_index) if lookup is not None else None
    if not intervals:
        return None
    idx = _interval_index(intervals, frame_index)
    if idx is None:
        return None
    text = intervals[idx][2]
    return text if text.strip() else None


def interval_success(
    lookup: Intervals | None,
    mistake_intervals: dict[int, set[int]] | None,
    episode_index: int,
    frame_index: int,
) -> bool | None:
    """Whether the subtask interval at this frame is a clean (non-mistake) attempt.

    ``None`` when the dataset/episode carries no mistake annotation (≠ success — the
    label is simply unavailable) or the frame falls outside any interval. ``True`` for
    a clean interval, ``False`` for one flagged in ``mistake_intervals``.
    """
    if mistake_intervals is None:
        return None
    mistakes = mistake_intervals.get(episode_index, set())
    intervals = (lookup or {}).get(episode_index)
    if not intervals:
        return None
    idx = _interval_index(intervals, frame_index)
    return None if idx is None else idx not in mistakes
