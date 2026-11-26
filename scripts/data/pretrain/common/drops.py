"""The list of excluded episodes of a dataset: the episodes ``curate`` drops, named by
content (``dedup.episode_key`` of the reader's stable columns), so that it does not depend on
episode order, leaf names or the machine.

``pretrain/drops/<dataset>.json``:
``{"dropped": [key, ...], "kept": {key: label, ...}}``.

- ``dropped``: every episode with this key is dropped (a detector hit).
- ``kept``: copies of one episode (duplicates). The first copy in (leaf, episode) order with this
  label is kept, the others are dropped. The label is the task text, or the reader's
  (``Reader.dedup_labels``).

``curate --detect`` makes the list again from the detectors and the duplicates. A copy that a
detector hits is not kept. Of the other copies, the copy with the label the existing list keeps
is kept, else the first.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import dedup

DIR = Path(__file__).parents[1] / "drops"

Copy = tuple[int, int, str]
"""One episode of a dataset: leaf index, episode index, label."""


@dataclass(frozen=True)
class Drops:
    dropped: list[dedup.Key]
    kept: dict[dedup.Key, str]


def path(name: str) -> Path:
    return DIR / f"{name}.json"


def load(name: str) -> Drops | None:
    """The list of excluded episodes of ``name``; None: there is none."""
    p = path(name)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Drops(d["dropped"], d["kept"])


def save(name: str, drops: Drops) -> None:
    p = path(name)
    p.write_text(json.dumps({"dropped": drops.dropped, "kept": drops.kept}, indent=0) + "\n")


def groups(
    leaves: list[Path],
    labels: Callable[[Path], list[str] | None],
    columns: dedup.Columns | None = None,
) -> dict[dedup.Key, list[Copy]]:
    """The episodes of ``leaves`` by key, in (leaf, episode) order. ``labels(leaf)``: the
    per-episode labels (None: the task texts); ``columns``: the key's (``Reader.key_columns``)."""
    out: dict[dedup.Key, list[Copy]] = {}
    for i, leaf in enumerate(leaves):
        leaf_labels = labels(leaf)
        for ep, task, state, action in dedup.leaf_episodes(leaf):
            label = task if leaf_labels is None else leaf_labels[ep]
            out.setdefault(dedup.episode_key(state, action, columns), []).append((i, ep, label))
    return out


def detect(
    groups: dict[dedup.Key, list[Copy]], hits: set[tuple[int, int]], old: Drops | None
) -> Drops:
    """The list of excluded episodes of ``groups``: the ``(leaf, episode)`` detector
    ``hits`` and the duplicates; ``old`` gives the kept label where it is still possible."""
    dropped: list[dedup.Key] = []
    kept: dict[dedup.Key, str] = {}
    for key, copies in groups.items():
        first: dict[str, Copy] = {}
        for copy in copies:
            first.setdefault(copy[2], copy)
        labels = [label for label, copy in first.items() if copy[:2] not in hits]
        if not labels:
            dropped.append(key)
        elif len(copies) > 1:
            prior = old.kept.get(key) if old else None
            kept[key] = prior if prior in labels else labels[0]
    return Drops(sorted(dropped), dict(sorted(kept.items())))


def apply(drops: Drops, groups: dict[dedup.Key, list[Copy]], n_leaves: int) -> list[list[int]]:
    """Per leaf, the sorted episodes that ``drops`` drops. Every listed key must occur and every
    duplicate must be listed: else the build differs from the one the list was made on."""
    missing = [k for k in [*drops.dropped, *drops.kept] if k not in groups]
    unlisted = [
        k
        for k, c in groups.items()
        if len(c) > 1 and k not in drops.kept and k not in drops.dropped
    ]
    if missing or unlisted:
        raise ValueError(
            f"{len(missing)} listed episodes not found, {len(unlisted)} duplicates not listed: the "
            "build differs from the released one. Run download and build of this dataset again. "
            "A different CPU, GPU or software version can also change the values: then "
            "`curate <dataset> --detect` makes the list again on this machine (a duplicate can "
            "keep another copy than the release). If the error stays, please open an issue at "
            "https://github.com/kyutai-labs/reki/issues"
        )
    out: list[set[int]] = [set() for _ in range(n_leaves)]
    for key in drops.dropped:
        for i, ep, _ in groups[key]:
            out[i].add(ep)
    for key, label in drops.kept.items():
        copies = groups[key]
        keep = next((c for c in copies if c[2] == label), None)
        if keep is None:
            raise ValueError(f"{key}: no copy has the kept label {label!r}")
        for i, ep, _ in copies:
            if (i, ep) != keep[:2]:
                out[i].add(ep)
    return [sorted(s) for s in out]
