"""The content key of an episode: duplicates within a dataset, and the episode ID of the lists
of excluded episodes (``drops``).

Key = the first 16 hex digits of sha1(float32 little-endian state ‖ action), then the length.
State and action are values computed by the reader; a reader whose values depend on the
hardware or software (a GPU fit, a BLAS kernel) hashes only its stable columns
(``Reader.key_columns``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .v30 import STATE_ACTION, episode_bounds, read_leaf

Key = str
Columns = tuple[tuple[int, ...], tuple[int, ...]]
"""The state and action columns that a key hashes."""


def episode_key(state: np.ndarray, action: np.ndarray, columns: Columns | None = None) -> Key:
    """The content key of one episode; ``columns``: hash only these (None: all)."""
    if columns is not None:
        state, action = state[:, list(columns[0])], action[:, list(columns[1])]
    h = hashlib.sha1(np.ascontiguousarray(state, dtype="<f4").tobytes())
    h.update(np.ascontiguousarray(action, dtype="<f4").tobytes())
    return f"{h.hexdigest()[:16]}-{len(state)}"


def leaf_episodes(leaf: Path) -> Iterator[tuple[int, str, np.ndarray, np.ndarray]]:
    """``(episode, task, state, action)`` of a v3.0 leaf; the task is ``meta/episodes``'s
    task list joined by newlines."""
    arrays = read_leaf(leaf, STATE_ACTION)
    files = sorted((leaf / "meta" / "episodes").glob("**/*.parquet"))
    tasks = pa.concat_tables([pq.read_table(f, columns=["tasks"]) for f in files])
    for f, t, ep, task in zip(*episode_bounds(leaf), tasks.column("tasks").to_pylist()):
        yield ep, "\n".join(task), arrays["observation.state"][f:t], arrays["action"][f:t]


def shared_across(keys: dict[str, set[Key]]) -> int:
    """How many keys occur in more than one dataset (reported, not removed)."""
    seen: dict[Key, int] = {}
    for dataset_keys in keys.values():
        for k in dataset_keys:
            seen[k] = seen.get(k, 0) + 1
    return sum(n > 1 for n in seen.values())
