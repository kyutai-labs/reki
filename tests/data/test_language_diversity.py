"""compute_language_diversity on a synthetic leaf: the texts training reads, kept frames only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from compute_language_diversity import effective_rank, leaf_text_frames

pytestmark = pytest.mark.cpu


def test_leaf_text_frames_reads_overrides_frame_tasks_and_drops(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        {"task_index": [0, 1]}, index=pd.Index(["pick_cup", "place cup."], name="task")
    ).to_parquet(tmp_path / "meta" / "tasks.parquet")
    pq.write_table(
        pa.table({"episode_index": [0, 0, 0, 1, 1, 2, 2], "task_index": [0, 0, 1, 0, 0, 0, 1]}),
        tmp_path / "data" / "file-000.parquet",
    )
    (tmp_path / "meta" / "subtask_lookup.jsonl").write_text(
        json.dumps({"episode_index": 1, "task": "Stack the cups.", "intervals": []}) + "\n"
    )
    (tmp_path / "meta" / "dropped_episodes.json").write_text("[2]")
    counts = leaf_text_frames(tmp_path, "subtask_lookup.jsonl")
    assert counts == {"Pick cup.": 2, "Place cup.": 1, "Stack the cups.": 2}


def test_effective_rank() -> None:
    one = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert effective_rank(one, np.array([1.0, 1.0])) == 1.0
    two = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(effective_rank(two, np.array([1.0, 1.0])), 2.0)
    assert effective_rank(two, np.array([9.0, 1.0])) < 2.0
