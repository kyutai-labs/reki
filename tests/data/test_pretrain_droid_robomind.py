"""Data-free parts of the DROID and RoboMIND readers: the DROID referenced-file list, the
failure and video-cut rules, the RoboMIND archive extraction, text rules, the UR frozen-arm detector, and the
registry check of the datasets that train without subtask lookups."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pretrain.common import detect
from pretrain.common.tasks import robomind_annotation, robomind_task
from pretrain.common.v30 import Episode, write_leaf
from pretrain.readers import droid, robomind

pytestmark = pytest.mark.cpu


def test_droid_downloads_only_referenced_files(tmp_path: Path) -> None:
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    info = {
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    cols = {"data/chunk_index": [0, 0], "data/file_index": [0, 2]}
    for cam, files in zip(droid.CAMERAS, ([0, 0], [1, 1], [0, 3])):
        cols[f"videos/observation.images.{cam}/chunk_index"] = [0, 0]
        cols[f"videos/observation.images.{cam}/file_index"] = files
    pq.write_table(
        pa.table(cols), tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert droid.referenced_files(tmp_path) == [
        "data/chunk-000/file-000.parquet",
        "data/chunk-000/file-002.parquet",
        "videos/observation.images.exterior_1_left/chunk-000/file-000.mp4",
        "videos/observation.images.exterior_2_left/chunk-000/file-001.mp4",
        "videos/observation.images.wrist_left/chunk-000/file-000.mp4",
        "videos/observation.images.wrist_left/chunk-000/file-003.mp4",
    ]


def test_video_cut_rule_drops_whole_stitched_files() -> None:
    a, b = Path("a.mp4"), Path("b.mp4")
    spans = {
        a: [(0, 0.0, 10.0), (1, 10.0, 20.0), (2, 20.0, 30.0), (3, 30.0, 40.0)],
        b: [(4, 0.0, 10.0), (5, 10.0, 20.0), (6, 20.0, 30.0), (7, 30.0, 40.0), (8, 40.0, 50.0)],
    }
    # a: one of four episodes holds a cut (25%: dropped); b: one of five, the other cuts sit
    # within 0.5 s of an episode edge (20%: kept).
    cuts = {a: [5.0, 10.2], b: [3.0, 10.3, 29.8, 40.4]}
    assert detect.cut_files(spans, cuts) == [0, 1, 2, 3]


def test_droid_failure_drops_unsuccessful_episodes(tmp_path: Path) -> None:
    # Episode 1 spans two data files; episode 3 is the other failure.
    files = {0: ([0, 0, 1], [True, True, False]), 1: ([1, 2, 3, 3], [False, True, False, False])}
    for f, (eps, ok) in files.items():
        path = tmp_path / "data" / "chunk-000" / f"file-{f:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"episode_index": eps, "is_episode_successful": ok}), path)
    assert detect.failure(tmp_path, "droid") == [1, 3]
    assert "failure" in droid.RECIPES["droid"].curate


def _archive(tmp_path: Path, members: dict[str, bytes]) -> list[Path]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=0) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    cut = len(raw) * 3 // 4
    parts = [tmp_path / "t.tar.gz.part-aa", tmp_path / "t.tar.gz.part-ab"]
    parts[0].write_bytes(raw[:cut])
    parts[1].write_bytes(raw[cut:])
    return parts


def test_robomind_extract_whole_and_first_part(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    members = {
        f"task/success_episodes/train/{i}/data/trajectory.hdf5": rng.bytes(4000) for i in range(4)
    }
    parts = _archive(tmp_path, members)
    robomind.extract(parts, tmp_path / "whole", whole=True)
    assert len(list((tmp_path / "whole").rglob("trajectory.hdf5"))) == 4
    robomind.extract(parts[:1], tmp_path / "first", whole=False)
    got = sorted(
        p.relative_to(tmp_path / "first").as_posix() for p in (tmp_path / "first").rglob("*.hdf5")
    )
    assert got == sorted(members)[: len(got)] and 0 < len(got) < 4
    assert all((tmp_path / "first" / g).read_bytes() == members[g] for g in got)
    with pytest.raises((EOFError, tarfile.ReadError)):
        robomind.extract(parts[:1], tmp_path / "strict", whole=True)


def test_robomind_text_rules() -> None:
    assert robomind_task("bread on the table") == "Place the bread on the table."
    assert robomind_task("pick up the mochine cup") == "Pick up the machine cup."
    assert robomind_annotation("placing two plates into a dish rack") == (
        "Place two plates into the dish rack."
    )


def test_ur_frozen_arm(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 40
    frozen = np.tile(rng.normal(size=7), (n, 1)).astype(np.float32)
    frozen[:, 6] = np.linspace(0, 1, n)
    moving = frozen.copy()
    moving[:, 0] += np.linspace(0, 0.5, n, dtype=np.float32)
    eps = [Episode(i, s, s, "t") for i, s in enumerate([frozen, moving])]
    write_leaf(eps, tmp_path / "leaf", 30, "ur")
    assert detect.ur_frozen_arm(tmp_path / "leaf", "robomind_ur_1rgb") == [0]


# ------------------------------------------------------------------ training


def test_subtask_coverage_is_explicit() -> None:
    from reki.data.robotics.combined_dataset import check_subtask_coverage

    names = ["droid", "galaxea", "galaxea"]
    check_subtask_coverage(names, [None, "subtask_lookup.jsonl", None])
    with pytest.raises(
        ValueError, match="no subtask lookup in \\['galaxea'\\].*subtask_lookup=True"
    ):
        check_subtask_coverage(names, [None, None, None])
    with pytest.raises(ValueError, match="\\['droid'\\] have a subtask lookup.*subtask_lookup"):
        check_subtask_coverage(names, ["subtask_lookup.jsonl", "subtask_lookup.jsonl", None])
    check_subtask_coverage(["robomind_ur_1rgb"], [None])
