"""The v3.0 writer and the v2.x converter of the pretraining pipeline (``pretrain.common.v30``)."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pretrain.common import verify
from pretrain.common.v30 import (
    Episode,
    convert_v2x,
    read_leaf,
    read_v2x_episodes,
    umask_modes,
    write_leaf,
)
from pretrain.readers.lerobot_v2x import V2X_COLUMNS

pytestmark = pytest.mark.cpu


def _episodes(n: int, frames: bool = False) -> list[Episode]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        length = 4 + i
        ep = Episode(
            i,
            rng.normal(size=(length, 3)).astype(np.float32),
            rng.normal(size=(length, 2)).astype(np.float32),
            f"task {i % 2}",
        )
        if frames:
            ep.frames["observation.images.top"] = rng.integers(
                0, 255, size=(length, 32, 32, 3), dtype=np.uint8
            )
        out.append(ep)
    return out


def test_write_leaf_round_trip(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = _episodes(3, frames=True)
    leaf = tmp_path / "leaf"
    write_leaf(eps, leaf, fps=10, robot_type="test")
    arrays = read_leaf(leaf, ["observation.state", "action", "episode_index", "task_index"])
    np.testing.assert_array_equal(
        arrays["observation.state"], np.concatenate([e.state for e in eps])
    )
    np.testing.assert_array_equal(arrays["action"], np.concatenate([e.action for e in eps]))
    np.testing.assert_array_equal(arrays["episode_index"], np.repeat([0, 1, 2], [4, 5, 6]))
    np.testing.assert_array_equal(arrays["task_index"], np.repeat([0, 1, 0], [4, 5, 6]))
    verify.check_videos(leaf)

    ds = LeRobotDataset(repo_id="local/leaf", root=leaf, revision="local")
    assert (ds.meta.total_episodes, ds.meta.total_frames) == (3, 15)
    item = ds[5]
    np.testing.assert_array_equal(item["action"].numpy(), eps[1].action[1])
    assert item["task"] == "task 1"
    assert tuple(item["observation.images.top"].shape) == (3, 32, 32)


def test_check_videos_allows_trailing_frames(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf"
    write_leaf(_episodes(2, frames=True), leaf, fps=10, robot_type="test")
    path = next((leaf / "meta" / "episodes").rglob("*.parquet"))
    table = pq.read_table(path).to_pydict()
    table["length"][0] -= 3  # upstream mp4 longer than its episode
    pq.write_table(pa.table(table), path)
    verify.check_videos(leaf)
    key = "videos/observation.images.top/from_timestamp"
    table[key][1] += 0.1  # a one-frame shift still fails
    pq.write_table(pa.table(table), path)
    with pytest.raises(ValueError, match="episode 1 has"):
        verify.check_videos(leaf)


def test_check_leaf_without_band_names_the_stats_command(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf"
    write_leaf(_episodes(2), leaf, fps=10, robot_type="test")
    command = "pretrain_data.py stats d --root R"
    with pytest.raises(KeyError, match=r"(?s)leaf: stats.json .* lacks .*stats d --root R"):
        verify.check_leaf(leaf, command)
    (leaf / "meta" / "stats.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"(?s)leaf: no meta/stats.json.*stats d --root R"):
        verify.check_leaf(leaf, command)


def test_write_leaf_rolls_data_files(tmp_path: Path) -> None:
    write_leaf(_episodes(4), tmp_path / "leaf", fps=10, robot_type="test", data_file_mb=0)
    assert len(list((tmp_path / "leaf" / "data").glob("*/*.parquet"))) == 4


def _v21(
    root: Path, lengths: list[int], v20: bool = False, float64: tuple[int, ...] = ()
) -> list[np.ndarray]:
    """A v2.1 dataset without videos (``v20``: one ``meta/stats.json`` instead of per-episode
    stats; ``float64``: episodes whose state/action are stored as ``list<double>``, as in some
    RoboCOIN leaves, though ``info.json`` says float32); returns each episode's action."""
    rng = np.random.default_rng(0)
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        **{
            k: {"dtype": "int64", "shape": [1], "names": None}
            for k in ("frame_index", "episode_index", "index", "task_index")
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "t"}) + "\n")
    actions, offset = [], 0
    with (
        open(root / "meta" / "episodes.jsonl", "w") as e,
        open(root / "meta" / "episodes_stats.jsonl", "w") as s,
    ):
        for i, n in enumerate(lengths):
            e.write(json.dumps({"episode_index": i, "tasks": ["t"], "length": n}) + "\n")
            action = rng.normal(size=(n, 2))
            if i not in float64:
                action = action.astype(np.float32)
            cols = {
                "observation.state": action * 2,
                "action": action,
                "reward": np.zeros((n, 1), np.float32),
            }
            table = pa.table(
                {
                    **{
                        k: pa.array(list(v))
                        if v.dtype == np.float64
                        else pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), v.shape[1])
                        for k, v in cols.items()
                    },
                    "timestamp": pa.array(np.arange(n, dtype=np.float32) / 10),
                    "frame_index": pa.array(np.arange(n)),
                    "episode_index": pa.array(np.full(n, i)),
                    "index": pa.array(np.arange(offset, offset + n)),
                    "task_index": pa.array(np.zeros(n, dtype=np.int64)),
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
            stats = {
                k: {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.1], "count": [n]}
                for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")
            }
            stats |= {
                k: {
                    "min": v.min(0).tolist(),
                    "max": v.max(0).tolist(),
                    "mean": v.mean(0).tolist(),
                    "std": v.std(0).tolist(),
                    "count": [n],
                }
                for k, v in cols.items()
            }
            if not v20:
                s.write(json.dumps({"episode_index": i, "stats": stats}) + "\n")
            actions.append(action)
            offset += n
    return actions


def test_v2x_enumerates_from_meta(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _v21(src, [3, 4])
    orphan = src / "data" / "chunk-000" / "episode_000002.parquet"
    orphan.write_bytes((src / "data" / "chunk-000" / "episode_000000.parquet").read_bytes())
    assert [e.index for e in read_v2x_episodes(src)] == [0, 1]
    eps = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    eps[1]["length"] = 5
    (src / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    with pytest.raises(ValueError, match="meta says 5"):
        list(read_v2x_episodes(src))


def test_convert_v2x_keeps_renumbers_and_filters(tmp_path: Path) -> None:
    src = tmp_path / "src"
    actions = _v21(src, [3, 4, 5])
    out = tmp_path / "out"
    keep_columns = (
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    )
    convert_v2x(src, out, [0, 2], keep_columns)
    arrays = read_leaf(out, ["action", "episode_index", "index"])
    np.testing.assert_array_equal(arrays["action"], np.concatenate([actions[0], actions[2]]))
    np.testing.assert_array_equal(arrays["episode_index"], np.repeat([0, 1], [3, 5]))
    np.testing.assert_array_equal(arrays["index"], np.arange(8))
    info = json.loads((out / "meta" / "info.json").read_text())
    assert (info["codebase_version"], info["total_episodes"], info["total_frames"]) == (
        "v3.0",
        2,
        8,
    )
    assert "reward" not in info["features"]
    assert "reward" not in pq.read_schema(next((out / "data").glob("*/*.parquet"))).names
    assert not (tmp_path / ".out.v2x").exists()


def test_convert_v2x_transform_and_meta_from_data(tmp_path: Path) -> None:
    src = tmp_path / "src"
    actions = _v21(src, [3, 4])
    eps = (src / "meta" / "episodes.jsonl").read_text().splitlines()
    (src / "meta" / "episodes.jsonl").write_text(eps[0] + "\n")
    out = tmp_path / "out"
    convert_v2x(src, out, [0, 1], V2X_COLUMNS, lambda s, a: (s, a * 2), rebuild_meta=True)
    arrays = read_leaf(out, ["action", "observation.state"])
    np.testing.assert_array_equal(arrays["action"], 2 * np.concatenate(actions))
    np.testing.assert_array_equal(arrays["observation.state"], 2 * np.concatenate(actions))
    assert json.loads((out / "meta" / "info.json").read_text())["total_episodes"] == 2


@pytest.mark.parametrize("keep_columns", [V2X_COLUMNS, (*V2X_COLUMNS, "reward")])
def test_convert_v2x_casts_mixed_dtypes_to_info(
    tmp_path: Path, keep_columns: tuple[str, ...]
) -> None:
    src = tmp_path / "src"
    actions = _v21(src, [3, 4, 5], float64=(1,))
    assert pq.read_schema(src / "data" / "chunk-000" / "episode_000001.parquet").field(
        "action"
    ).type == pa.list_(pa.float64())
    out = tmp_path / "out"
    convert_v2x(src, out, [0, 1, 2], keep_columns)
    for f in (out / "data").glob("*/*.parquet"):
        schema = pq.read_schema(f)
        for key in ("observation.state", "action"):
            assert schema.field(key).type.value_type == pa.float32()
    arrays = read_leaf(out, ["action", "observation.state"])
    expected = np.concatenate([a.astype(np.float32) for a in actions])
    np.testing.assert_array_equal(arrays["action"], expected)
    np.testing.assert_array_equal(
        arrays["observation.state"], np.concatenate([(2 * a).astype(np.float32) for a in actions])
    )


def test_convert_v20_links_data(tmp_path: Path) -> None:
    src = tmp_path / "src"
    actions = _v21(src, [3, 4], v20=True)
    (src / "meta" / "episodes_stats.jsonl").unlink()
    (src / "meta" / "stats.json").write_text(json.dumps({"action": {"q01": [0.0, 0.0]}}))
    info = json.loads((src / "meta" / "info.json").read_text())
    (src / "meta" / "info.json").write_text(json.dumps({**info, "total_videos": 2}))
    out = tmp_path / "out"
    convert_v2x(src, out, [1], V2X_COLUMNS)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_videos"], info["total_chunks"]) == (1, 0, 1)
    arrays = read_leaf(out, ["action", "episode_index"])
    np.testing.assert_array_equal(arrays["action"], actions[1])
    np.testing.assert_array_equal(arrays["episode_index"], np.zeros(4))
    assert json.loads((out / "meta" / "stats.json").read_text())["action"]["q01"] == [0.0, 0.0]
    shutil.rmtree(src)
    assert not (tmp_path.parent / ".v2x_stage" / tmp_path.name / "out").exists()
    np.testing.assert_array_equal(read_leaf(out, ["action"])["action"], actions[1])


def test_frame_tasks_set_task_index(tmp_path: Path) -> None:
    (ep,) = _episodes(1)
    ep.frame_tasks = ["a", "a", "b", "b"]
    write_leaf([ep], tmp_path / "leaf", fps=10, robot_type="test")
    assert read_leaf(tmp_path / "leaf", ["task_index"])["task_index"].tolist() == [1, 1, 2, 2]
    tasks = pq.read_table(tmp_path / "leaf" / "meta" / "tasks.parquet").to_pylist()
    assert [t["task"] for t in sorted(tasks, key=lambda t: t["task_index"])] == ["task 0", "a", "b"]


def test_transform_changing_widths_restates_features(tmp_path: Path) -> None:
    src = tmp_path / "src"
    actions = _v21(src, [3, 4])
    out = tmp_path / "out"
    convert_v2x(src, out, [0, 1], V2X_COLUMNS, lambda s, a: (s, a[:, :1] + 1))
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [1]
    assert info["features"]["action"]["names"] is None
    stats = json.loads((out / "meta" / "stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["max"], np.concatenate(actions)[:, :1].max(0) + 1)


def test_v2x_without_frame_index_gets_one(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _v21(src, [3, 4])
    for path in (src / "data").glob("*/*.parquet"):
        pq.write_table(pq.read_table(path).drop_columns(["frame_index"]), path)
    info = json.loads((src / "meta" / "info.json").read_text())
    del info["features"]["frame_index"]
    (src / "meta" / "info.json").write_text(json.dumps(info))
    out = tmp_path / "coll" / "leaf"
    convert_v2x(src, out, [0, 1], V2X_COLUMNS)
    np.testing.assert_array_equal(
        read_leaf(out, ["frame_index"])["frame_index"], [0, 1, 2, 0, 1, 2, 3]
    )
    assert not (tmp_path / "coll" / ".leaf.v2x").exists()


def test_built_files_follow_the_umask(tmp_path: Path) -> None:
    import os
    import stat

    outside = tmp_path / "upstream.mp4"
    outside.write_bytes(b"x")
    outside.chmod(0o600)
    write_leaf(_episodes(2, frames=True), tmp_path / "leaf", fps=10, robot_type="test")
    (tmp_path / "leaf" / "link.mp4").symlink_to(outside)
    (tmp_path / "leaf" / "private").mkdir(mode=0o700)
    mask = os.umask(0o027)
    try:
        umask_modes(tmp_path / "leaf")
    finally:
        os.umask(mask)
    modes = {
        p.name: stat.S_IMODE(p.lstat().st_mode)
        for p in (tmp_path / "leaf").rglob("*")
        if not p.is_symlink()
    }
    assert {m for n, m in modes.items() if n.endswith((".mp4", ".parquet", ".json"))} == {0o640}
    assert modes["private"] == 0o750
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


def _mp4_episodes(tmp_path: Path, sizes: list[tuple[int, int]], length: int) -> list[Episode]:
    """Episodes of ``length`` frames whose camera is an x264 mp4 (``video_src``, B-frames as
    RoboMIND's) at 30 fps, one per frame size in ``sizes``."""
    from pretrain.common.video import FrameEncoder
    from pretrain.readers.robomind import X264

    rng = np.random.default_rng(0)
    out = []
    for i, (h, w) in enumerate(sizes):
        mp4 = tmp_path / "src" / f"{i}.mp4"
        mp4.parent.mkdir(exist_ok=True)
        encoder = FrameEncoder(mp4, 30, h, w, X264)
        for f in range(length):
            encoder.write(np.full((h, w, 3), (i * 37 + f * 5) % 256, dtype=np.uint8))
        encoder.close()
        arr = rng.normal(size=(length, 2)).astype(np.float32)
        out.append(Episode(i, arr, arr, "t", video_src={"observation.images.top": mp4}))
    return out


def test_joined_frames_sit_on_the_loader_grid(tmp_path: Path) -> None:
    """52-frame episodes: ffmpeg's concat demuxer (lerobot's ``concatenate_video_files``)
    offsets each by 1733333 us, 1/3 us early, and ~100 joins later every frame sits one
    1/15360 tick early, which the loader skips (RoboMIND franka_3rgb). The writer's frames
    sit at exactly k/30 and verify catches the concat demuxer's file."""
    from lerobot.datasets.video_utils import concatenate_video_files
    from pretrain.common.video import packet_pts

    eps = _mp4_episodes(tmp_path, [(16, 16)] * 150, 52)
    leaf = tmp_path / "leaf"
    write_leaf(eps, leaf, fps=30, robot_type="test")
    (video,) = (leaf / "videos").rglob("*.mp4")
    pts, time_base, fps = packet_pts(video)
    assert (time_base, fps) == (Fraction(1, 15360), 30.0)
    np.testing.assert_array_equal(pts, np.arange(150 * 52) * 512)
    verify.check_videos(leaf)

    concatenate_video_files([e.video_src["observation.images.top"] for e in eps], video)
    with pytest.raises(ValueError, match="decoded off its timestamps"):
        verify.check_videos(leaf)


def test_a_new_frame_size_starts_a_new_file(tmp_path: Path) -> None:
    """RoboMIND cameras switch between 1280x720 and 640x480 across episodes; one file of both
    fails to decode (torchcodec allocates the header's size), so each size run gets a file."""
    from lerobot.datasets.video_utils import concatenate_video_files

    eps = _mp4_episodes(tmp_path, [(32, 48), (32, 48), (16, 24), (32, 48)], 40)
    leaf = tmp_path / "leaf"
    write_leaf(eps, leaf, fps=30, robot_type="test")
    table = pq.read_table(next((leaf / "meta" / "episodes").rglob("*.parquet"))).to_pydict()
    assert table["videos/observation.images.top/file_index"] == [0, 0, 1, 2]
    assert table["videos/observation.images.top/from_timestamp"] == [0, 40 / 30, 0, 0]
    verify.check_videos(leaf)

    files = sorted((leaf / "videos").rglob("*.mp4"))
    # file 0 with episode 1's segment in the other size, as the concat demuxer joins them
    concatenate_video_files([eps[i].video_src["observation.images.top"] for i in (0, 2)], files[0])
    with pytest.raises(ValueError, match="episodes 0-1: .*pre-allocated tensor"):
        verify.check_videos(leaf)


def test_link_tree_outlives_its_source(tmp_path: Path) -> None:
    from shared import link_tree

    src = tmp_path / "upstream" / "videos"
    (src / "cam").mkdir(parents=True)
    (src / "cam" / "a.mp4").write_bytes(b"a")
    (tmp_path / "blob").write_bytes(b"b")
    (src / "cam" / "b.mp4").symlink_to(tmp_path / "blob")
    link_tree(src, tmp_path / "leaf" / "videos")
    link_tree(src / "cam" / "a.mp4", tmp_path / "leaf" / "a.mp4")
    shutil.rmtree(tmp_path / "upstream")
    (tmp_path / "blob").unlink()
    leaf = tmp_path / "leaf"
    assert [(leaf / p).read_bytes() for p in ("videos/cam/a.mp4", "videos/cam/b.mp4", "a.mp4")] == [
        b"a",
        b"b",
        b"a",
    ]
    assert not any(p.is_symlink() for p in leaf.rglob("*"))
