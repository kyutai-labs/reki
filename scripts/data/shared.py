"""Helpers shared by the data scripts: shard selection, ``qk*`` quantile stats, the v3.0
rewrite of a GR00T-style LeRobot v2.x export, and the axis-angle → euler conversion of the
euler stats. The scripts run from the repo root as ``python scripts/data/<script>.py``, which
puts this directory on ``sys.path``."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

QK_KEYS = {"q01": "qk01", "q10": "qk10", "q50": "qk50", "q90": "qk90", "q99": "qk99"}
"""lerobot quantile key → its ``qk*`` key (normalization reads qk01/qk99)."""
STATE_ACTION = ("action", "observation.state")


def identity(value: str) -> str:
    """The default ``task_text`` mapping: no change."""
    return value


def parse_shard(spec: str | None) -> tuple[int, int]:
    """``"i/N"`` → ``(i, N)``: a script processes ``items[i::N]``. ``None`` → everything."""
    if spec is None:
        return 0, 1
    i, n = (int(x) for x in spec.split("/"))
    return i, n


# ------------------------------------------------------------------ qk quantiles


def has_qk_stats(stats: dict[str, dict] | None) -> bool:
    """Whether any feature has normalization-band quantiles."""
    return stats is not None and any("qk01" in feature_stats for feature_stats in stats.values())


def read_state_action(name: str, lerobot_dir: Path) -> dict[str, np.ndarray]:
    """The action and state columns of a LeRobot dataset, as float32 arrays."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from reki.data.robotics.lerobot_patches import install_lerobot_patches

    install_lerobot_patches()
    dataset = LeRobotDataset(repo_id=f"local/{name}", root=lerobot_dir, revision="local")
    columns = [k for k in STATE_ACTION if k in dataset.features]
    if not columns:
        raise ValueError(f"{name}: no action/state columns")
    subset = dataset.select_columns(columns)
    return {key: np.array(subset[key], dtype=np.float32) for key in columns}


def qk_quantiles(arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    """lerobot feature stats of each array, with ``q*`` renamed to ``qk*`` (all keys kept)."""
    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats

    out: dict[str, dict] = {}
    for key, data in arrays.items():
        stats = get_feature_stats(
            data, axis=0, keepdims=data.ndim == 1, quantile_list=DEFAULT_QUANTILES
        )
        out[key] = {QK_KEYS.get(k, k): v for k, v in stats.items()}
    return out


def alias_qk_band(feature_stats: dict) -> None:
    """Copy q01/q99 to qk01/qk99, keeping the original quantiles."""
    for q_old, q_new in (("q01", "qk01"), ("q99", "qk99")):
        if q_old in feature_stats:
            feature_stats[q_new] = feature_stats[q_old]


def merge_stats(stats_dir: Path, new: dict[str, dict], only_qk: bool) -> None:
    """Merge ``new`` into ``stats_dir/meta/stats.json``, keeping every other key; with
    ``only_qk`` only the ``qk*`` keys of ``new``. Without a stats.json, write ``new``."""
    from lerobot.datasets.io_utils import load_stats, write_stats

    existing = load_stats(stats_dir)
    if existing is None:
        write_stats(new, stats_dir)
        return
    merged = {k: dict(v) for k, v in existing.items()}
    for col, col_stats in new.items():
        keep = (
            {k: v for k, v in col_stats.items() if k in QK_KEYS.values()} if only_qk else col_stats
        )
        merged.setdefault(col, {}).update(keep)
    write_stats(merged, stats_dir)


def apply_global_quantiles(leaves: list[Path], all_stats: bool = False) -> None:
    """Pool state and action over every frame of ``leaves``; write the same ``qk01/qk99`` into
    each leaf's stats.json. A collection with an absolute control mode needs this: training
    rejects per-leaf ``qk`` bands on it. ``all_stats``: also write the pooled q01/q99, min/max
    and mean/std."""
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in STATE_ACTION}
    for leaf in leaves:
        for pq_path in sorted(leaf.glob("data/**/*.parquet")):
            table = pq.read_table(pq_path, columns=list(STATE_ACTION))
            for k in STATE_ACTION:
                column = table.column(k).combine_chunks()
                flat = column.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
                pooled[k].append(flat.reshape(len(column), -1))
    stats: dict[str, dict] = {}
    for k, arrays in pooled.items():
        values = np.concatenate(arrays, axis=0)
        q01, q99 = np.quantile(values, 0.01, axis=0), np.quantile(values, 0.99, axis=0)
        stats[k] = {"qk01": q01, "qk99": q99}
        if all_stats:
            stats[k] |= {"q01": q01, "q99": q99, "min": values.min(0), "max": values.max(0)}
            stats[k] |= {"mean": values.mean(0), "std": values.std(0)}
    for leaf in leaves:
        merge_stats(leaf, stats, only_qk=not all_stats)


# ------------------------------------------------------------------ GR00T-style v2.x → v3.0

V30_META_FEATURES = {
    "episode_index": {"dtype": "int64", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "index": {"dtype": "int64", "shape": [1]},
    "task_index": {"dtype": "int64", "shape": [1]},
    "timestamp": {"dtype": "float32", "shape": [1]},
}
V30_VIDEO_PATH = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4"


def convert_groot_leaf(
    src_leaf: Path,
    dst_leaf: Path,
    dims: dict[str, int],
    task_text: Callable[[str], str] = identity,
) -> None:
    """Write a pseudo-v3.0 copy of a GR00T-style v2.x leaf (one parquet and one mp4 per
    episode, kept) to ``dst_leaf``: state/action cast to ``float32[dims[key]]``, a
    ``frame_index`` column, v3.0 ``meta/`` (info, tasks, episodes; task strings through
    ``task_text``), and ``stats.json`` with ``q01/q99`` aliased as ``qk01/qk99``. Videos are
    symlinked."""
    from lerobot.datasets.io_utils import write_stats

    info = json.load(open(src_leaf / "meta" / "info.json"))
    info["codebase_version"] = "v3.0"
    for key, dim in dims.items():
        info["features"][key] = {"dtype": "float32", "shape": [dim]}
    for k, v in V30_META_FEATURES.items():
        info["features"].setdefault(k, v)
    info["video_path"] = V30_VIDEO_PATH
    (dst_leaf / "meta").mkdir(parents=True, exist_ok=True)
    json.dump(info, open(dst_leaf / "meta" / "info.json", "w"), indent=4, ensure_ascii=False)

    for src in sorted(src_leaf.glob("data/**/*.parquet")):
        dst = dst_leaf / src.relative_to(src_leaf)
        dst.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(src)
        cols: dict[str, pa.Array] = {}
        for name in table.column_names:
            if name in dims:
                arr = np.stack(
                    [np.asarray(v, dtype=np.float32) for v in table.column(name).to_pylist()]
                )
                cols[name] = pa.FixedSizeListArray.from_arrays(
                    pa.array(arr.reshape(-1), pa.float32()), dims[name]
                )
            elif name == "timestamp":
                cols[name] = table.column(name).cast(pa.float32())
            else:
                cols[name] = table.column(name)
        if "frame_index" not in cols:
            cols["frame_index"] = pa.array(np.arange(table.num_rows, dtype=np.int64))
        pq.write_table(pa.table(cols), dst)

    write_v30_tasks(src_leaf, dst_leaf, task_text)
    fps = info.get("fps", 20)
    write_v30_episodes(src_leaf, dst_leaf, info, lambda _idx, length: length / fps, task_text)

    raw = json.load(open(src_leaf / "meta" / "stats.json"))
    stats = {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}
    for feat in stats.values():
        alias_qk_band(feat)
    write_stats(stats, dst_leaf)

    videos_link = dst_leaf / "videos"
    if videos_link.is_symlink() or videos_link.exists():
        videos_link.unlink()
    os.symlink(src_leaf / "videos", videos_link)


def write_v30_tasks(
    src_leaf: Path, dst_leaf: Path, task_text: Callable[[str], str] = identity
) -> None:
    """Write v3.0 task metadata, transforming each task string."""
    tasks = [json.loads(line) for line in open(src_leaf / "meta" / "tasks.jsonl")]
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([t["task_index"] for t in tasks], pa.int64()),
                "task": pa.array([task_text(t["task"]) for t in tasks], pa.string()),
            }
        ),
        dst_leaf / "meta" / "tasks.parquet",
    )


def write_v30_episodes(
    src_leaf: Path,
    dst_leaf: Path,
    info: dict,
    to_timestamp: Callable[[int, int], float],
    task_text: Callable[[str], str] = identity,
) -> int:
    """Write v3.0 ``meta/episodes`` and return the number of episodes."""
    chunks_size = info.get("chunks_size", 1000)
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    episodes = [json.loads(line) for line in open(src_leaf / "meta" / "episodes.jsonl")]

    data: dict[str, list] = {
        "episode_index": [],
        "length": [],
        "tasks": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        "meta/data/chunk_index": [],
        "meta/data/file_index": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": [],
    }
    for vk in video_keys:
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            data[f"videos/{vk}/{suffix}"] = []

    cumsum = 0
    for ep in episodes:
        idx, length = ep["episode_index"], ep["length"]
        data["episode_index"].append(idx)
        data["length"].append(length)
        data["tasks"].append([task_text(t) for t in ep.get("tasks", [])])
        data["dataset_from_index"].append(cumsum)
        cumsum += length
        data["dataset_to_index"].append(cumsum)
        data["meta/data/chunk_index"].append(idx // chunks_size)
        data["meta/data/file_index"].append(idx)
        data["meta/episodes/chunk_index"].append(0)
        data["meta/episodes/file_index"].append(0)
        end = to_timestamp(idx, length)
        for vk in video_keys:
            data[f"videos/{vk}/chunk_index"].append(idx // chunks_size)
            data[f"videos/{vk}/file_index"].append(idx)
            data[f"videos/{vk}/from_timestamp"].append(0.0)
            data[f"videos/{vk}/to_timestamp"].append(end)

    ep_dir = dst_leaf / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), ep_dir / "file-000.parquet")
    return len(episodes)


def replace_dir(path: Path, overwrite: bool) -> bool:
    """True when ``path`` is free to (re)write: absent, or removed with ``overwrite``."""
    if not path.exists():
        return True
    if not overwrite:
        return False
    shutil.rmtree(path)
    return True


# ------------------------------------------------------------------ euler stats


def raw_rot_blocks(
    blocks: list[tuple[int, int]], slice_list: list[int] | None
) -> list[tuple[int, int]]:
    """Map post-slice rotation blocks (``transforms.ROT_BLOCKS``) to raw column coords.

    The euler sidecar is written pre-slice (the loader slices it with the dataset's state /
    action slice), so a rotation block must be located in raw coords. Identity when the dataset
    is unsliced or the slice does not shift the block. Raises when a block is not contiguous
    in raw coords.
    """
    if slice_list is None:
        return blocks
    out = []
    for lo, hi in blocks:
        raw = [slice_list[j] for j in range(lo, hi)]
        if raw != list(range(raw[0], raw[0] + len(raw))):
            raise ValueError(f"rotation block {(lo, hi)} not contiguous in raw coords: {raw}")
        out.append((raw[0], raw[-1] + 1))
    return out


def axis_angle_to_euler(arr: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Convert each ``[lo:hi]`` block of a [N, D] array from axis-angle rotvec to euler xyz, as
    ``transforms.axisangle_to_euler`` does at load (same scipy ``"xyz"`` order), checking that
    the two describe the same rotation."""
    from scipy.spatial.transform import Rotation as R

    out = arr.astype(np.float64).copy()
    for lo, hi in blocks:
        rot = out[:, lo:hi]
        src = R.from_rotvec(rot)
        eul = src.as_euler("xyz")
        rel = (R.from_euler("xyz", eul) * src.inv()).magnitude()
        assert rel.max() < 1e-4, f"axis-angle↔euler rotation mismatch (max {rel.max():.2e} rad)"
        out[:, lo:hi] = eul
    return out
