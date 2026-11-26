"""LeRobot v3.0 output: the writer for readers that yield episode arrays, the lerobot
v2.x → v3.0 converter and the v3.0 episode selection for upstream LeRobot data, and the leaf
readers the other stages use."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shared import link_tree

STATE_ACTION = ("observation.state", "action")
DATA_FILE_MB = 100
VIDEO_FILE_MB = 200
ENCODER = {"vcodec": "libsvtav1", "pix_fmt": "yuv420p", "g": 2, "crf": 30}
"""Pinned encoder settings for frames a reader yields as arrays (lerobot's defaults)."""
ENCODE_WORKERS = 8
"""Episode videos that ``write_leaf`` encodes at the same time."""


def umask_modes(top: Path) -> None:
    """Give every file and directory under ``top`` (not symlinks, not files hard-linked to
    upstream) the mode a plain ``open`` / ``mkdir`` gives under the process umask. Tools that
    write through temp files (lerobot's video concatenation) leave 0600 files, which a group
    cannot read."""
    mask = os.umask(0)
    os.umask(mask)
    for dirpath, dirnames, filenames in os.walk(top):
        for name, mode in [(d, 0o777) for d in dirnames] + [(f, 0o666) for f in filenames]:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            if os.path.islink(path) or (mode == 0o666 and st.st_nlink > 1):
                continue
            if st.st_mode & 0o777 != mode & ~mask:
                os.chmod(path, mode & ~mask)


@dataclass
class Episode:
    """One episode from a reader: ``[N, D]`` state/action, the task text, and per camera key
    either ``frames`` (``[N, H, W, 3]`` uint8, encoded) or ``video_src`` (an mp4, stream-copied).
    ``extra`` holds other ``[N, ...]`` columns; ``index`` is the upstream episode index.
    ``frame_tasks`` (len N): per-frame task texts for ``task_index``; ``meta/episodes`` keeps
    ``[task]``."""

    index: int
    state: np.ndarray
    action: np.ndarray
    task: str
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    video_src: dict[str, Path] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    frame_tasks: list[str] | None = None


# ------------------------------------------------------------------ leaf readers


def data_files(leaf: Path) -> list[Path]:
    """The data parquets of a leaf, sorted."""
    files = sorted((leaf / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{leaf}: no data parquets")
    return files


def read_columns(path: Path, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Columns of one parquet as numpy arrays in their stored dtype; list columns are 2-D."""
    return table_columns(pq.read_table(path, columns=list(columns)), columns)


def table_columns(table: pa.Table, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """``read_columns`` of an in-memory table."""
    out: dict[str, np.ndarray] = {}
    for col in columns:
        arr = table.column(col).combine_chunks()
        if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            out[col] = arr.flatten().to_numpy(zero_copy_only=False).reshape(len(arr), -1)
        else:
            out[col] = arr.to_numpy(zero_copy_only=False)
    return out


def concat_columns(
    cols: dict[str, np.ndarray], layout: Sequence[str | tuple[str, int]]
) -> tuple[np.ndarray, list[str]]:
    """The ``layout`` columns of ``cols`` side by side as ``[N, D]`` float32, and the dim names
    ``<column>.<i>``. A ``(column, width)`` entry must have that width."""
    parts, names = [], []
    for entry in layout:
        col, width = entry if isinstance(entry, tuple) else (entry, None)
        arr = cols[col].reshape(len(cols[col]), -1).astype(np.float32)
        if width is not None and arr.shape[1] != width:
            raise ValueError(f"{col} has {arr.shape[1]} dims, layout {width}")
        parts.append(arr)
        names += [f"{col}.{i}" for i in range(arr.shape[1])]
    return np.concatenate(parts, axis=1), names


def read_leaf(leaf: Path, columns: Sequence[str] = STATE_ACTION) -> dict[str, np.ndarray]:
    """``columns`` of every frame of a leaf, in data-file order."""
    parts = [read_columns(f, columns) for f in data_files(leaf)]
    return {col: np.concatenate([p[col] for p in parts]) for col in columns}


def episode_bounds(leaf: Path) -> tuple[list[int], list[int], list[int]]:
    """``(first frame, end frame, episode index)`` lists from ``meta/episodes``."""
    files = sorted((leaf / "meta" / "episodes").glob("**/*.parquet"))
    cols = ["dataset_from_index", "dataset_to_index", "episode_index"]
    table = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    from_idx, to_idx, eps = (table.column(c).to_pylist() for c in cols)
    return from_idx, to_idx, eps


# ------------------------------------------------------------------ upstream LeRobot → v3.0


Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""A per-episode ``(state, action) → (state, action)`` value transform of a reader."""


def _value_type(typ: pa.DataType) -> str:
    """The numpy dtype name of a column's values (of its items, for a list column)."""
    if pa.types.is_list(typ) or pa.types.is_fixed_size_list(typ):
        typ = typ.value_type
    return np.dtype(typ.to_pandas_dtype()).name


def _cast_type(typ: pa.DataType, dtype: str) -> pa.DataType:
    """``typ`` with ``dtype`` values."""
    value = pa.from_numpy_dtype(np.dtype(dtype))
    if pa.types.is_fixed_size_list(typ):
        return pa.list_(value, typ.list_size)
    return pa.list_(value) if pa.types.is_list(typ) else value


def _rewrite_episode(
    src: Path,
    dst: Path,
    columns: list[str],
    episode: int,
    offset: int,
    transform: Transform | None,
    add_frame_index: bool,
    dtypes: dict[str, str],
) -> dict[str, np.ndarray]:
    """Write one episode with ``columns``, renumbered and transformed (``add_frame_index``: a
    ``frame_index`` column is added), state/action cast to ``dtypes`` (the ``info.json``
    ones); returns the written state/action when ``transform`` changed them (else empty)."""
    table = pq.read_table(
        src, columns=[c for c in columns if c != "frame_index" or not add_frame_index]
    )
    n = table.num_rows
    for key in STATE_ACTION:
        typ = table.schema.field(key).type if key in table.column_names else None
        if typ is not None and _value_type(typ) != dtypes[key]:
            column = table.column(key).cast(_cast_type(typ, dtypes[key]))
            table = table.set_column(table.column_names.index(key), key, column)
    if add_frame_index:
        table = table.append_column("frame_index", pa.array(np.arange(n, dtype=np.int64)))
    replace: dict[str, pa.Array] = {
        "episode_index": pa.array(np.full(n, episode, dtype=np.int64)),
        "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
    }
    out: dict[str, np.ndarray] = {}
    if transform is not None:
        cols = read_columns(src, STATE_ACTION)
        state, action = transform(cols["observation.state"], cols["action"])
        for key, arr in (("observation.state", state), ("action", action)):
            arr = np.ascontiguousarray(arr, dtype=dtypes[key])
            replace[key] = pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1)), arr.shape[1]
            )
            out[key] = arr
    for name, arr in replace.items():
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, arr)
    pq.write_table(table, dst)
    return out


def _feature_stats(arr: np.ndarray) -> dict[str, list]:
    from lerobot.datasets.compute_stats import get_feature_stats

    stats = get_feature_stats(arr.astype(np.float64), axis=0, keepdims=False)
    return {k: np.asarray(v).reshape(-1).tolist() for k, v in stats.items()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def video_keys(info: dict) -> list[str]:
    """The video features of an ``info.json``, in order."""
    return [k for k, v in info["features"].items() if v["dtype"] == "video"]


def v2x_files(src: Path, info: dict, ep: int) -> tuple[Path, dict[str, Path]]:
    """The data parquet and the per-camera mp4s of episode ``ep`` of the v2.x dataset ``src``."""
    chunk = ep // info["chunks_size"]
    data = src / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    videos = {
        key: src / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep)
        for key in video_keys(info)
    }
    return data, videos


def check_info_totals(src: Path, episodes: int, frames: int | None) -> dict:
    """``meta/info.json`` of ``src``; its totals must be the pinned ones."""
    info = json.loads((src / "meta" / "info.json").read_text())
    if (info["total_episodes"], info["total_frames"]) != (episodes, frames):
        raise ValueError(
            f"{src}: {info['total_episodes']} episodes / {info['total_frames']} frames, "
            f"pinned {episodes} / {frames}"
        )
    return info


def _meta_from_data(src: Path, info: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    """``episodes.jsonl`` and ``episodes_stats.jsonl`` entries rebuilt from the data files, for
    an upstream whose meta lists fewer episodes than its data holds."""
    tasks = {t["task_index"]: t["task"] for t in read_jsonl(src / "meta" / "tasks.jsonl")}
    numeric = [k for k, v in info["features"].items() if v["dtype"] not in ("video", "image")]
    episodes, stats = {}, {}
    for path in sorted((src / "data").glob("*/episode_*.parquet")):
        index = int(path.stem.split("_")[-1])
        cols = read_columns(path, numeric)
        n = len(cols[numeric[0]])
        task_ids = sorted({int(t) for t in cols["task_index"].reshape(-1)})
        episodes[index] = {
            "episode_index": index,
            "tasks": [tasks[t] for t in task_ids],
            "length": n,
        }
        stats[index] = {
            "episode_index": index,
            "stats": {k: _feature_stats(v.reshape(n, -1)) for k, v in cols.items()},
        }
    return episodes, stats


def _stage_v2x(
    src: Path,
    stage: Path,
    keep: list[int],
    keep_columns: Sequence[str],
    transform: Transform | None,
    rebuild_meta: bool,
) -> None:
    """A v2.x view of ``src`` with only the ``keep`` episodes (renumbered in order), the
    ``keep_columns`` data columns and ``transform`` applied. Unchanged episode files are
    hard-linked. A v2.0 source has one ``meta/stats.json`` instead of per-episode stats; it is
    copied. An upstream without ``frame_index`` gets it (``arange`` per episode) when
    ``keep_columns`` asks for it. State/action are cast to their ``info.json`` dtype (RoboCOIN
    leaves mix float32 and float64 episodes, which the v3.0 converter rejects). With
    ``transform``, the state/action shapes in ``info.json`` (names dropped when the width
    changes) and their stats come from the transformed data. ``rebuild_meta``: episodes
    come from the data files, not ``meta/episodes.jsonl``."""
    info = json.loads((src / "meta" / "info.json").read_text())
    stats_path = src / "meta" / "episodes_stats.jsonl"
    if rebuild_meta:
        episodes, ep_stats = _meta_from_data(src, info)
    else:
        episodes = {e["episode_index"]: e for e in read_jsonl(src / "meta" / "episodes.jsonl")}
        ep_stats = (
            {e["episode_index"]: e for e in read_jsonl(stats_path)} if stats_path.exists() else {}
        )
    cameras = video_keys(info)
    data_cols = [k for k, v in info["features"].items() if v["dtype"] != "video"]
    add_frame_index = "frame_index" in keep_columns and "frame_index" not in data_cols
    missing = set(keep_columns) - set(data_cols) - {"frame_index"}
    if missing:
        raise KeyError(f"{src}: keep_columns not in the upstream features: {sorted(missing)}")
    rewrite = (
        transform is not None or add_frame_index or any(c not in keep_columns for c in data_cols)
    )
    chunks = info["chunks_size"]
    dtypes = {k: info["features"][k]["dtype"] for k in STATE_ACTION if k in info["features"]}

    offset = 0
    transformed: dict[int, dict[str, np.ndarray]] = {}
    for new, old in enumerate(keep):
        src_data, src_videos = v2x_files(src, info, old)
        dst_data, dst_videos = v2x_files(stage, info, new)
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        schema = pq.read_schema(src_data)
        stored = {k: _value_type(schema.field(k).type) for k in dtypes if k in schema.names}
        if new == old and not rewrite and stored.items() <= dtypes.items():
            link_tree(src_data, dst_data)
        else:
            transformed[new] = _rewrite_episode(
                src_data,
                dst_data,
                list(keep_columns),
                new,
                offset,
                transform,
                add_frame_index,
                dtypes,
            )
        offset += episodes[old]["length"]
        for key in cameras:
            dst_videos[key].parent.mkdir(parents=True, exist_ok=True)
            link_tree(src_videos[key], dst_videos[key])

    info["features"] = {
        k: v for k, v in info["features"].items() if k in keep_columns or k in cameras
    }
    if add_frame_index:
        info["features"]["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    if transform is not None:
        for key, arr in transformed[0].items():
            feature = info["features"][key]
            if feature["shape"] != [arr.shape[1]]:
                info["features"][key] = {**feature, "shape": [arr.shape[1]], "names": None}
    info["total_episodes"] = len(keep)
    info["total_frames"] = sum(episodes[e]["length"] for e in keep)
    info["splits"] = {"train": f"0:{len(keep)}"}
    if "total_videos" in info:
        info["total_videos"] = len(keep) * len(cameras)
    if "total_chunks" in info:
        info["total_chunks"] = -(-len(keep) // chunks)
    (stage / "meta").mkdir(parents=True, exist_ok=True)
    (stage / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(src / "meta" / "tasks.jsonl", stage / "meta" / "tasks.jsonl")
    ep_lines = [json.dumps({**episodes[old], "episode_index": new}) for new, old in enumerate(keep)]
    (stage / "meta" / "episodes.jsonl").write_text("".join(f"{line}\n" for line in ep_lines))
    if ep_stats:
        stats_lines = []
        for new, old in enumerate(keep):
            stats = {k: v for k, v in ep_stats[old]["stats"].items() if k in info["features"]}
            stats |= {k: _feature_stats(v) for k, v in transformed.get(new, {}).items()}
            stats_lines.append(json.dumps({"episode_index": new, "stats": stats}))
        (stage / "meta" / "episodes_stats.jsonl").write_text(
            "".join(f"{line}\n" for line in stats_lines)
        )
    else:
        stats = json.loads((src / "meta" / "stats.json").read_text())
        if transform is not None:
            for key in STATE_ACTION:
                stats[key] = _feature_stats(np.concatenate([t[key] for t in transformed.values()]))
        (stage / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))


def v2x_lengths(src: Path, rebuild_meta: bool = False) -> dict[int, int]:
    """Episode index → length of a v2.x dataset, in meta order (or from the data files)."""
    if rebuild_meta:
        return {
            int(p.stem.split("_")[-1]): pq.read_metadata(p).num_rows
            for p in sorted((src / "data").glob("*/episode_*.parquet"))
        }
    return {e["episode_index"]: e["length"] for e in read_jsonl(src / "meta" / "episodes.jsonl")}


def pinned_lengths(
    src: Path, episodes: int, frames: int | None, rebuild_meta: bool = False
) -> dict[int, int]:
    """``v2x_lengths`` of ``src``; their totals and (unless ``rebuild_meta``) the ``info.json``
    totals must be the pinned ones."""
    lengths = v2x_lengths(src, rebuild_meta)
    if (len(lengths), sum(lengths.values())) != (episodes, frames):
        raise ValueError(
            f"{src}: {len(lengths)} episodes / {sum(lengths.values())} frames in the meta, "
            f"pinned {episodes} / {frames}"
        )
    if not rebuild_meta:
        check_info_totals(src, episodes, frames)
    return lengths


def convert_v2x(
    src: Path,
    out: Path,
    keep: list[int],
    keep_columns: Sequence[str],
    transform: Transform | None = None,
    rebuild_meta: bool = False,
) -> None:
    """Convert the ``keep`` episodes of the LeRobot v2.x dataset ``src`` to v3.0 at ``out``.

    v2.1 (per-episode stats): the env's lerobot converter (data and videos concatenated into
    files, stats aggregated). v2.0: ``build_v30.convert`` with hard-linked data (per-episode
    files kept, meta rebuilt, ``stats.json`` copied). The staged view
    ``.v2x_stage/<collection>/<leaf>`` is deleted after either path.
    """
    from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
    from lerobot.scripts import convert_dataset_v21_to_v30 as conv

    stage = out.parent.parent / ".v2x_stage" / out.parent.name / out.name
    for path in (stage, out):
        if path.exists():
            shutil.rmtree(path)
    _stage_v2x(src, stage, keep, keep_columns, transform, rebuild_meta)
    if not (stage / "meta" / "episodes_stats.jsonl").exists():
        from build_v30 import convert as convert_v20

        convert_v20(stage, out, link_data=True)
        shutil.rmtree(stage)
        return
    conv.convert_info(stage, out, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    conv.convert_tasks(stage, out)
    episodes_meta = conv.convert_data(stage, out, DEFAULT_DATA_FILE_SIZE_IN_MB)
    videos_meta = conv.convert_videos(stage, out, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    conv.convert_episodes_metadata(stage, out, episodes_meta, videos_meta)
    shutil.rmtree(stage)


def select_v30(src: Path, out: Path, keep: Sequence[int] | None) -> tuple[int, int]:
    """The ``keep`` episodes (strictly increasing; None: all) of the LeRobot v3.0 dataset ``src``
    as a leaf at ``out``, values unchanged. Data parquets and ``meta/episodes`` are rewritten for
    them (renumbered in order; all episodes: identical values); ``tasks.parquet`` and
    ``stats.json`` are copied; video files are hard-linked unchanged, each episode keeps its video
    file and timestamps. Returns the leaf's episode and frame totals."""
    import pyarrow.compute as pc

    info = json.loads((src / "meta" / "info.json").read_text())
    files = sorted((src / "meta" / "episodes").glob("**/*.parquet"))
    episodes = pa.concat_tables([pq.read_table(f) for f in files])
    upstream = episodes.column("episode_index").to_numpy()
    if not np.array_equal(upstream, np.arange(len(upstream))):
        raise ValueError(f"{src}: meta/episodes is not contiguous from episode 0")
    kept = np.array(keep if keep is not None else upstream, dtype=np.int64)
    if not np.all(np.diff(kept) > 0):
        raise ValueError(f"{src}: the kept episodes are not strictly increasing")
    episodes = episodes.take(pa.array(kept))
    lengths = episodes.column("length").to_numpy()

    if out.exists():
        shutil.rmtree(out)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    new_index = np.full(len(upstream), -1, dtype=np.int64)
    new_index[kept] = np.arange(len(kept))
    to_idx = np.cumsum(lengths)
    from_idx = to_idx - lengths
    offset = 0
    data = zip(
        episodes.column("data/chunk_index").to_pylist(),
        episodes.column("data/file_index").to_pylist(),
    )
    for chunk, file in dict.fromkeys(data):
        rel = info["data_path"].format(chunk_index=chunk, file_index=file)
        table = pq.read_table(src / rel)
        old = table.column("episode_index").to_numpy()
        table = table.filter(pa.array(new_index[old] >= 0))
        new = new_index[table.column("episode_index").to_numpy()]
        if not np.all(np.diff(new) >= 0):
            raise ValueError(f"{src / rel}: episodes out of order")
        n = table.num_rows
        for col, arr in (
            ("episode_index", new),
            ("index", np.arange(offset, offset + n, dtype=np.int64)),
        ):
            i = table.column_names.index(col)
            table = table.set_column(i, col, pa.array(arr, type=table.schema.field(col).type))
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / rel)
        offset += n
    if offset != to_idx[-1]:
        raise ValueError(f"{src}: data holds {offset} rows of the kept episodes, meta {to_idx[-1]}")

    replace = {
        "episode_index": np.arange(len(kept), dtype=np.int64),
        "dataset_from_index": from_idx,
        "dataset_to_index": to_idx,
        "meta/episodes/chunk_index": np.zeros(len(kept), dtype=np.int64),
        "meta/episodes/file_index": np.zeros(len(kept), dtype=np.int64),
    }
    for col, arr in replace.items():
        i = episodes.column_names.index(col)
        episodes = episodes.set_column(
            i, col, pc.cast(pa.array(arr), episodes.schema.field(col).type)
        )
    pq.write_table(episodes, out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    for key in video_keys(info):
        video_files = zip(
            episodes.column(f"videos/{key}/chunk_index").to_pylist(),
            episodes.column(f"videos/{key}/file_index").to_pylist(),
        )
        for chunk, file in dict.fromkeys(video_files):
            rel = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            link_tree(src / rel, out / rel)
    for meta in ("tasks.parquet", "stats.json"):
        shutil.copy(src / "meta" / meta, out / "meta" / meta)
    info["total_episodes"] = len(kept)
    info["total_frames"] = int(to_idx[-1])
    info["splits"] = {"train": f"0:{len(kept)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return len(kept), int(to_idx[-1])


def read_v2x_episodes(src: Path) -> Iterable[Episode]:
    """The episodes of a LeRobot v2.x dataset in meta order (``meta/episodes.jsonl``, never a
    data glob); a data file whose length differs from the meta raises."""
    info = json.loads((src / "meta" / "info.json").read_text())
    chunks = info["chunks_size"]
    for ep in read_jsonl(src / "meta" / "episodes.jsonl"):
        index = ep["episode_index"]
        path = src / info["data_path"].format(episode_chunk=index // chunks, episode_index=index)
        cols = read_columns(path, STATE_ACTION)
        if len(cols["action"]) != ep["length"]:
            raise ValueError(f"{path}: {len(cols['action'])} rows, meta says {ep['length']}")
        yield Episode(index, cols["observation.state"], cols["action"], "\n".join(ep["tasks"]))


# ------------------------------------------------------------------ writer


def list_column(arr: np.ndarray) -> pa.Array:
    flat = pa.array(np.ascontiguousarray(arr).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, int(np.prod(arr.shape[1:])))


def _feature(arr: np.ndarray) -> dict:
    return {"dtype": str(arr.dtype), "shape": list(arr.shape[1:]) or [1], "names": None}


def _encode(frames: np.ndarray, path: Path, fps: int) -> None:
    from lerobot.datasets.video_utils import encode_video_frames
    from PIL import Image

    with tempfile.TemporaryDirectory(dir=path.parent) as imgs:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(imgs) / f"frame-{i:06d}.png")
        encode_video_frames(imgs, path, fps, **ENCODER)


class _VideoFiles:
    """Per camera: joins per-episode mp4s into ``videos/<key>/chunk-*/file-*.mp4``
    (``video.join_mp4s``: frame ``k`` of a file at ``k / fps``). A file holds one stream
    format; an episode in another format (RoboMIND mixes 1280x720 and 640x480 cameras)
    starts a new file."""

    def __init__(self, out: Path, key: str, file_mb: int, fps: int) -> None:
        self.out, self.key, self.file_mb, self.fps = out, key, file_mb, fps
        self.pending: list[Path] = []
        self.format: tuple | None = None
        self.size = 0.0
        self.frames = 0
        self.file = 0

    def add(self, mp4: Path) -> dict:
        from .video import stream_format

        size = mp4.stat().st_size / 1024**2
        fmt, frames = stream_format(mp4)
        if self.pending and (self.size + size >= self.file_mb or fmt != self.format):
            self.flush()
        rec = {
            f"videos/{self.key}/chunk_index": self.file // 1000,
            f"videos/{self.key}/file_index": self.file % 1000,
            f"videos/{self.key}/from_timestamp": self.frames / self.fps,
            f"videos/{self.key}/to_timestamp": (self.frames + frames) / self.fps,
        }
        self.pending.append(mp4)
        self.format = fmt
        self.size += size
        self.frames += frames
        return rec

    def flush(self) -> None:
        from lerobot.datasets.utils import DEFAULT_VIDEO_PATH

        from .video import join_mp4s

        if not self.pending:
            return
        dst = self.out / DEFAULT_VIDEO_PATH.format(
            video_key=self.key, chunk_index=self.file // 1000, file_index=self.file % 1000
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        join_mp4s(self.pending, dst, self.fps)
        self.pending, self.size, self.frames = [], 0.0, 0
        self.file += 1


def write_leaf(
    episodes: Iterable[Episode],
    out: Path,
    fps: int,
    robot_type: str,
    data_file_mb: int = DATA_FILE_MB,
) -> None:
    """Write ``episodes`` (in order, renumbered from 0) as a v3.0 leaf at ``out``: data parquets,
    ``meta/{info.json, episodes/, tasks.parquet, stats.json}`` and videos."""
    import pandas as pd
    from datasets import Dataset
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.io_utils import write_episodes, write_info, write_stats, write_tasks
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
    from lerobot.datasets.video_utils import get_video_info

    if out.exists():
        shutil.rmtree(out)
    work = out / ".episode_videos"
    work.mkdir(parents=True)
    tasks: dict[str, int] = {}
    videos: dict[str, _VideoFiles] = {}
    features: dict[str, dict] = {}
    records: list[dict] = []
    tables: list[pa.Table] = []
    pending_mb = 0.0
    data_file = 0
    total = 0
    stats_parts: dict[str, list[np.ndarray]] = {k: [] for k in STATE_ACTION}
    queue: deque[tuple[dict, list[tuple[str, Path, Future | None]]]] = deque()

    def add_videos(rec: dict, jobs: list[tuple[str, Path, Future | None]]) -> None:
        for key, mp4, encoding in jobs:
            if encoding is not None:
                encoding.result()
            videos.setdefault(key, _VideoFiles(out, key, VIDEO_FILE_MB, fps))
            rec |= videos[key].add(mp4)
        records.append(rec)

    def flush_data() -> None:
        nonlocal tables, pending_mb, data_file
        if tables:
            path = out / DEFAULT_DATA_PATH.format(
                chunk_index=data_file // 1000, file_index=data_file % 1000
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(tables), path)
            tables, pending_mb = [], 0.0
            data_file += 1

    with ThreadPoolExecutor(ENCODE_WORKERS) as pool:
        for new, ep in enumerate(episodes):
            n = len(ep.state)
            arrays = {"observation.state": ep.state, "action": ep.action, **ep.extra}
            for key, arr in arrays.items():
                if len(arr) != n:
                    raise ValueError(
                        f"episode {ep.index}: {key} has {len(arr)} rows, state has {n}"
                    )
                features.setdefault(key, _feature(arr))
            task_index = tasks.setdefault(ep.task, len(tasks))
            if ep.frame_tasks is not None:
                if len(ep.frame_tasks) != n:
                    raise ValueError(
                        f"episode {ep.index}: {len(ep.frame_tasks)} frame tasks, {n} rows"
                    )
                frame_task_index = [tasks.setdefault(t, len(tasks)) for t in ep.frame_tasks]
            else:
                frame_task_index = [task_index] * n
            index_cols = {
                "timestamp": (np.arange(n) / fps).astype(np.float32),
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, new, dtype=np.int64),
                "index": np.arange(total, total + n, dtype=np.int64),
                "task_index": np.asarray(frame_task_index, dtype=np.int64),
            }
            table = pa.table(
                {
                    **{k: list_column(v) for k, v in arrays.items()},
                    **{k: pa.array(v) for k, v in index_cols.items()},
                }
            )
            if tables and pending_mb + table.nbytes / 1024**2 >= data_file_mb:
                flush_data()
            rec: dict = {
                "episode_index": new,
                "tasks": [ep.task],
                "length": n,
                "data/chunk_index": data_file // 1000,
                "data/file_index": data_file % 1000,
                "dataset_from_index": total,
                "dataset_to_index": total + n,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            tables.append(table)
            pending_mb += table.nbytes / 1024**2
            jobs: list[tuple[str, Path, Future | None]] = []
            for key in sorted({*ep.frames, *ep.video_src}):
                if key in ep.frames:
                    mp4 = work / key / f"episode_{new:06d}.mp4"
                    mp4.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append((key, mp4, pool.submit(_encode, ep.frames[key], mp4, fps)))
                else:
                    jobs.append((key, ep.video_src[key], None))
            queue.append((rec, jobs))
            while len(queue) > ENCODE_WORKERS:
                add_videos(*queue.popleft())
            for k in STATE_ACTION:
                stats_parts[k].append(arrays[k])
            total += n
        while queue:
            add_videos(*queue.popleft())
    flush_data()
    for v in videos.values():
        v.flush()
    shutil.rmtree(work)

    for key in videos:
        first = out / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
        video_info = get_video_info(first)
        features[key] = {
            "dtype": "video",
            "shape": [video_info["video.height"], video_info["video.width"], 3],
            "names": ["height", "width", "channels"],
            "info": video_info,
        }
    for k, v in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        features[k] = {"dtype": v, "shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(records),
        "total_frames": total,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": data_file_mb,
        "video_files_size_in_mb": VIDEO_FILE_MB,
        "fps": fps,
        "splits": {"train": f"0:{len(records)}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH if videos else None,
        "features": features,
    }
    write_info(info, out)
    write_tasks(
        pd.DataFrame(
            {"task_index": list(tasks.values())}, index=pd.Index(list(tasks), name="task")
        ),
        out,
    )
    write_episodes(Dataset.from_list(records), out)
    write_stats(
        {
            k: get_feature_stats(np.concatenate(v), axis=0, keepdims=False)
            for k, v in stats_parts.items()
        },
        out,
    )
