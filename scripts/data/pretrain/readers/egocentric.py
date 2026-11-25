"""Shared pieces of the egocentric human-hand readers (egodex, egoverse, xperience): the hand-action
math, the reader base (``EgoReader``), and the ``ego_wrist`` stats op.

Stored layout of every ego leaf (the loader composes the camera-frame target at load time,
``transforms.wrist_to_camera_frame``):

- ``action`` [32]: per hand (left 0:16, right 16:32) the ABSOLUTE world wrist
  ``[translation(3), axis-angle(3)]`` and 10 eigengrasp coefficients; a frame where the
  hand is not tracked has its 16 columns at zero.
- ``observation.state`` [12]: the world camera pose ``[translation(3), axis-angle(3)]`` in
  0:6, zeros elsewhere (ego is stateless; the column carries the camera for the compose).
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from reki.data.robotics.normalization import EGO_WRIST_DIMS

from ..common import dedup, v30
from ..pins import Pin
from ..recipe import Reader, collection_dir

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

logger = logging.getLogger(__name__)

STATE_DIM = 12
VIDEO_KEY = "observation.images.camera_top"
ROBOT_TYPE = "human_hand"
FALLBACK_TASK = "manipulate the objects."
LEAVES = 96
"""Episodes go round-robin into ``LEAVES`` leaves ``<dataset>_{i:02d}``."""
PCA_PATH = Path(__file__).parent / "assets" / "eigengrasp_pca.npz"
"""The shared eigengrasp basis (10-component PCA of the 45-dim MANO finger pose)."""
HAND_BLOCKS = ((0, 16), (16, 32))
"""Each hand's action block; an untracked row is all zero."""
WRIST_DIMS = EGO_WRIST_DIMS
EGO_WRIST_MAX_ROWS = 4_000_000


def rotmat_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    """``[..., 3, 3]`` rotation matrices → ``[..., 3]`` axis-angle."""
    return Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(*rotmat.shape[:-2], 3)


def valid_so3(rotmat: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Per matrix of ``[..., 3, 3]``: a rotation (untracked mocap frames are all zero or, in
    xperience, all NaN: their NaN det fails the test)."""
    flat = np.asarray(rotmat, np.float64).reshape(-1, 3, 3)
    with np.errstate(invalid="ignore"):
        det = np.linalg.det(flat)
    orth = np.abs(np.einsum("nij,nkj->nik", flat, flat) - np.eye(3)).reshape(len(flat), -1).max(1)
    return ((np.abs(det - 1.0) < tol) & (orth < tol)).reshape(rotmat.shape[:-2])


@dataclass(frozen=True)
class Eigengrasps:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def load(cls, path: Path = PCA_PATH) -> Eigengrasps:
        d = np.load(path)
        return cls(d["mean"], d["components"])

    def transform(self, finger_aa: np.ndarray) -> np.ndarray:
        """``[N, 45]`` finger axis-angle → ``[N, 10]`` coefficients (float64)."""
        return (np.asarray(finger_aa, np.float64) - self.mean) @ self.components.T


def hand_columns(
    wrist_t: np.ndarray, wrist_aa: np.ndarray, eig: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """One hand's ``[N, 16]`` float32 action block; untracked frames are zero."""
    out = np.zeros((len(wrist_t), 16), np.float32)
    out[:, 0:3] = wrist_t.astype(np.float32)
    out[:, 3:6] = wrist_aa.astype(np.float32)
    out[:, 6:16] = eig.astype(np.float32)
    out[~valid] = 0.0
    return out


def camera_state(camera_t: np.ndarray, camera_aa: np.ndarray) -> np.ndarray:
    """``[N, 12]`` float32 state carrying the camera pose in 0:6."""
    out = np.zeros((len(camera_t), STATE_DIM), np.float32)
    out[:, 0:3] = camera_t.astype(np.float32)
    out[:, 3:6] = camera_aa.astype(np.float32)
    return out


def leaf_dirs(collection: Path, prefix: str) -> list[Path]:
    """The built leaves ``<prefix>_NN`` of a collection, sorted."""
    leaves = sorted(p for p in collection.glob(f"{prefix}_[0-9][0-9]") if p.is_dir())
    if not leaves:
        raise FileNotFoundError(f"{collection}: no {prefix}_NN leaves")
    return leaves


class EgoReader(Reader):
    """An ego dataset: leaves ``<collection>/<prefix>_NN``."""

    prefix: str

    def leaf_dir(self, name: str, root: Path, leaf: int) -> Path:
        return collection_dir(name, root) / f"{self.prefix}_{leaf:02d}"

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return leaf_dirs(collection_dir(name, root), self.prefix)

    def key_columns(self) -> dedup.Columns:
        """The camera translation only: the action holds GPU MANO fits, whose low bits differ
        across GPUs and software (``mano_fit``)."""
        return (0, 1, 2), ()

    def write_leaf(
        self,
        name: str,
        root: Path,
        leaf: int,
        episodes: Callable[[Path], Iterator[v30.Episode]],
        fps: int,
    ) -> None:
        """Leaf ``leaf`` from ``episodes(work)``, which writes its mp4s into ``work`` (removed
        after); no episode: no leaf."""
        out = self.leaf_dir(name, root, leaf)
        work = out.parent / f".{out.name}.mp4"
        work.mkdir(parents=True, exist_ok=True)
        it = episodes(work)
        first = next(it, None)
        if first is None:
            logger.info("%s: no episode", out.name)
        else:
            v30.write_leaf(itertools.chain([first], it), out, fps, ROBOT_TYPE)
            info = json.loads((out / "meta" / "info.json").read_text())
            logger.info(
                "%s: %d episodes / %d frames",
                out.name,
                info["total_episodes"],
                info["total_frames"],
            )
        shutil.rmtree(work)


# ------------------------------------------------------------------ stats


def _leaf_wrist_rows(
    leaf: Path, episodes: list[int], windows: int, window: int, pace: float
) -> dict[int, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Per episode of ``episodes``: the wrist rows of its ``windows`` composed chunks and their
    tracked masks (one leaf; a process-pool task)."""
    import torch

    from reki.data.robotics.transforms import (
        interpolate_wrist_chunk,
        tracked_hand_dims,
        wrist_to_camera_frame,
    )

    from ..common.stats import kept_mask
    from ..common.v30 import episode_bounds, read_leaf

    torch.set_num_threads(1)
    arrays = read_leaf(leaf)
    keep = kept_mask(leaf, window)
    from_idx, to_idx, _ = episode_bounds(leaf)
    out: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for j in episodes:
        f, t = from_idx[j], to_idx[j]
        action, state = arrays["action"][f:t], arrays["observation.state"][f:t]
        kept = np.flatnonzero(keep[f:t])
        chunks: list[torch.Tensor] = []
        if len(kept):
            positions = np.linspace(0, max(len(kept) - 2, 0), windows).astype(int)
            for s in kept[positions]:
                chunk = torch.tensor(action[s : min(s + window, t - f)])
                chunk = interpolate_wrist_chunk(chunk, pace)
                chunks.append(wrist_to_camera_frame(chunk, torch.tensor(state[s, :6])))
        out[j] = (
            [c[:, WRIST_DIMS].numpy() for c in chunks],
            [tracked_hand_dims(c)[:, WRIST_DIMS].numpy() for c in chunks],
        )
    return out


def _fps(name: str, leaves: list[Path]) -> int:
    """The one fps of the leaves."""
    fps = {json.loads((leaf / "meta" / "info.json").read_text())["fps"] for leaf in leaves}
    if len(fps) != 1:
        raise ValueError(f"{name}: leaves at several fps {sorted(fps)}")
    return fps.pop()


def ego_stats(name: str, root: Path, leaves: list[Path], subset: bool, train: TrainConfig) -> None:
    """``global_qk`` over the frames the loader keeps at the training chunk of ``train`` (a
    frame whose chunk has no tracked hand is not sampled), untracked hand rows out of the band."""
    from reki.data.robotics.factory import chunk_size_for

    from ..common.stats import global_qk

    window = chunk_size_for(train, _fps(name, leaves))
    global_qk(name, root, leaves, subset, zero_blocks=HAND_BLOCKS, chunk_size=window)


def ego_wrist(
    name: str,
    root: Path,
    leaves: list[Path],
    subset: bool,
    train: TrainConfig,
    windows: int,
    max_episodes: int | None = None,
    interleave: bool = False,
    workers: int = 16,
) -> None:
    """Write the camera-frame wrist band of the training horizon of ``train`` into every leaf,
    as ``meta/<ego_wrist_sidecar>`` (the loader stamps it over the 12 wrist dims of the action
    ``qk01/qk99``): the 1/99 percentiles of the chunks the loader composes. A chunk is the
    ``window = chunk_size_for(train, fps)`` frames from its start (cut at the episode end),
    paced by ``interpolate_wrist_chunk`` at ``train.action_pace[name]``, then
    ``wrist_to_camera_frame``.

    Per episode, ``windows`` chunk starts are spread evenly over its kept frames (as the loader
    keeps them at this window: a frame whose chunk has no tracked hand is not one). Episodes are
    taken in leaf order, or in build order (``interleave``: episode ``g`` of the dataset is
    episode ``g // n`` of leaf ``g % n``); ``max_episodes`` keeps every ``N // max_episodes``-th.
    The rows are capped at ``EGO_WRIST_MAX_ROWS`` by an even subsample. A hand's untracked rows
    (stored all zero) are left out of its band. The leaves are read by ``workers`` processes;
    the band does not depend on it.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from reki.data.robotics.factory import chunk_size_for
    from reki.data.robotics.normalization import ego_wrist_sidecar

    from ..common.v30 import episode_bounds

    leaf_fps = _fps(name, leaves)
    window = chunk_size_for(train, leaf_fps)
    pace = float(train.action_pace.get(name, 1.0))
    counts = [len(episode_bounds(leaf)[2]) for leaf in leaves]
    if interleave:
        order = [(i, j) for j in range(max(counts)) for i in range(len(leaves)) if j < counts[i]]
    else:
        order = [(i, j) for i in range(len(leaves)) for j in range(counts[i])]
    if max_episodes is not None and len(order) > max_episodes:
        order = order[:: max(1, len(order) // max_episodes)]
    wanted: dict[int, list[int]] = {}
    for i, j in order:
        wanted.setdefault(i, []).append(j)

    tasks = sorted(wanted)
    args = [(leaves[i], sorted(wanted[i]), windows, window, pace) for i in tasks]
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(min(workers, len(args)), mp_context=ctx) as ex:
            results = list(ex.map(_leaf_wrist_rows, *zip(*args)))
    else:
        results = [_leaf_wrist_rows(*a) for a in args]
    per_leaf = dict(zip(tasks, results))
    pooled = np.concatenate([r for i, j in order for r in per_leaf[i][j][0]])
    pooled_valid = np.concatenate([v for i, j in order for v in per_leaf[i][j][1]])
    if len(pooled) > EGO_WRIST_MAX_ROWS:
        pick = np.linspace(0, len(pooled) - 1, EGO_WRIST_MAX_ROWS).astype(int)
        pooled, pooled_valid = pooled[pick], pooled_valid[pick]
    qk01, qk99 = np.empty(len(WRIST_DIMS)), np.empty(len(WRIST_DIMS))
    for d in range(len(WRIST_DIMS)):
        column = pooled[pooled_valid[:, d], d]
        qk01[d], qk99[d] = np.percentile(column, 1), np.percentile(column, 99)
    sidecar = {
        "chunk_duration_s": train.chunk_duration_s,
        "chunk_size": train.chunk_size,
        "window": window,
        "fps": leaf_fps,
        "pace": pace,
        "sampling": {
            "windows": windows,
            "max_episodes": max_episodes,
            "interleave": interleave,
            "max_rows": EGO_WRIST_MAX_ROWS,
            "rows": len(pooled),
        },
        "dims": WRIST_DIMS,
        "qk01": qk01.tolist(),
        "qk99": qk99.tolist(),
    }
    file = ego_wrist_sidecar(train.chunk_duration_s, train.chunk_size, pace)
    for leaf in leaves:
        (leaf / "meta" / file).write_text(json.dumps(sidecar, indent=4))
