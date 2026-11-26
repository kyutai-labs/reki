"""Normalization stats ops, all over KEPT frames (``combined_dataset.kept_frames``, the loader's
own selection: dropped episodes, keep ranges, subtask intervals); a leaf without kept frames
(every episode dropped) gets the stats of all its frames and stays out of pooled bands, so that
it loads (the loader samples nothing from it). A dataset's recipe is an
ordered list of ops, each called with the registry name, the datasets root, its leaves and
whether this is a ``--subset`` build.

Every op writes the state/action stats of a leaf with ``leaf_stats``: ``min/max/mean/std/count``
over the kept frames (mean and std two-pass in float64) and ``q01/q99`` = the leaf's own 1/99
quantiles. ``qk01/qk99`` is the band training normalizes with; no other key is written.
Training reads, from ``stats.json``: ``qk01/qk99`` (normalization), ``max`` (gripper-unit leaves,
``apply_gripper_unit``) and action ``mean/std`` (``drop_static_wrist_camera``); from the sidecars:
the same keys of ``norm_stats_euler*.json`` and the ego wrist ``qk01/qk99``.

- ``kept_qk``: per leaf, the 1/99 quantiles by lerobot's ``get_feature_stats`` (``exact``:
  ``np.quantile``) as ``q01/q99`` and ``qk01/qk99``, replacing the state/action stats.
- ``global_qk``: one ``np.percentile`` 1/99 band pooled over the leaves (``pool``: over the
  leaves of these registry datasets, e.g. both ABC stations), written as ``qk01/qk99`` into each
  (``pool_dims``: only the first N dims; ``clip_min`` on ``clip_dims``; ``zero_blocks``: per
  ``[lo, hi)`` action block, the rows where the block is all zero (untracked ego hands) are
  left out of that block's band; ``chunk_size``: the kept frames also drop the frames whose
  training chunk has no tracked hand, as the loader does for egocentric leaves).
- ``euler``: the ``norm_stats_euler.json`` sidecar of an axis-angle dataset; ``pooled`` stamps
  one pooled band of the axis-angle features into every leaf, except ``keep_dims`` and the
  dataset's ``gripper_unit_index``, which stay per leaf.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np

from ..recipe import StatsOp
from .v30 import STATE_ACTION, episode_bounds, read_leaf

logger = logging.getLogger(__name__)

LOOKUP = "subtask_lookup.jsonl"
"""The lookup ``configs/cpt.yaml`` reads (``subtask_lookup_filename``)."""


def dropped(leaf: Path) -> set[int]:
    """The episodes of ``meta/dropped_episodes.json`` (none without the file)."""
    path = leaf / "meta" / "dropped_episodes.json"
    return set(json.loads(path.read_text())) if path.exists() else set()


def kept_mask(leaf: Path, chunk_size: int | None = None) -> np.ndarray:
    """Per frame of the leaf: whether the loader samples it. ``chunk_size`` (egocentric leaves,
    the training chunk): also the loader's exclusion of frames whose chunk has no tracked hand."""
    from reki.data.robotics.combined_dataset import kept_frames
    from reki.data.robotics.subtasks import load_subtask_lookup
    from reki.data.robotics.transforms import untracked_frames

    lookup = load_subtask_lookup(leaf, LOOKUP)
    bounds = episode_bounds(leaf)
    untracked = None
    if chunk_size is not None:
        untracked = untracked_frames(read_leaf(leaf, ("action",))["action"])
    kept = kept_frames(
        bounds,
        dropped(leaf),
        lookup.keep_ranges,
        lookup.intervals,
        False,
        untracked,
        chunk_size or 1,
    )
    keep = np.zeros(max(bounds[1], default=0), dtype=bool)
    keep[kept] = True
    return keep


def kept_arrays(leaf: Path, columns: Sequence[str] = STATE_ACTION) -> dict[str, np.ndarray]:
    """float32 ``columns`` over the kept frames of a leaf."""
    arrays = read_leaf(leaf, columns)
    keep = kept_mask(leaf)
    return {k: v.astype(np.float32)[keep] for k, v in arrays.items()}


def leaf_arrays(
    leaf: Path, columns: Sequence[str] = STATE_ACTION, chunk_size: int | None = None
) -> dict[str, np.ndarray]:
    """``kept_arrays``; for a leaf without kept frames, all its frames:
    the loader samples nothing from it, its stats only keep it loadable."""
    arrays = read_leaf(leaf, columns)
    keep = kept_mask(leaf, chunk_size)
    if not keep.any():
        keep[:] = True
    return {k: v.astype(np.float32)[keep] for k, v in arrays.items()}


def _with_kept(leaves: list[Path], chunk_size: int | None = None) -> list[Path]:
    """The leaves with kept frames: only these enter a pooled band."""
    return [leaf for leaf in leaves if kept_mask(leaf, chunk_size).any()]


def _load(leaf: Path, name: str = "stats.json") -> dict:
    return json.loads((leaf / "meta" / name).read_text())


def _dump(leaf: Path, stats: dict, name: str = "stats.json") -> None:
    (leaf / "meta" / name).write_text(json.dumps(stats, indent=4, sort_keys=True))


def _lists(stats: dict) -> dict:
    return {k: np.asarray(v).reshape(-1).tolist() for k, v in stats.items()}


def _log(leaf: Path, file: str, arrays: dict[str, np.ndarray], band: str) -> None:
    """The one log line per leaf of every op."""
    columns = "/".join("state" if c == "observation.state" else c for c in arrays)
    frames = len(next(iter(arrays.values())))
    logger.info("%s: %s %s over %d frames, qk01/qk99 %s", leaf.name, file, columns, frames, band)


def leaf_stats(data: np.ndarray, exact: bool = False) -> dict[str, list]:
    """``min/max/mean/std/count`` of ``data`` (rows = frames), mean and std two-pass in float64,
    and its 1/99 quantiles as ``q01/q99`` and ``qk01/qk99``: lerobot's ``get_feature_stats``
    estimator (5000-bin histograms) or, with ``exact``, ``np.quantile``."""
    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats

    if exact:
        q01, q99 = np.quantile(data, 0.01, axis=0), np.quantile(data, 0.99, axis=0)
    else:
        est = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        q01, q99 = est["q01"], est["q99"]
    wide = data.astype(np.float64)
    out = {
        "min": data.min(axis=0),
        "max": data.max(axis=0),
        "mean": wide.mean(axis=0),
        "std": wide.std(axis=0),
        "count": np.array([len(data)]),
        "q01": q01,
        "q99": q99,
        "qk01": q01,
        "qk99": q99,
    }
    return _lists(out)


def kept_qk(name: str, root: Path, leaves: list[Path], subset: bool, exact: bool = False) -> None:
    """Per-leaf state/action stats over kept frames."""
    for leaf in leaves:
        stats = _load(leaf)
        arrays = leaf_arrays(leaf)
        for key, data in arrays.items():
            stats[key] = leaf_stats(data, exact)
        _dump(leaf, stats)
        _log(leaf, "stats.json", arrays, "per leaf")


def global_qk(
    name: str,
    root: Path,
    leaves: list[Path],
    subset: bool,
    columns: Sequence[str] = STATE_ACTION,
    pool: Sequence[str] | None = None,
    pool_dims: int | None = None,
    clip_min: float | None = None,
    clip_dims: Sequence[int] | None = None,
    zero_blocks: Sequence[tuple[int, int]] = (),
    chunk_size: int | None = None,
) -> None:
    """One pooled ``qk01/qk99`` band over the kept frames of ``leaves`` (or of every leaf of the
    ``pool`` datasets, which must all be built), written into each of ``leaves`` only, over
    their own ``leaf_stats``. ``chunk_size``: kept frames as ``kept_mask`` with it."""
    sources = leaves
    if pool is not None:
        sources = pool_leaves(name, root, pool, subset)
    sources = _with_kept(sources, chunk_size)
    arrays = {
        leaf: leaf_arrays(leaf, columns, chunk_size) for leaf in dict.fromkeys(sources + leaves)
    }
    pooled: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for leaf in sources:
        for col, arr in arrays[leaf].items():
            if pool_dims is not None:
                if arr.shape[1] < pool_dims:
                    raise ValueError(f"{leaf}: {col} width {arr.shape[1]} < pool_dims {pool_dims}")
                arr = arr[:, :pool_dims]
            pooled[col].append(arr)
    band: dict[str, dict[str, list[float]]] = {}
    for col, parts in pooled.items():
        arr = np.concatenate(parts)
        if clip_min is not None:
            dims = list(clip_dims) if clip_dims is not None else slice(None)
            arr[:, dims] = np.clip(arr[:, dims], clip_min, None)
        qk01, qk99 = np.percentile(arr, 1, axis=0), np.percentile(arr, 99, axis=0)
        if col == "action":
            for lo, hi in zero_blocks:
                block = arr[arr[:, lo:hi].any(axis=1), lo:hi]
                qk01[lo:hi] = np.percentile(block, 1, axis=0)
                qk99[lo:hi] = np.percentile(block, 99, axis=0)
        band[col] = {"qk01": qk01.tolist(), "qk99": qk99.tolist()}
    for leaf in leaves:
        stats = _load(leaf)
        for col, new in band.items():
            cur = leaf_stats(arrays[leaf][col])
            for qk in ("qk01", "qk99"):
                cur[qk][: len(new[qk])] = new[qk]
            stats[col] = cur
        _dump(leaf, stats)
        pooled_over = f"pooled over {len(sources)} leaves of {list(pool or [name])}"
        _log(leaf, "stats.json", arrays[leaf], pooled_over)


def pool_leaves(name: str, root: Path, pool: Sequence[str], subset: bool) -> list[Path]:
    """The leaves of the ``pool`` datasets. Raises, with the commands that build them, when a
    pool dataset is not built under ``root``."""
    from ..catalog import pin, recipes

    by_name = {n: recipes()[n].reader.leaves(n, root, pin(n, subset)) for n in pool}
    missing = [
        n
        for n, leaves in by_name.items()
        if not leaves or not all((leaf / "meta" / "info.json").exists() for leaf in leaves)
    ]
    if missing:
        flags = f"{' --subset' if subset else ''} --root {root}"
        commands = "\n".join(
            f"  pixi run -e data python scripts/data/pretrain_data.py {stage} {n}{flags}"
            for n in missing
            for stage in ("download", "build", "curate")
        )
        raise FileNotFoundError(
            f"{name}: the stats band pools {list(pool)}; not built under {root}: {missing}. "
            f"Build and curate them first:\n{commands}"
        )
    return [leaf for leaves in by_name.values() for leaf in leaves]


def _euler_data(name: str, leaf: Path) -> dict[str, np.ndarray]:
    """Kept state/action of a leaf with each axis-angle rotation block in euler xyz."""
    from shared import axis_angle_to_euler, raw_rot_blocks

    from reki.data.robotics.registry import AXIS_ANGLE, DATASETS
    from reki.data.robotics.transforms import recenter_euler_roll, rot_blocks

    spec = DATASETS[name]
    blocks = rot_blocks(spec.embodiment)
    reps = {"action": spec.action_rot_rep, "observation.state": spec.state_rot_rep}
    slices = {"action": spec.action_slice, "observation.state": spec.state_slice}
    out = leaf_arrays(leaf)
    for key, data in out.items():
        if reps[key] != AXIS_ANGLE:
            continue
        raw = raw_rot_blocks(blocks, slices[key])
        out[key] = axis_angle_to_euler(data, raw)
        if spec.recenter_state_roll and key == "observation.state":
            recenter_euler_roll(out[key], raw)
    return out


def euler(
    name: str,
    root: Path,
    leaves: list[Path],
    subset: bool,
    pooled: bool = False,
    keep_dims: Sequence[int] = (),
) -> None:
    """The euler sidecar of every leaf; ``pooled``: one band for the axis-angle features."""
    from reki.data.robotics.registry import AXIS_ANGLE, DATASETS

    spec = DATASETS[name]
    sidecar = "norm_stats_euler_roll.json" if spec.recenter_state_roll else "norm_stats_euler.json"
    data = {leaf: _euler_data(name, leaf) for leaf in leaves}
    kept = _with_kept(leaves)
    per_leaf = {leaf: {k: leaf_stats(v) for k, v in d.items()} for leaf, d in data.items()}
    if pooled:
        per_dim = set(keep_dims) | (
            {spec.gripper_unit_index} if spec.gripper_unit_index is not None else set()
        )
        reps = {"action": spec.action_rot_rep, "observation.state": spec.state_rot_rep}
        for key, rep in reps.items():
            if rep != AXIS_ANGLE:
                continue
            band = leaf_stats(np.concatenate([data[leaf][key] for leaf in kept]))
            for stats in per_leaf.values():
                stamped = {f: list(v) for f, v in band.items()}
                for d in per_dim:
                    for f, v in stamped.items():
                        if d < len(v):
                            v[d] = stats[key][f][d]
                stats[key] = stamped
    for leaf, stats in per_leaf.items():
        _dump(leaf, stats, sidecar)
        _log(leaf, sidecar, data[leaf], f"pooled over {len(kept)} leaves" if pooled else "per leaf")


OPS: dict[str, Callable[..., None]] = {"kept_qk": kept_qk, "global_qk": global_qk, "euler": euler}


def op(kind: str, **kwargs: object) -> StatsOp:
    """A recipe step: ``op("global_qk", pool_dims=40)``."""
    return partial(OPS[kind], **kwargs)


def run(name: str, root: Path, leaves: list[Path], recipe: Sequence[StatsOp], subset: bool) -> None:
    """Apply a dataset's ordered stats recipe; the ``pool`` datasets of its steps are checked
    first, so that a missing one raises before any leaf is written."""
    for step in recipe:
        if isinstance(step, partial) and step.keywords.get("pool") is not None:
            pool_leaves(name, root, step.keywords["pool"], subset)
    for step in recipe:
        step(name, root, leaves, subset)
