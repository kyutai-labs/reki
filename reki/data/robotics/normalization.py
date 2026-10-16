"""Per-leaf normalization stats, their guards, and the normalization itself.

Training normalizes each leaf with its own ``meta.stats``, prepared by ``prepare_leaf_stats``
(euler sidecar) and gathered to the model-visible dims by ``slice_leaf_stats``.
The checkpoint export builds the stats it ships with the same two functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_STATE

from reki.data.robotics.mirror import MirrorSpec
from reki.data.robotics.registry import AXIS_ANGLE
from reki.data.robotics.transforms import gather_with_pad


def quantile_keys(stats: dict[str, torch.Tensor]) -> tuple[str, str]:
    """The quantile band normalization reads: ``qk01/qk99``. Raises when it is
    missing."""
    if "qk01" not in stats or "qk99" not in stats:
        raise KeyError(
            f"stats have no qk01/qk99 band: {sorted(stats)}. Run "
            "scripts/data/augment_quantiles.py (see docs/data.md, Normalization)."
        )
    return "qk01", "qk99"


def normalize_feature(x: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Map ``x`` from the [qk01, qk99] band to [-1, 1], clamped."""
    lo, hi = quantile_keys(stats)
    q01 = stats[lo].to(dtype=x.dtype, device=x.device)
    q99 = stats[hi].to(dtype=x.dtype, device=x.device)
    denom = q99 - q01
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    return (2.0 * (x - q01) / denom - 1.0).clamp(-1.0, 1.0)


def normalize_sample(
    item: dict, stats: dict[str, dict[str, torch.Tensor]], raw: tuple[str, ...] = ()
) -> None:
    """Normalize the state and action of one sample in place, except the ``raw`` keys."""
    for key in (OBS_STATE, ACTION):
        if key in raw or key not in item or key not in stats:
            continue
        item[key] = normalize_feature(item[key], stats[key])


def prepare_leaf_stats(
    raw_stats: dict,
    *,
    root: Path,
    action_rot_rep: str | None,
    state_rot_rep: str | None,
    recenter_roll: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    """One leaf's ``meta.stats`` as training normalizes it, before mirroring and slicing: the
    euler sidecar replaces the stats of an axis-angle feature (the roll-recentered sidecar with
    ``recenter_roll``)."""
    stats = {k: {s: torch.as_tensor(v) for s, v in sv.items()} for k, sv in raw_stats.items()}
    aa_keys = [
        k for k, rep in ((ACTION, action_rot_rep), (OBS_STATE, state_rot_rep)) if rep == AXIS_ANGLE
    ]
    if aa_keys:
        euler = load_euler_stats(root, recenter_roll)
        for k in aa_keys:
            if k in euler:
                stats[k] = {s: torch.as_tensor(v) for s, v in euler[k].items()}
    return stats


def slice_leaf_stats(
    stats: dict[str, dict[str, torch.Tensor]],
    state_slice: list[int | None] | None,
    action_slice: list[int] | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Gather leaf stats to the model-visible state/action dims, in place. Shared by the plain
    and mirrored stats (so they cannot diverge) and by the checkpoint export."""
    if state_slice is not None and OBS_STATE in stats:
        stats[OBS_STATE] = {
            k: v if v.numel() == 1 else gather_with_pad(v, state_slice)
            for k, v in stats[OBS_STATE].items()
        }
    if action_slice is not None and ACTION in stats:
        stats[ACTION] = {
            k: v if v.numel() == 1 else v[action_slice] for k, v in stats[ACTION].items()
        }
    return stats


def load_euler_stats(root: Path, recenter_roll: bool = False) -> dict[str, dict[str, list]]:
    """Load the euler action/state stats sidecar of an axis-angle dataset.

    ``scripts/data/recompute_euler_stats.py`` writes it. With ``recenter_roll`` the file is
    ``norm_stats_euler_roll.json`` (state roll shifted by π), else ``norm_stats_euler.json``.
    Raises when the file is missing.
    """
    p = root / "meta" / ("norm_stats_euler_roll.json" if recenter_roll else "norm_stats_euler.json")
    if not p.exists():
        raise FileNotFoundError(
            f"unify_ee_euler_rotation needs euler stats at {p} (run "
            f"scripts/data/recompute_euler_stats.py "
            "for this dataset)"
        )
    return json.loads(p.read_text())


def mirror_stats_block(
    key: str, block: dict[str, torch.Tensor], spec: MirrorSpec
) -> dict[str, torch.Tensor]:
    """Mirror one feature's stats tensors. Features other than state and action are unchanged.

    State and action stats must have the spec's raw width, because their values are mirrored
    before the slice. Another width raises.
    """
    feature = {OBS_STATE: "state", ACTION: "action"}.get(key)
    per_dim = {k: v for k, v in block.items() if v.numel() > 1}
    if feature is None:
        return block
    expected = spec.dim(feature)
    wrong_width = [k for k, v in per_dim.items() if v.numel() != expected]
    if wrong_width:
        raise ValueError(
            f"mirroring {key!r} stats under spec {spec.name!r} needs {expected}-dim fields, got "
            f"{ {k: per_dim[k].numel() for k in wrong_width} } — a slice or rescale changed the "
            "width, so the values would be mirrored while these stats were not"
        )
    if not per_dim:
        return block
    mirrored = spec.mirror_stats({k: v.tolist() for k, v in per_dim.items()}, feature)
    out = {k: v for k, v in block.items() if v.numel() == 1}
    out.update({k: torch.as_tensor(v, dtype=block[k].dtype) for k, v in mirrored.items()})
    return out


def find_absolute_norm_violations(
    dataset_names: list[str],
    control_modes: list[str],
    per_ds_stats: list[dict[str, dict[str, torch.Tensor]]],
    cols: tuple[str, ...],
) -> list[str]:
    """Find absolute-control datasets whose qk01/qk99 band is missing or differs across leaves.

    Checks the sliced, model-visible dims. Per-leaf bands map one physical pose to different
    normalized values on different leaves, which corrupts the shared state-MLP input and the
    flow targets. Delta and relative datasets are exempt (deltas are centred near 0), and so
    are single-leaf datasets. Returns one message per violation (empty when clean)."""
    by_name: dict[str, list[int]] = {}
    for i, nm in enumerate(dataset_names):
        by_name.setdefault(nm, []).append(i)
    violations: list[str] = []
    for nm, idxs in by_name.items():
        if len(idxs) < 2 or "absolute" not in control_modes[idxs[0]]:
            continue
        for col in cols:
            mismatch = _qk_band_mismatch([per_ds_stats[i].get(col) for i in idxs])
            if mismatch == "missing":
                violations.append(f"{nm} [{col}]: missing global qk01/qk99 on some leaves")
            elif mismatch == "vary":
                violations.append(
                    f"{nm} [{col}]: qk01/qk99 vary across {len(idxs)} leaves (per-leaf, not global)"
                )
    return violations


def band_clipping_violation(
    name: str,
    root: Path,
    ds_stats: dict[str, dict[str, torch.Tensor]],
    cols: tuple[str, ...],
    rot_reps: dict[str, str | None] | None = None,
    *,
    ep_tol_frac: float = 0.02,
    clip_threshold: float = 0.05,
    max_rows: int = 50_000,
    max_files: int = 32,
) -> list[str]:
    """Report quantile bands that are lerobot's per-episode average and clip too many frames.

    ``aggregate_feature_stats`` merges min/max/mean/std exactly, but it merges quantiles by a
    count-weighted mean. A mean of quantiles is not a quantile: the band shrinks toward the
    centre when the episodes differ. Frames outside the band are clamped in the flow targets and
    in the state input. ``find_absolute_norm_violations`` does not catch this on a single leaf.

    A band is reported when it is within ``ep_tol_frac`` (of band width) of the per-episode
    average from ``meta/episodes`` AND more than ``clip_threshold`` of sampled frames fall
    outside it (a true 1/99 band leaves ~2%). Either check alone gives false positives.
    Axis-angle features are skipped: their band comes from the euler sidecar. Returns one
    message per reported band.
    """
    ep_dir, data_dir = root / "meta" / "episodes", root / "data"
    if not ep_dir.is_dir():
        return []
    ep_files = sorted(ep_dir.glob("**/*.parquet"))
    if not ep_files:
        return []
    import pyarrow.parquet as pq

    available = set(pq.read_schema(ep_files[0]).names)
    out: list[str] = []
    for col in cols:
        band = ds_stats.get(col, {})
        if "qk01" not in band or (rot_reps or {}).get(col) == AXIS_ANGLE:
            continue
        ep_cols = [f"stats/{col}/{k}" for k in ("q01", "q99", "count")]
        if not set(ep_cols) <= available:
            continue
        tbl = pq.read_table(ep_files, columns=ep_cols)
        ep_lo = np.asarray(tbl[ep_cols[0]].to_pylist(), dtype=np.float64)
        ep_hi = np.asarray(tbl[ep_cols[1]].to_pylist(), dtype=np.float64)
        cnt = np.asarray(
            [c[0] if isinstance(c, list) else c for c in tbl[ep_cols[2]].to_pylist()],
            dtype=np.float64,
        )
        lo = band["qk01"].to(torch.float64).numpy()
        hi = band["qk99"].to(torch.float64).numpy()
        if ep_lo.ndim != 2 or ep_lo.shape[1] != lo.shape[0] or cnt.shape[0] != ep_lo.shape[0]:
            continue
        avg_lo = (ep_lo * cnt[:, None]).sum(0) / cnt.sum()
        avg_hi = (ep_hi * cnt[:, None]).sum(0) / cnt.sum()
        width = np.maximum(hi - lo, 1e-12)
        if (
            not (np.abs(lo - avg_lo) / width < ep_tol_frac).all()
            or not (np.abs(hi - avg_hi) / width < ep_tol_frac).all()
        ):
            continue
        frac, n = _sampled_clip_fraction(data_dir, col, lo, hi, max_rows, max_files)
        if frac is None:
            continue
        worst = int(np.argmax(frac))
        if frac[worst] > clip_threshold:
            out.append(
                f"{name} [{col}]: qk01/qk99 is the per-episode AVERAGE, and clips "
                f"{frac[worst]:.1%} of frames on dim {worst} (band {lo[worst]:.2f}.."
                f"{hi[worst]:.2f}; ~2% expected), measured on {n} sampled frames"
            )
    return out


def _sampled_clip_fraction(
    data_dir: Path,
    col: str,
    lo: np.ndarray,
    hi: np.ndarray,
    max_rows: int,
    max_files: int,
) -> tuple[np.ndarray | None, int]:
    """Per-dim fraction of sampled frames outside ``[lo, hi]``, over up to ``max_files`` evenly
    spaced data files. Dims with no spread in the sample are reported as 0: a sliver band over a
    pad channel clips "everything" while costing nothing."""
    import pyarrow.parquet as pq

    files = sorted(data_dir.glob("**/*.parquet"))
    if not files:
        return None, 0
    step = max(1, len(files) // max_files)
    picked = files[::step][:max_files]
    rows: list[np.ndarray] = []
    n = 0
    for f in picked:
        try:
            batch = next(
                pq.ParquetFile(f).iter_batches(
                    batch_size=max_rows // len(picked) + 1, columns=[col]
                )
            )
        except (StopIteration, KeyError, OSError):
            continue
        arr = np.asarray(batch[col].to_pylist(), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != lo.shape[0]:
            return None, 0
        rows.append(arr)
        n += arr.shape[0]
        if n >= max_rows:
            break
    if not rows or n < 5_000:
        return None, 0
    data = np.concatenate(rows)
    live = np.ptp(data, axis=0) > 1e-6
    return np.where(live, ((data < lo) | (data > hi)).mean(0), 0.0), n


def check_norm_bands(
    band_violations: list[str],
    names: list[str],
    control_modes: list[str],
    per_ds_stats: list[dict[str, dict[str, torch.Tensor]]],
    cols: tuple[str, ...],
) -> None:
    """Raise when an absolute-control dataset's quantile band is a per-episode average
    (``band_violations``, from ``band_clipping_violation``) or differs across its leaves
    (``find_absolute_norm_violations``)."""
    if band_violations:
        raise RuntimeError(
            "These absolute-control bands are lerobot's mean of per-episode quantiles, not a "
            "global quantile, and clip far more than the ~2% of a real 1/99 band:\n  "
            + "\n  ".join(band_violations)
            + "\n\nFix: pixi run python scripts/data/augment_quantiles.py --leaves "
            "<collection root> --global (one repo: --leaf <dir> --overwrite)."
        )
    violations = find_absolute_norm_violations(names, control_modes, per_ds_stats, cols)
    if violations:
        raise RuntimeError(
            "Absolute-control datasets need the same qk01/qk99 in every leaf, but these differ "
            "per leaf (one pose would normalize to different values):\n  "
            + "\n  ".join(violations)
            + "\n\nFix: pixi run python scripts/data/augment_quantiles.py --leaves "
            "<collection root> --global (see docs/data.md, Normalization)."
        )


def _qk_band_mismatch(
    bands: list[dict[str, torch.Tensor] | None],
) -> Literal["missing", "vary"] | None:
    """Whether the qk01/qk99 bands of the leaves are absent somewhere, differ, or agree (None)."""
    present = [b for b in bands if b is not None and "qk01" in b and "qk99" in b]
    if len(present) != len(bands):
        return "missing"
    if not all(
        torch.equal(b["qk01"], present[0]["qk01"]) and torch.equal(b["qk99"], present[0]["qk99"])
        for b in present[1:]
    ):
        return "vary"
    return None


def qk_identical_across(
    per_ds: list[dict[str, dict[str, torch.Tensor]]], cols: tuple[str, ...]
) -> bool:
    """True when every leaf carries the same qk01/qk99 on every column (a stamped global band)."""
    for col in cols:
        if _qk_band_mismatch([st.get(col) for st in per_ds]) is not None:
            return False
    return True


def with_qk_as_q(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """Copy qk* over q* for an external adapter.

    Foreign normalizers read q01/q99 (lerobot's QUANTILES mode), and ``normalize_feature`` reads
    qk01/qk99. Without the copy, the two would normalize with different bands.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, st in stats.items():
        new = dict(st)
        for q in ("01", "10", "50", "90", "99"):
            if f"qk{q}" in st:
                new[f"q{q}"] = st[f"qk{q}"]
        out[key] = new
    return out
