#!/usr/bin/env python
"""Opt-in data-content checks on datasets under ``$REKI_DATASETS_ROOT``.

The checks read ``meta/info.json``, ``meta/stats.json``, euler-stat sidecars and raw parquets.
They check:

- embodiment-group conventions, including stats, quantile and physical-range checks,
- raw state, action and camera layouts against the registry,
- gripper flips against the polarity in the raw data.

Datasets absent on disk are skipped. Run the checks after you add or re-process a dataset:

    pixi run --frozen python scripts/data/check_datasets.py            # every check
    pixi run --frozen python scripts/data/check_datasets.py -k gripper  # name filter

The script exits with code 1 on any FAIL or ERROR.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from reki.data.robotics.lerobot_io import (
    find_collection_leaves,
    find_lerobot_dir,
    is_collection,
    resolve_lerobot_dirs,
)
from reki.data.robotics.mirror import MIRROR_SPECS
from reki.data.robotics.registry import AXIS_ANGLE, DATASETS
from reki.policy.train_config import TrainConfig


class Skip(Exception):
    """A dataset check skipped because its required data is unavailable."""


def skip(msg: str) -> None:
    """Skip the current dataset check with a reason."""
    raise Skip(msg)


def params(values: list) -> Callable:
    """Attach parameter values to a dataset check."""

    def deco(fn: Callable) -> Callable:
        fn.params = list(values)  # type: ignore[attr-defined]
        return fn

    return deco


_DECLARED_DIMS: dict[str, int] = {f.name: f for f in dataclasses.fields(TrainConfig)}[
    "state_mlp_embodiments"
].default_factory()  # type: ignore[misc]

_STATE = "observation.state"
_SENTINEL_RATIO = 40.0
_JOINT_RAD_MAX = 8.0
_EE_DELTA_EMB = {"single_arm_ee_euler", "single_arm_ee_quat"}

# To onboard a robot, add its embodiment here (see docs/data.md, "Onboarding a dataset").
# Joint-angle state layouts → their gripper dims, which the radian range check skips.
_GRIPPER_DIMS: dict[str, set[int]] = {
    "franka": {7},
    "ur5e": {6},
    "agilex": {6, 13},
    "aloha": {6, 13},
    "yam": {6, 13},
    "bimanual_joint": {7, 15},
    "r1lite": {15, 16},
    "g2a": {0, 1},
    "bimanual_dexterous": set(),
    "gr1": set(),
    "airbot_mmk2": set(),
    "unitree_g1": set(),
}
# Embodiments whose gripper dims are an open fraction in [0, 1].
_OPEN_FRACTION_GRIPPER: set[str] = {"aloha", "agilex", "franka", "yam"}
# Gripper dims for the polarity check, indexed in the sliced state layout.
_GRIPPER_KEY_STATE_IDX: dict[str, list[int]] = {
    "franka": [7],
    "ur5e": [6],
    "single_arm_ee_euler": [7],
    "single_arm_ee_quat": [7],
    "aloha": [6, 13],
    "agilex": [6, 13],
    "yam": [6, 13],
    "bimanual_joint": [7, 15],
    "r1lite": [15, 16],
}


def _leaf_dirs(name: str) -> list[Path]:
    """On-disk lerobot dirs for a dataset name (collection → leaves), or [] if absent."""
    if is_collection(name):
        try:
            return [d for d in find_collection_leaves(name) if (d / "meta" / "stats.json").exists()]
        except Exception:
            return []
    try:
        d = find_lerobot_dir(name)
        return [d] if (d / "meta" / "stats.json").exists() else []
    except Exception:
        return []


def _stats(leaf: Path, key: str) -> dict | None:
    try:
        s = json.loads((leaf / "meta" / "stats.json").read_text()).get(key)
        return s if s and "min" in s else None
    except Exception:
        return None


def _euler_sidecar(name: str, leaf: Path) -> dict | None:
    """The euler stats sidecar that training reads for an axis-angle member under
    unify_ee_euler_rotation (``norm_stats_euler_roll.json`` when the dataset shifts the state
    roll), or None if absent."""
    p = leaf / "meta" / _sidecar_name(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _sidecar_name(name: str) -> str:
    return (
        "norm_stats_euler_roll.json"
        if DATASETS[name].recenter_state_roll
        else "norm_stats_euler.json"
    )


def _live_stats(name: str, leaf: Path, key: str) -> dict | None:
    """The stats of ``key`` that a ``unify_ee_euler_rotation=True`` run normalizes with.

    For an axis-angle member this is the euler sidecar (dims 3:6 are euler). For other
    members it is ``meta/stats.json``. None when the stats are absent."""
    rep = DATASETS[name].action_rot_rep if key == "action" else DATASETS[name].state_rot_rep
    if rep == AXIS_ANGLE:
        sc = _euler_sidecar(name, leaf)
        s = sc.get(key) if sc else None
        return s if s and "min" in s else None
    return _stats(leaf, key)


_GROUPS: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
_EMBODIMENT_OF = {n: s.embodiment for n, s in DATASETS.items() if s.embodiment is not None}
for _name, _emb in sorted(_EMBODIMENT_OF.items()):
    for _leaf in _leaf_dirs(_name):
        _label = f"{_name}/{_leaf.name}" if is_collection(_name) else _name
        _GROUPS[_emb].append((_name, _label, _leaf))

_GROUP_IDS = sorted(_GROUPS)


@params(_GROUP_IDS)
def check_state_dim_matches_embodiment(emb: str) -> None:
    """Every member's per-dim state stats, sliced as registered, match the declared dim."""
    if emb not in _DECLARED_DIMS:
        skip(f"{emb}: no declared dim")
    declared = _DECLARED_DIMS[emb]
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, "observation.state")
        if st is None:
            continue
        sl = DATASETS[name].state_slice
        sliced = len(sl) if sl is not None else len(st["min"])
        if sliced != declared:
            bad.append(f"{label}: sliced state dim {sliced} != declared {declared}")
    assert not bad, (
        f"{emb} state-dim mismatch (wrong state slice or wrong mapping):\n  " + "\n  ".join(bad)
    )


@params([e for e in _GROUP_IDS if e == "single_arm_ee_quat"])
def check_single_arm_ee_quat_state_is_unit_quaternion(emb: str) -> None:
    """The single_arm_ee_quat state is [xyz, quat4 (xyzw), gripper]. Each quaternion component
    (dims 3:7) is within ±1.05 on every member. A larger value means a non-unit or mislabeled
    rotation."""
    bad = []
    for _name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, _STATE)
        if st is None or len(st["min"]) < 7:
            continue
        comp = np.maximum(
            np.abs(np.asarray(st["min"])[3:7]), np.abs(np.asarray(st["max"])[3:7])
        ).max()
        if comp > 1.05:
            bad.append(f"{label}: max |quat component| = {comp:.2f} > 1.05 (non-unit)")
    assert not bad, (
        "single_arm_ee_quat state quaternion is non-unit (convert or quarantine):\n  "
        + "\n  ".join(bad)
    )


@params([e for e in _GROUP_IDS if e in _EE_DELTA_EMB])
def check_ee_action_is_zero_centered_delta(emb: str) -> None:
    """The action translation (dims 0:3) of each EE-delta member is zero-centered.

    A member with ``|mean| / range > 0.30`` on these dims has absolute-pose actions in a delta
    group."""
    bad = []
    for _name, label, leaf in _GROUPS[emb]:
        a = _stats(leaf, "action")
        if a is None or len(a["min"]) < 3:
            continue
        frac = _centered_ratio(a)
        if frac > 0.30:
            bad.append(
                f"{label}: action translation |mean|/range = {frac:.2f} (>0.30 → absolute, not delta)"
            )
    assert not bad, f"{emb} action is not zero-centered delta:\n  " + "\n  ".join(bad)


def check_delta_ee_control_mode_matches_action() -> None:
    """Each dataset with control mode "delta end-effector" has a zero-centered delta action.

    The check is ``|mean| / range ≤ 0.30`` on translation dims 0:3. The control mode is set per
    dataset (``DatasetSpec.control_mode``), not per embodiment, so a DELTA_EE dataset of any
    embodiment is checked."""
    bad = []
    seen: set[str] = set()
    for emb in _GROUP_IDS:
        for name, label, leaf in _GROUPS[emb]:
            if name in seen:
                continue
            seen.add(name)
            if DATASETS[name].control_mode != "delta end-effector":
                continue
            a = _stats(leaf, "action")
            if a is None or len(a["min"]) < 3:
                continue
            frac = _centered_ratio(a)
            if frac > 0.30:
                bad.append(
                    f"{label}: control_mode='delta end-effector' but action |mean|/range="
                    f"{frac:.2f} (>0.30 → not a delta; wrong control_mode in DATASETS?)"
                )
    assert not bad, "control mode 'delta end-effector' on a non-delta action:\n  " + "\n  ".join(
        bad
    )


def _centered_ratio(a: dict) -> float:
    mean = np.asarray(a["mean"], dtype=np.float64)[:3]
    span = (np.asarray(a["max"], dtype=np.float64) - np.asarray(a["min"], dtype=np.float64))[:3]
    return float((np.abs(mean) / np.maximum(span, 1e-6)).max())


def _band(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper), preferring qk quantiles, then q quantiles, then raw extrema."""
    lower = np.asarray(s.get("qk01", s.get("q01", s["min"])), dtype=np.float64)
    upper = np.asarray(s.get("qk99", s.get("q99", s["max"])), dtype=np.float64)
    return lower, upper


def _robust_band(a: dict, lo: int, hi: int) -> float:
    """max(|q01|, |q99|) over dims [lo:hi], preferring qk/q quantiles over raw min/max."""
    lower, upper = _band(a)
    lower, upper = lower[lo:hi], upper[lo:hi]
    return float(np.maximum(np.abs(upper), np.abs(lower)).max())


@params([e for e in _GROUP_IDS if e in _EE_DELTA_EMB])
def check_ee_rotation_delta_within_pi(emb: str) -> None:
    """The robust band (qk99/q99) of the action rotation (dims 3:6) is at most π.

    These dims are per-step delta angles. A larger band means degrees, an absolute or cumulative
    angle, or mixed axis-angle and euler."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        a = _live_stats(name, leaf, "action")
        if a is None or len(a["min"]) < 6:
            continue
        band = _robust_band(a, 3, 6)
        if band > np.pi + 0.05:
            bad.append(f"{label}: rotation qk99 band = {band:.2f} > π")
    assert not bad, f"{emb} per-step rotation delta exceeds π (radians/rep bug):\n  " + "\n  ".join(
        bad
    )


def _slice_arr(sl: list[int | None] | None, a: np.ndarray) -> np.ndarray:
    """Apply a STATE_SLICE to a per-dim stat array; pad (None) slots → nan (skipped)."""
    if sl is None:
        return a
    out = np.full(len(sl), np.nan)
    for i, idx in enumerate(sl):
        if idx is not None:
            out[i] = a[idx]
    return out


def _sliced_band(name: str, st: dict) -> tuple[np.ndarray, np.ndarray]:
    sl = DATASETS[name].state_slice
    lo, hi = _band(st)
    return _slice_arr(sl, lo), _slice_arr(sl, hi)


@params(_GROUP_IDS)
def check_state_stats_finite(emb: str) -> None:
    """No NaN/Inf in any state stat field (a non-finite stat poisons normalization)."""
    bad = []
    for _name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, _STATE)
        if st is None:
            continue
        for k in ("min", "max", "mean", "std", "qk01", "qk99", "q01", "q99"):
            if k in st and not np.all(np.isfinite(np.asarray(st[k], dtype=np.float64))):
                bad.append(f"{label}: non-finite {k}")
    assert not bad, f"{emb} state has non-finite stats:\n  " + "\n  ".join(bad)


@params(_GROUP_IDS)
def check_state_quantiles_well_formed(emb: str) -> None:
    """State qk01/qk99 are present, finite and ordered (q01 ≤ q99) on every used dim.

    Quantile normalization divides by ``q99 - q01``. An inverted band flips the sign of the dim.
    A degenerate band (q99 == q01) is a parked dim and is not flagged: ``normalize_feature`` sets
    its denominator to 1, so the dim normalizes to -1 (no masking). Global qk01/qk99 are pooled
    over all leaves, so the band is not compared with the per-leaf min/max."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _live_stats(name, leaf, _STATE)
        if st is None:
            continue
        qlo, qhi = st.get("qk01"), st.get("qk99")
        if qlo is None or qhi is None:
            bad.append(f"{label}: missing state qk01/qk99")
            continue
        sl = DATASETS[name].state_slice
        q01 = _slice_arr(sl, np.asarray(qlo, dtype=np.float64))
        q99 = _slice_arr(sl, np.asarray(qhi, dtype=np.float64))
        for i in range(len(q99)):
            if np.isnan(q99[i]):
                continue
            if not (np.isfinite(q01[i]) and np.isfinite(q99[i])):
                bad.append(f"{label}: dim {i} non-finite quantile")
            elif q99[i] < q01[i] - 1e-9:
                bad.append(f"{label}: dim {i} inverted band (q01={q01[i]:.4g} > q99={q99[i]:.4g})")
    assert not bad, f"{emb} state quantiles ill-formed:\n  " + "\n  ".join(bad)


def _frame_rot_sample(leaf: Path, col: str, lo: int, hi: int, n: int = 1000) -> np.ndarray | None:
    """First ~n frames of ``col`` rotation dims [lo:hi] from a leaf's first data parquet,
    or None."""
    files = sorted(glob.glob(str(leaf / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        return None
    try:
        arr = np.asarray(
            pq.read_table(files[0], columns=[col]).column(col).to_pylist(), dtype=np.float64
        )
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < hi:
        return None
    return arr[:n, lo:hi]


@params([e for e in _GROUP_IDS if e == "single_arm_ee_euler"])
def check_axisangle_members_raw_rotation_is_valid_rotvec(emb: str) -> None:
    """The raw rotation (dims 3:6) of each axis-angle member is a valid axis-angle rotvec.

    LIBERO's quaternion conversion emits angles in [0, 2π), so the check catches corrupt or
    out-of-convention values, not ordinary euler data mislabeled as axis-angle. State and action
    are checked separately: a member can have an axis-angle action and an euler state."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        for col, rep in (
            ("action", DATASETS[name].action_rot_rep),
            ("observation.state", DATASETS[name].state_rot_rep),
        ):
            if rep != "axis_angle":
                continue
            rot = _frame_rot_sample(leaf, col, 3, 6)
            if rot is None or len(rot) < 16:
                continue
            norm = np.linalg.norm(rot, axis=1)
            frac_over = float(np.mean(norm > 2 * np.pi + 0.05))
            if frac_over > 0.02:
                bad.append(
                    f"{label}: {col} dims 3:6 are not a valid axis-angle rotvec — "
                    f"‖rot‖>2π in {frac_over:.0%} of frames (max ‖rot‖={norm.max():.2f})"
                )
    assert not bad, (
        f"{emb} axis-angle member rotation is outside the supported convention:\n  "
        + "\n  ".join(bad)
    )


@params([e for e in _GROUP_IDS if e == "single_arm_ee_euler"])
def check_axisangle_members_have_valid_euler_sidecar(emb: str) -> None:
    """Each axis-angle member (e.g. libero) has a valid euler sidecar.

    With ``unify_ee_euler_rotation=True``, training normalizes these members with
    ``meta/norm_stats_euler.json`` (``norm_stats_euler_roll.json`` for a dataset that shifts the
    state roll), not with ``stats.json``. Without the sidecar, training raises
    FileNotFoundError. For each axis-angle feature, the sidecar qk01/qk99 on dims 3:6 must be
    finite, ordered and within π."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        if DATASETS[name].action_rot_rep != AXIS_ANGLE:
            continue
        sc = _euler_sidecar(name, leaf)
        if sc is None:
            bad.append(f"{label}: missing/unreadable {_sidecar_name(name)} (unify run requires it)")
            continue
        cols = [
            c
            for c, rep in (
                ("action", DATASETS[name].action_rot_rep),
                ("observation.state", DATASETS[name].state_rot_rep),
            )
            if rep == "axis_angle"
        ]
        for col in cols:
            s = sc.get(col)
            qlo = s and s.get("qk01")
            qhi = s and s.get("qk99")
            if not qlo or not qhi or len(qhi) < 6:
                bad.append(f"{label}: sidecar {col} missing/short quantiles")
                continue
            lo = np.asarray(qlo, dtype=np.float64)[3:6]
            hi = np.asarray(qhi, dtype=np.float64)[3:6]
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                bad.append(f"{label}: sidecar {col} euler rot band non-finite")
            elif np.any(hi < lo - 1e-9):
                bad.append(f"{label}: sidecar {col} euler rot band inverted (q01>q99)")
            elif float(np.maximum(np.abs(lo), np.abs(hi)).max()) > np.pi + 0.05:
                bad.append(f"{label}: sidecar {col} euler rot band > π (stale axis-angle stats?)")
    assert not bad, f"{emb} euler sidecar invalid:\n  " + "\n  ".join(bad)


@params(_GROUP_IDS)
def check_state_no_sentinel_outliers(emb: str) -> None:
    """No sliced state dim has a raw |min| or |max| above ``_SENTINEL_RATIO`` times its robust
    band. Such a value is a sensor sentinel or a glitch frame, e.g. a 65535 uint16 gripper
    reading."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, _STATE)
        if st is None:
            continue
        sl = DATASETS[name].state_slice
        lo, hi = _sliced_band(name, st)
        mn = _slice_arr(sl, np.asarray(st["min"], dtype=np.float64))
        mx = _slice_arr(sl, np.asarray(st["max"], dtype=np.float64))
        for i in range(len(hi)):
            band_mag = max(abs(lo[i]), abs(hi[i]))
            if np.isnan(band_mag) or band_mag < 1e-3:
                continue
            ratio = max(abs(mn[i]), abs(mx[i])) / band_mag
            if ratio > _SENTINEL_RATIO:
                bad.append(
                    f"{label}: dim {i} |extreme| {ratio:.0f}× robust band "
                    f"(min={mn[i]:.4g} max={mx[i]:.4g} q01={lo[i]:.4g} q99={hi[i]:.4g})"
                )
    assert not bad, f"{emb} state has sentinel/glitch dims:\n  " + "\n  ".join(bad)


@params([e for e in _GROUP_IDS if e == "single_arm_ee_euler"])
def check_state_euler_pad_slot_is_zero(emb: str) -> None:
    """single_arm_ee_euler slot 6 is a pad. On an unsliced member, slot 6 is all zero. Signal in
    slot 6 means that single_arm_ee_euler is the wrong mapping."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        if DATASETS[name].state_slice is not None:
            continue
        st = _stats(leaf, _STATE)
        if st is None or len(st["min"]) < 7:
            continue
        if not all(abs(st[k][6]) < 1e-6 for k in ("min", "max", "std")):
            bad.append(
                f"{label}: single_arm_ee_euler pad slot 6 carries signal "
                f"(min={st['min'][6]} max={st['max'][6]})"
            )
    assert not bad, (
        f"{emb} pad slot 6 not zero (wrong single_arm_ee_euler mapping):\n  " + "\n  ".join(bad)
    )


@params([e for e in _GROUP_IDS if e == "single_arm_ee_euler"])
def check_state_euler_within_pi(emb: str) -> None:
    """The single_arm_ee_euler state rotation (sliced dims 3:6) is an absolute euler angle, so its
    robust band is at most π. A larger band means degrees or a wrong rotation rep."""
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _live_stats(name, leaf, _STATE)
        if st is None:
            continue
        lo, hi = _sliced_band(name, st)
        if len(hi) < 6:
            continue
        band = float(np.nanmax(np.maximum(np.abs(lo[3:6]), np.abs(hi[3:6]))))
        if band > np.pi + 0.1:
            bad.append(f"{label}: euler rotation band = {band:.2f} > π (degrees/rep bug)")
    assert not bad, f"{emb} absolute euler rotation exceeds π:\n  " + "\n  ".join(bad)


@params([e for e in _GROUP_IDS if e in _GRIPPER_DIMS])
def check_state_joint_within_physical_range(emb: str) -> None:
    """The robust band of each non-gripper joint dim is within ``_JOINT_RAD_MAX`` radians. A dim
    in degrees or raw encoder counts exceeds it."""
    grip = _GRIPPER_DIMS[emb]
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, _STATE)
        if st is None:
            continue
        lo, hi = _sliced_band(name, st)
        for i in range(len(hi)):
            if i in grip or np.isnan(hi[i]):
                continue
            b = max(abs(lo[i]), abs(hi[i]))
            if b > _JOINT_RAD_MAX:
                bad.append(
                    f"{label}: dim {i} robust |angle| = {b:.1f} > {_JOINT_RAD_MAX} rad (degrees/encoder-count?)"
                )
    assert not bad, f"{emb} joint dim outside physical radian range:\n  " + "\n  ".join(bad)


@params([e for e in _GROUP_IDS if e in _OPEN_FRACTION_GRIPPER])
def check_gripper_dims_are_open_fraction(emb: str) -> None:
    """Declared gripper dims for [0,1]-open-fraction embodiments must be in that band: a wrong
    gripper index (arm joint in the slot) or a non-normalized gripper escapes it."""
    grip = _GRIPPER_DIMS[emb]
    bad = []
    for name, label, leaf in _GROUPS[emb]:
        st = _stats(leaf, _STATE)
        if st is None:
            continue
        lo, hi = _sliced_band(name, st)
        for g in grip:
            if np.isnan(hi[g]):
                continue
            if lo[g] < -0.3 or hi[g] > 1.3:
                bad.append(
                    f"{label}: gripper dim {g} band [{lo[g]:.2f}, {hi[g]:.2f}] outside [0,1]"
                )
    assert not bad, f"{emb} gripper dims not [0,1] open-fraction:\n  " + "\n  ".join(bad)


_MAX_LEAVES = 3
_MAX_EPS = 8
_CORR_STRONG = 0.7
_CORR_MARGIN = 0.15
_TRAVEL_MIN = 0.6


def _raw_state_idx(ds: str, canon: int) -> int | None:
    sl = DATASETS[ds].state_slice
    if sl is None:
        return canon
    if canon >= len(sl):
        return None
    return sl[canon]


def _episodes(ds: str):
    """Yield (state[T,Ds], action[T,Da]) per episode, for up to ``_MAX_EPS`` episodes.

    Reads the first 6000 rows of up to 2 files in each of up to ``_MAX_LEAVES`` leaves."""
    try:
        leaves = resolve_lerobot_dirs(ds)
    except Exception:
        return
    n = 0
    for leaf in leaves[:_MAX_LEAVES]:
        files = sorted(glob.glob(f"{leaf}/data/**/*.parquet", recursive=True))
        for fp in files[:2]:
            pf = pq.ParquetFile(fp)
            cols = pf.schema_arrow.names
            if "observation.state" not in cols or "action" not in cols:
                break
            epk = next((k for k in ("episode_index", "episode") if k in cols), None)
            batch = next(
                pf.iter_batches(
                    batch_size=6000,
                    columns=["observation.state", "action"] + ([epk] if epk else []),
                )
            )
            S = np.asarray(batch.column("observation.state").to_pylist(), float)
            A = np.asarray(batch.column("action").to_pylist(), float)
            eps = np.asarray(batch.column(epk).to_pylist()) if epk else np.zeros(len(S))
            for e in np.unique(eps):
                yield S[eps == e], A[eps == e]
                n += 1
                if n >= _MAX_EPS:
                    return


def _temporal_polarity(col_per_ep: list[np.ndarray]) -> str:
    """CLOSED/OPEN/AMBIGUOUS from frame0(open)→grasp travel; ≥2 unanimous clean episodes."""
    votes = []
    for v in col_per_ep:
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-3:
            continue
        f0n = (v[0] - lo) / (hi - lo)
        extn = (v[int(np.argmax(np.abs(v - v[0])))] - lo) / (hi - lo)
        if abs(f0n - extn) < _TRAVEL_MIN:
            continue
        votes.append("CLOSED" if f0n < 0.5 else "OPEN")
    if len(votes) < 2 or len(set(votes)) != 1:
        return "AMBIGUOUS"
    return votes[0]


def _verdicts(eps: list[tuple[np.ndarray, np.ndarray]], sidx: int) -> tuple[str, str]:
    """(state_polarity, action_polarity), each CLOSED, OPEN or AMBIGUOUS.

    The action gripper is the action dim with the strongest correlation to the state gripper.
    Each polarity comes from its own trajectory. When the state polarity is ambiguous, it is the
    action polarity, flipped if the correlation is negative. If the two disagree, both are
    AMBIGUOUS.
    """
    svs, avs = [], []
    for s, a in eps:
        if sidx < s.shape[1]:
            svs.append(s[:, sidx])
            avs.append(a)
    if len(svs) < 2:
        return "AMBIGUOUS", "AMBIGUOUS"
    Sc = np.concatenate(svs)
    Ac = np.concatenate(avs, 0)
    if Sc.std() < 1e-6:
        return "AMBIGUOUS", "AMBIGUOUS"
    ranked = sorted(
        (abs(float(np.corrcoef(Sc, Ac[:, j])[0, 1])), float(np.corrcoef(Sc, Ac[:, j])[0, 1]), j)
        for j in range(Ac.shape[1])
        if Ac[:, j].std() > 1e-6
    )[::-1]
    best_c, action_dim = 0.0, None
    if (
        ranked
        and ranked[0][0] >= _CORR_STRONG
        and (len(ranked) < 2 or ranked[0][0] - ranked[1][0] >= _CORR_MARGIN)
    ):
        best_c, action_dim = ranked[0][1], ranked[0][2]

    state_t = _temporal_polarity(svs)
    action_t = (
        _temporal_polarity([a[:, action_dim] for a in avs])
        if action_dim is not None
        else "AMBIGUOUS"
    )
    if state_t == "AMBIGUOUS" and action_t == "AMBIGUOUS":
        return "AMBIGUOUS", "AMBIGUOUS"

    def _prop(p: str) -> str:
        return p if best_c > 0 else ("OPEN" if p == "CLOSED" else "CLOSED")

    if state_t != "AMBIGUOUS" and action_t != "AMBIGUOUS" and state_t != _prop(action_t):
        return "AMBIGUOUS", "AMBIGUOUS"
    state_pol = (
        state_t
        if state_t != "AMBIGUOUS"
        else (_prop(action_t) if action_t != "AMBIGUOUS" else "AMBIGUOUS")
    )
    return state_pol, action_t


def check_gripper_flip_table_matches_data() -> None:
    """``DatasetSpec.gripper_flip_*`` match the gripper polarity measured in the raw episodes."""
    gripper_dss = [d for d, k in _EMBODIMENT_OF.items() if k in _GRIPPER_KEY_STATE_IDX]
    mismatches: list[str] = []
    checked = 0
    for ds in gripper_dss:
        eps = list(_episodes(ds))
        if not eps:
            continue
        checked += 1
        key = _EMBODIMENT_OF[ds]
        tbl_state = set(DATASETS[ds].gripper_flip_state or [])
        tbl_action_flipped = bool(DATASETS[ds].gripper_flip_action)
        action_confident = []
        for canon in _GRIPPER_KEY_STATE_IDX[key]:
            raw = _raw_state_idx(ds, canon)
            if raw is None:
                continue
            state_pol, action_pol = _verdicts(eps, raw)
            if state_pol in ("CLOSED", "OPEN") and (state_pol == "CLOSED") != (canon in tbl_state):
                mismatches.append(
                    f"{ds} [{key}] STATE grip canon{canon}(raw{raw}): data={state_pol} but table "
                    f"{'flips' if canon in tbl_state else 'does NOT flip'} it"
                )
            if action_pol in ("CLOSED", "OPEN"):
                action_confident.append(action_pol)
        if action_confident:
            data_action_closed = "CLOSED" in action_confident
            if data_action_closed != tbl_action_flipped:
                mismatches.append(
                    f"{ds} [{key}] ACTION: data={'CLOSED' if data_action_closed else 'OPEN'} but "
                    f"table action flip is {'present' if tbl_action_flipped else 'absent'}"
                )
    if checked == 0:
        skip("no gripper datasets resolvable on disk")
    assert not mismatches, (
        f"The gripper flips disagree with data for {len(mismatches)} case(s) "
        f"(checked {checked} datasets):\n  " + "\n  ".join(mismatches)
    )


def check_raw_layout_matches_registry() -> None:
    """Raw camera keys and vector widths support the registered slices and mirror specs."""
    problems = []
    inspected = 0
    for name, spec in DATASETS.items():
        mirror = MIRROR_SPECS[spec.mirror] if spec.mirror else None
        for leaf in _leaf_dirs(name):
            inspected += 1
            label = f"{name}/{leaf.name}" if is_collection(name) else name
            features = json.loads((leaf / "meta" / "info.json").read_text())["features"]
            present = {raw for raw in spec.cameras if raw in features}
            if not present:
                problems.append(f"{label}: no registered camera key is present")
            cameras = {spec.cameras[raw] for raw in present}
            # A lone pair member would keep the wrong side under the mirror.
            for pair in mirror.paired_cameras if mirror else ():
                if (pair[0] in cameras) != (pair[1] in cameras):
                    problems.append(f"{label}: only one of camera pair {pair} is present")
            w_s = features[_STATE]["shape"][0]
            w_a = features["action"]["shape"][0]
            for feature, width, sl in (
                ("state", w_s, spec.state_slice),
                ("action", w_a, spec.action_slice),
            ):
                indices = [i for i in sl or [] if i is not None]
                if indices and max(indices) >= width:
                    problems.append(f"{label}: {feature} slice exceeds raw width {width}")
            if mirror and (w_s != mirror.dim("state") or w_a != mirror.dim("action")):
                problems.append(
                    f"{label}: raw widths state={w_s}, action={w_a} do not match mirror"
                )
    if inspected == 0:
        skip("no registered dataset on disk")
    assert not problems, "raw layout does not match registry:\n  " + "\n  ".join(problems)


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in data-content checks on your datasets.")
    parser.add_argument("-k", default="", help="run only checks whose name contains this")
    args = parser.parse_args()
    if not os.environ.get("REKI_DATASETS_ROOT"):
        raise SystemExit("set REKI_DATASETS_ROOT to the directory holding your datasets")
    checks = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("check_") and callable(fn) and args.k in name
    ]
    counts: dict[str, int] = defaultdict(int)
    for name, fn in checks:
        for p in getattr(fn, "params", [None]):
            label = name if p is None else f"{name}[{p}]"
            try:
                fn() if p is None else fn(p)
                status = "PASS"
            except Skip as e:
                status = f"SKIP ({e})"
            except AssertionError as e:
                status = f"FAIL\n{e}"
            except Exception:
                status = f"ERROR\n{traceback.format_exc()}"
            counts[status.split()[0]] += 1
            print(f"{label}: {status}", flush=True)
    print(dict(counts))
    sys.exit(1 if counts["FAIL"] or counts["ERROR"] else 0)


if __name__ == "__main__":
    main()
