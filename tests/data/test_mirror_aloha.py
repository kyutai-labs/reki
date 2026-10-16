"""RoboTwin ALOHA sagittal mirror. ``test_sign_reproduces_sagittal_reflection`` runs forward
kinematics on the real ARX5 chains: an involution check cannot catch a wrong ±1 sign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reki.data.robotics.mirror import (
    ALOHA_ARX5_SPEC,
    ALOHA_MIRROR_SIGN_ARX5,
    mirror_aloha,
)

pytestmark = pytest.mark.cpu


_CHAINS = json.loads((Path(__file__).parents[1] / "fixtures" / "arx5_chain.json").read_text())[
    "chains"
]

_M4 = np.diag([1.0, -1.0, 1.0, 1.0])

_MAX_POS_RESIDUAL_M = 0.05
_MAX_ROT_RESIDUAL = 0.20


def _rpy_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_mat(*rpy)
    T[:3, 3] = xyz
    return T


def _axis_rotation(axis: list[float], q: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s, C = np.cos(q), np.sin(q), 1 - np.cos(q)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
    return T


def _fk(side: str, arm_q: np.ndarray) -> np.ndarray:
    """Footprint-frame pose of <side>_link6 for the 6 arm joints, in data order."""
    T = np.eye(4)
    consumed = 0
    for joint in _CHAINS[side]:
        T = T @ _transform(joint["xyz"], joint["rpy"])
        if joint["revolute"]:
            T = T @ _axis_rotation(joint["axis"], arm_q[consumed])
            consumed += 1
    assert consumed == 6, f"{side}: consumed {consumed} revolute joints, expected 6"
    return T


def _reflection_residual(sign: tuple[int, ...], joint_states: np.ndarray) -> tuple[float, float]:
    """Worst-case (position, rotation) gap between the mirrored arm and the reflected one."""
    arm_sign = np.asarray(sign[:6], float)
    worst_pos = worst_rot = 0.0
    for q in joint_states:
        left_q, right_q = q[0:6], q[7:13]
        pairs = (
            (_fk("fl", arm_sign * right_q), _M4 @ _fk("fr", right_q) @ _M4),
            (_fk("fr", arm_sign * left_q), _M4 @ _fk("fl", left_q) @ _M4),
        )
        for mirrored, reflected in pairs:
            worst_pos = max(worst_pos, float(np.linalg.norm(mirrored[:3, 3] - reflected[:3, 3])))
            worst_rot = max(worst_rot, float(np.linalg.norm(mirrored[:3, :3] - reflected[:3, :3])))
    return worst_pos, worst_rot


def _sample_joint_states(n: int = 64) -> np.ndarray:
    """Plausible ALOHA configurations: ±1 rad per joint, grippers in [0, 1]."""
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=(n, 14))
    q[:, 6] = rng.uniform(0.0, 1.0, size=n)
    q[:, 13] = rng.uniform(0.0, 1.0, size=n)
    return q


def test_sign_reproduces_sagittal_reflection() -> None:
    """FK check: the shipped sign mirrors the arm across y = 0. Slot 5 (wrist roll) spins the
    gripper without moving link6's origin, so only the rotation residual catches it."""
    pos, rot = _reflection_residual(ALOHA_MIRROR_SIGN_ARX5, _sample_joint_states())
    assert pos < _MAX_POS_RESIDUAL_M, f"position residual {pos:.4f} m — is the sign still right?"
    assert rot < _MAX_ROT_RESIDUAL, f"rotation residual {rot:.4f} — is the sign still right?"


def _exact_stats(arr: np.ndarray) -> dict[str, list]:
    q = {p: np.quantile(arr, p / 100, axis=0) for p in (1, 50, 99)}
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": q[1].tolist(),
        "q50": q[50].tolist(),
        "q99": q[99].tolist(),
    }


def test_mirrored_stats_match_a_recompute_on_mirrored_rows() -> None:
    """The closed form must equal stats actually measured on the mirrored data."""
    rows = np.random.default_rng(3).normal(size=(4096, 14)) * np.arange(1, 15)
    analytic = ALOHA_ARX5_SPEC.mirror_stats(_exact_stats(rows), "state")
    for field, expected in _exact_stats(mirror_aloha(rows)).items():
        np.testing.assert_allclose(
            analytic[field], expected, rtol=0, atol=1e-12, err_msg=f"field {field}"
        )
