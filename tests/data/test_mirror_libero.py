"""LIBERO (``single_arm_ee_euler``) sagittal mirror. ``test_lateral_axis_is_the_measured_one``
re-derives the left-right axis from real-data extrema (an involution check cannot catch a wrong
axis); the other signs are tested as relations (rotation = -position, invariant aperture)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reki.data.robotics.mirror import (
    LIBERO_SPEC,
)

pytestmark = pytest.mark.cpu

_POS = slice(0, 3)
_ROT = slice(3, 6)


def _fixture() -> dict:
    path = Path(__file__).parents[1] / "fixtures" / "libero_ee_extrema.json"
    return json.loads(path.read_text())


def test_lateral_axis_is_the_measured_one() -> None:
    """The negated position axis is the one measured as left-right on real episodes."""
    from scripts.data.derive_libero_mirror_axis import lateral_axis

    measured, evidence = lateral_axis(_fixture())
    negated = [i for i, s in enumerate(LIBERO_SPEC.state_sign[_POS]) if s == -1]
    assert negated == [measured], (
        f"spec negates position axes {negated} but the data says the lateral axis is "
        f"{measured} (evidence: {evidence})"
    )
    assert [i for i, s in enumerate(LIBERO_SPEC.action_sign[_POS]) if s == -1] == [measured]


def test_rotation_is_the_pseudovector_of_the_position_sign() -> None:
    """A reflection maps a rotvec by ``-M``, so the rotation sign is the position sign negated,
    componentwise. Pinning the relation catches either block being edited alone."""
    for sign in (LIBERO_SPEC.state_sign, LIBERO_SPEC.action_sign):
        pos, rot = np.asarray(sign[_POS]), np.asarray(sign[_ROT])
        np.testing.assert_array_equal(rot, -pos)


def test_gripper_aperture_survives_the_mirror() -> None:
    """A reflection cannot change how far open a hand is. The two Franka finger joints sit at
    ±q, and the mirror exchanges their identities, so swap-and-negate leaves the aperture
    alone; negating without the swap would report a NEGATIVE aperture instead."""
    q0, q1 = 0.0400, -0.0398
    state = np.asarray([0.1, 0.2, 0.9, 3.0, 0.1, -0.1, q0, q1], dtype=np.float32)
    out = LIBERO_SPEC.mirror(state, "state")
    assert out[6] == pytest.approx(-q1, abs=1e-7), "kept finger must stay a positive aperture"
    assert out[7] == pytest.approx(-q0, abs=1e-7)
    assert out[6] > 0 > out[7], "the ± finger convention must be preserved"
    action = np.asarray([0.01, -0.02, 0.03, 0.1, -0.2, 0.3, -1.0], dtype=np.float32)
    assert LIBERO_SPEC.mirror(action, "action")[6] == -1.0


def test_mirrored_stats_match_a_recompute_on_mirrored_rows() -> None:
    """The closed-form stats mapping must equal stats measured on the mirrored rows — including
    for the roll dim, whose euler values pile up near ±π and cross the wrap under negation."""

    def exact(arr: np.ndarray) -> dict[str, list]:
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

    rng = np.random.default_rng(3)
    rows = rng.normal(size=(4096, 8)) * np.arange(1, 9)
    rows[:, 3] = rng.choice([-1.0, 1.0], 4096) * (np.pi - rng.exponential(0.05, 4096))
    analytic = LIBERO_SPEC.mirror_stats(exact(rows), "state")
    for field, expected in exact(LIBERO_SPEC.mirror(rows, "state")).items():
        np.testing.assert_allclose(
            analytic[field], expected, rtol=0, atol=1e-12, err_msg=f"field {field}"
        )
    assert np.all(np.asarray(analytic["std"]) > 0), "a negated dim must not get a negative std"
    assert np.all(np.asarray(analytic["min"]) <= np.asarray(analytic["max"]))
