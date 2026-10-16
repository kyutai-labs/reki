"""Rotation-rep registry: every axis-angle dataset has an axis-angle→euler unification path.

The data-content convention checks live in ``scripts/data/check_datasets.py`` (opt-in, needs data).
"""

from __future__ import annotations

import pytest

from reki.data.robotics.registry import AXIS_ANGLE, DATASETS
from reki.data.robotics.transforms import ROT_BLOCKS

pytestmark = pytest.mark.cpu


def test_rot_rep_registry_consistency() -> None:
    """Every axis-angle-declared dataset must map to an embodiment with a defined rotation-block
    convention — ``single_arm_ee_euler`` (single-arm, default [(3,6)]) or any key in
    ``ROT_BLOCKS`` (e.g. ``bimanual_joint`` = [(3,6),(10,13)]). Catches a stray rep tag on a
    bucket that has no axis-angle→euler unification path."""
    allowed = {"single_arm_ee_euler", *ROT_BLOCKS}
    for name, spec in DATASETS.items():
        if spec.action_rot_rep != AXIS_ANGLE:
            continue
        emb = spec.embodiment
        assert emb in allowed, (
            f"{name} has an axis-angle action but maps to embodiment {emb!r}, "
            f"which has no rotation-block convention (expected one of {sorted(allowed)})."
        )
