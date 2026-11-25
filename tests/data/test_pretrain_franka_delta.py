"""The Franka joint → ΔEEF readers (``pretrain.readers.franka_delta``)."""

from __future__ import annotations

import numpy as np
import pytest
from pretrain import catalog
from pretrain.readers.franka_delta import (
    ROBOSET_DIRS,
    ROBOSET_SUBSET_DIRS,
    RPT_SUBSET_EPISODES,
    delta_ee,
    make_fk,
    roboset_episodes,
    roboset_task,
    trial_arrays,
)
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.cpu


def _fake_fk(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return joints[:3].copy(), Rotation.from_rotvec([0.0, 0.0, joints[3]]).as_matrix()


def _rotvec(rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot).as_rotvec()


def test_delta_ee_layout() -> None:
    rng = np.random.default_rng(0)
    state = rng.uniform(-1, 1, size=(5, 8)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(5, 8)).astype(np.float32)
    st, ac = delta_ee(state, action, _fake_fk, _rotvec)
    assert st.shape == (5, 8) and ac.shape == (5, 7)
    np.testing.assert_array_equal(st[:, 0:3], state[:, 0:3])
    np.testing.assert_allclose(st[:, 5], state[:, 3], atol=1e-6)
    np.testing.assert_array_equal(st[:, 6], 0)
    np.testing.assert_array_equal(st[:, 7], state[:, 7])
    np.testing.assert_allclose(ac[:-1, 0:3], np.diff(state[:, 0:3], axis=0), atol=1e-6)
    np.testing.assert_allclose(ac[:-1, 5], np.diff(state[:, 3]), atol=1e-6)
    np.testing.assert_array_equal(ac[-1, 0:6], 0)
    np.testing.assert_array_equal(ac[:, 6], action[:, 7])


class _Trial(dict):
    name = "/Trial0/data"


def _trial(n: int, nan_rows: list[int]) -> _Trial:
    rng = np.random.default_rng(1)
    ctrl = rng.normal(size=(n, 7)).astype(np.float16)
    ctrl[nan_rows] = np.nan
    trial = _Trial(
        qp_arm=rng.normal(size=(n, 7)).astype(np.float16),
        qp_ee=rng.normal(size=n).astype(np.float16),
        ctrl_arm=ctrl,
        ctrl_ee=rng.normal(size=(n, 1)).astype(np.float16),
    )
    for raw in ("rgb_top", "rgb_wrist", "rgb_left", "rgb_right"):
        trial[raw] = np.zeros((n, 2, 2, 3), np.uint8)
    return trial


def test_trial_arrays_cuts_trailing_nan_control() -> None:
    state, action, frames = trial_arrays(_trial(6, [4, 5]), with_frames=True)
    assert state.shape == (4, 8) and action.shape == (4, 8) and state.dtype == np.float32
    assert {len(v) for v in frames.values()} == {4}
    assert trial_arrays(_trial(6, []), with_frames=False)[0].shape == (6, 8)
    with pytest.raises(ValueError):
        trial_arrays(_trial(6, [2]), with_frames=False)


def test_roboset_tasks_and_pins() -> None:
    assert roboset_task("heat_soup/scene_2/baking_close_oven_scene_2") == "Close the oven door."
    assert roboset_task("baking_prep/scene_1/baking_prep_pick_butter_scene_1") == (
        "Pick up the butter."
    )
    assert roboset_task("make_tea/scene_2/make_tea_place_tea_scene_2") == "Put down the tea box."
    assert len(ROBOSET_DIRS) == 38 and sum(f for f, _ in ROBOSET_DIRS.values()) == 740
    assert [f for f, _ in ROBOSET_DIRS.values()].count(25) == 24
    assert {ROBOSET_DIRS[d][0] for d in ROBOSET_SUBSET_DIRS} == {10, 25}
    eps = roboset_episodes()
    assert eps[min(ROBOSET_DIRS)].start == 0 and max(r.stop for r in eps.values()) == 9500
    assert len(RPT_SUBSET_EPISODES) == 187
    allow = catalog.pin("berkeley_rpt_delta", True).sources[0].allow  # type: ignore[union-attr]
    assert allow is not None and len(allow) == 1 + 2 * 187 and "meta/**" in allow
    assert "data/chunk-000/episode_000903.parquet" in allow
    for d in ROBOSET_DIRS:
        roboset_task(d)


def test_robosuite_fk_reference_pose() -> None:
    """robosuite 1.4.0 + mujoco 3.3.2 (the trained data's FK) on robosuite's Panda init pose."""
    pytest.importorskip("robosuite")
    fk, axis_angle = make_fk()
    joints = np.array([0.0, np.pi / 16, 0.0, -np.pi / 2 - np.pi / 3, 0.0, np.pi - 0.2, np.pi / 4])
    pos, rot = fk(joints)
    np.testing.assert_allclose(pos, [-0.10276829389898158, 0.0, 1.0110438715851882], atol=1e-7)
    np.testing.assert_allclose(
        axis_angle(rot), [2.15510630607605, 2.1561686992645264, 0.1373031735420227], atol=1e-6
    )
