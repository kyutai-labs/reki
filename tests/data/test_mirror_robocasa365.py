"""Geometry tests for the RoboCasa365 (``panda_omron``) mirror.

Poses are rebuilt as rotation matrices and compared with ``M R M``, so a test cannot agree
with a sign typo. Every single-slot state sign flip must fail the check.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reki.data.robotics.mirror import ROBOCASA365_SPEC, MirrorSpec
from reki.data.robotics.registry import DATASETS

pytestmark = pytest.mark.cpu

M = np.diag([1.0, -1.0, 1.0])


def _assert_state_geometry(spec: MirrorSpec, x: np.ndarray) -> None:
    y = spec.mirror(x, "state")
    np.testing.assert_allclose(y[:3], M @ x[:3])
    np.testing.assert_allclose(
        Rotation.from_quat(y[3:7]).as_matrix(), M @ Rotation.from_quat(x[3:7]).as_matrix() @ M
    )
    np.testing.assert_allclose(y[7:10], M @ x[7:10])
    np.testing.assert_allclose(
        Rotation.from_quat(y[10:14]).as_matrix(),
        M @ Rotation.from_quat(x[10:14]).as_matrix() @ M,
    )
    assert y[6] >= 0 and y[13] >= 0
    np.testing.assert_allclose(y[14] - y[15], x[14] - x[15])


def test_state_spec_reflects_pose_geometry() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=16)
    x[3:7] = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_quat()
    x[10:14] = Rotation.random(random_state=rng).as_quat()
    if x[13] < 0:
        x[10:14] *= -1
    _assert_state_geometry(ROBOCASA365_SPEC, x)

    x[3:7] = Rotation.random(random_state=rng).as_quat()
    if x[6] < 0:
        x[3:7] *= -1
    for slot in range(16):
        sign = list(ROBOCASA365_SPEC.state_sign)
        sign[slot] *= -1
        mutant = MirrorSpec(
            "mutant",
            ROBOCASA365_SPEC.state_perm,
            tuple(sign),
            ROBOCASA365_SPEC.action_perm,
            ROBOCASA365_SPEC.action_sign,
        )
        try:
            _assert_state_geometry(mutant, x)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"state sign mutant at slot {slot} survived")


def test_action_spec_reflects_twist_and_delta() -> None:
    x = np.array([0.4, -0.3, 0.7, 0.2, -1.0, 0.1, 0.5, -0.2, 0.3, -0.4, 0.6, 1.0])
    y = ROBOCASA365_SPEC.mirror(x, "action")
    np.testing.assert_allclose(y[:2], (M @ np.r_[x[:2], 0.0])[:2])
    assert y[2] == -x[2]
    assert y[3] == x[3]
    assert y[4] == x[4]
    np.testing.assert_allclose(y[5:8], M @ x[5:8])
    expected = M @ Rotation.from_rotvec(x[8:11]).as_matrix() @ M
    np.testing.assert_allclose(Rotation.from_rotvec(y[8:11]).as_matrix(), expected)
    assert y[11] == x[11]


def test_agentview_cameras_are_the_declared_mirror_pair() -> None:
    # robocasa/utils/camera_utils.py, CAM_CONFIGS["DEFAULT"].
    pos_l = np.array([-0.5, 0.35, 1.05])
    pos_r = np.array([-0.5, -0.35, 1.05])
    quat_l = np.array([0.55624, 0.29935, -0.37679, -0.67751])
    quat_r = np.array([0.67751, 0.37679, -0.29935, -0.55624])
    rot_l = Rotation.from_quat(np.roll(quat_l, -1)).as_matrix()
    rot_r = Rotation.from_quat(np.roll(quat_r, -1)).as_matrix()
    np.testing.assert_allclose(M @ pos_l, pos_r, atol=1e-4)
    np.testing.assert_allclose(M @ rot_l @ np.diag([-1.0, 1.0, 1.0]), rot_r, atol=1e-4)

    cam_map = DATASETS["robocasa365"].cameras
    assert ROBOCASA365_SPEC.paired_cameras == (
        (
            cam_map["observation.images.robot0_agentview_left"],
            cam_map["observation.images.robot0_agentview_right"],
        ),
    )
    wrist = cam_map["observation.images.robot0_eye_in_hand"]
    assert all(wrist not in pair for pair in ROBOCASA365_SPEC.paired_cameras)
