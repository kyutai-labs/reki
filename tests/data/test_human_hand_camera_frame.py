"""Parity test for the human-hand camera-frame wrist compose (QwenVLA/VITRA ego rep).

`wrist_to_camera_frame` must express each stored absolute-world wrist pose in the observation camera
frame as the full relative SE(3) `T_cam^-1 T_wrist`, matching a direct 4x4-matrix computation.
Eigengrasp / non-wrist dims pass through untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from reki.data.robotics.transforms import HUMAN_HAND_WRIST_STARTS, wrist_to_camera_frame

pytestmark = pytest.mark.cpu


def _T(t: np.ndarray, aa: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_rotvec(aa).as_matrix()
    M[:3, 3] = t
    return M


def test_matches_direct_se3():
    rng = np.random.default_rng(0)
    n = 12
    cam = np.concatenate([rng.normal(0, 0.3, 3), rng.normal(0, 1.0, 3)])
    action = np.zeros((n, 32), np.float64)
    abs_pose = {}
    for hand, s in enumerate(HUMAN_HAND_WRIST_STARTS):
        t = rng.normal(0, 0.3, (n, 3))
        aa = rng.normal(0, 1.0, (n, 3))
        action[:, s : s + 3] = t
        action[:, s + 3 : s + 6] = aa
        action[:, s + 6 : s + 16] = rng.normal(0, 1, (n, 10))
        abs_pose[hand] = (t, aa)
    eig_in = action[:, [*range(6, 16), *range(22, 32)]].copy()

    out = wrist_to_camera_frame(torch.tensor(action), torch.tensor(cam)).numpy()

    Tc_inv = np.linalg.inv(_T(cam[:3], cam[3:6]))
    for hand, s in enumerate(HUMAN_HAND_WRIST_STARTS):
        t, aa = abs_pose[hand]
        for i in range(n):
            rel = Tc_inv @ _T(t[i], aa[i])
            assert np.allclose(out[i, s : s + 3], rel[:3, 3], atol=1e-6), (hand, i, "transl")
            got = R.from_rotvec(out[i, s + 3 : s + 6]).as_matrix()
            assert np.allclose(got, rel[:3, :3], atol=1e-6), (hand, i, "rot")
    assert np.allclose(out[:, [*range(6, 16), *range(22, 32)]], eig_in, atol=1e-9)


def test_untracked_rows_stay_zero_and_leave_the_mask():
    """A stored all-zero hand row (untracked) is not composed into ``T_cam^-1``'s origin; the
    dataset marks its 16 dims invalid (``ACTION_DIM_VALID``), and an all-tracked chunk gets
    no key."""
    from types import SimpleNamespace

    from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
    from reki.data.robotics.registry import RELATIVE_EE
    from reki.data.robotics.transforms import tracked_hand_dims
    from reki.policy.constants import ACTION, ACTION_DIM_VALID, OBS_STATE

    rng = np.random.default_rng(1)
    action = torch.tensor(rng.normal(size=(10, 32)))
    action[2:5, 0:16] = 0
    action[7, 16:32] = 0
    cam = torch.tensor(rng.normal(size=6))
    out = wrist_to_camera_frame(action, cam)
    assert not out[2:5, 0:16].any() and not out[7, 16:32].any()
    assert out[[0, 1, 5, 6, 8, 9], 0:16].all(dim=1).any()
    valid = tracked_hand_dims(out)
    expected = torch.ones(10, 32, dtype=torch.bool)
    expected[2:5, 0:16] = False
    expected[7, 16:32] = False
    assert torch.equal(valid, expected)

    stub = SimpleNamespace(
        _specs=[SimpleNamespace(state_slice=None, action_slice=None, embodiment=None)],
        _action_rot_reps=[None],
        _state_rot_reps=[None],
        _control_modes=[RELATIVE_EE],
        _action_paces=[0.6],
    )
    state = torch.zeros(12, dtype=torch.float64)
    state[:6] = cam
    item = {ACTION: action.clone(), OBS_STATE: state}
    CombinedLeRobotDataset._state_action_to_model_layout(stub, item, 0)  # type: ignore[arg-type]
    assert torch.equal(item[ACTION_DIM_VALID], tracked_hand_dims(item[ACTION]))
    assert not item[ACTION_DIM_VALID].all()
    tracked = {ACTION: torch.tensor(rng.normal(size=(10, 32))), OBS_STATE: state}
    CombinedLeRobotDataset._state_action_to_model_layout(stub, tracked, 0)  # type: ignore[arg-type]
    assert ACTION_DIM_VALID not in tracked


def test_history_chunks_use_their_own_frame():
    """With ``history_k`` frames (state ``[k, D]``), each of the k stacked chunks is paced on its
    own and expressed in the camera of its own frame."""
    from reki.data.robotics.transforms import interpolate_wrist_chunk, wrist_action_in_camera_frame

    rng = np.random.default_rng(2)
    k, cs, pace = 3, 10, 0.6
    action = torch.tensor(rng.normal(size=(k * cs, 32)))
    state = torch.tensor(rng.normal(size=(k, 12)))
    out = wrist_action_in_camera_frame(action, state, pace)
    for f in range(k):
        chunk = interpolate_wrist_chunk(action[f * cs : (f + 1) * cs], pace)
        want = wrist_to_camera_frame(chunk, state[f, :6])
        torch.testing.assert_close(out[f * cs : (f + 1) * cs], want, rtol=0, atol=0)


def test_ego_history_needs_state_history():
    """Without per-frame state (``state_encoding`` other than adaln), ego history raises."""
    from reki.data.robotics.factory import validate_ego_history
    from reki.policy.train_config import TrainConfig

    with pytest.raises(ValueError, match="no per-frame camera pose"):
        validate_ego_history(TrainConfig(history_k=3, state_encoding="none"), ["libero", "egodex"])
    validate_ego_history(TrainConfig(history_k=3, state_encoding="adaln"), ["egodex"])
    validate_ego_history(TrainConfig(history_k=1, state_encoding="none"), ["egodex"])
    validate_ego_history(TrainConfig(history_k=3, state_encoding="none"), ["libero"])
