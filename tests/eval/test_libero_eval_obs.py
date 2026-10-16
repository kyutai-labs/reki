"""The LIBERO env → EnvObs state path: the eval slice must match the libero state slice."""

from __future__ import annotations

import numpy as np
import pytest

from reki.data.robotics.registry import DATASETS
from reki.eval.rollout.libero import libero_obs_to_envobs, quat_xyzw_to_axisangle

pytestmark = pytest.mark.cpu

_POS = np.asarray([0.10, 0.20, 0.95], dtype=np.float32)
_QPOS = np.asarray([0.0400, -0.0398], dtype=np.float32)
_QUAT = np.asarray([0.5, -0.3, 0.4, 0.7071], dtype=np.float32)


class _Env:
    task_description = "pick up the black bowl on the left"


def _raw_obs() -> dict:
    """Shaped like the libero env's nested obs, with distinguishable 1x3 images."""
    return {
        "pixels": {
            "image": np.asarray([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]], dtype=np.uint8),
            "image2": np.asarray([[[7, 7, 7], [8, 8, 8], [9, 9, 9]]], dtype=np.uint8),
        },
        "robot_state": {
            "eef": {"pos": _POS.copy(), "quat": _QUAT.copy()},
            "gripper": {"qpos": _QPOS.copy()},
        },
    }


def test_state_is_sliced_like_training() -> None:
    obs = libero_obs_to_envobs(_Env(), _raw_obs())
    rot = quat_xyzw_to_axisangle(_QUAT).astype(np.float32)
    raw = np.concatenate([_POS, rot, _QPOS]).astype(np.float32)
    keep = [i for i in DATASETS["libero"].state_slice or [] if i is not None]
    np.testing.assert_allclose(obs.state_raw[[0, 1, 2]], _POS, rtol=0, atol=0)
    assert obs.state_raw[6] == 0.0
    np.testing.assert_allclose(obs.state_raw[7], raw[keep[-1]], rtol=0, atol=0)
    assert obs.task_text == _Env.task_description, "verbatim from the env"
