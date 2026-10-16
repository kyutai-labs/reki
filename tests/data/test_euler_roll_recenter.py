"""``recenter_euler_roll``: the LIBERO state roll is near ±π (downward gripper), where euler xyz
wraps; the π shift removes the jump, and training and eval apply it alike."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import robot_dataset
from lerobot.utils.constants import OBS_STATE
from scipy.spatial.transform import Rotation as R

from reki.data.robotics.mirror import EE_EULER_STATE_PERM, EE_EULER_STATE_SIGN, mirror_vector
from reki.data.robotics.transforms import axisangle_to_euler, recenter_euler_roll
from reki.eval.rollout.libero import LiberoAdapter, libero_obs_to_envobs, quat_xyzw_to_axisangle
from reki.inference.spec import leaf_norm_stats

pytestmark = pytest.mark.cpu

# a downward gripper whose roll crosses π
_ROT = R.from_euler(
    "xyz", np.stack([np.pi + np.linspace(-0.2, 0.2, 41), np.full(41, 0.1), np.full(41, -0.3)], 1)
)
_POS = np.asarray([0.10, 0.20, 0.95], dtype=np.float32)
_QPOS = np.asarray([0.0400, -0.0398], dtype=np.float32)


class _Env:
    task_description = "put the white mug on the left plate"


def _raw_obs(quat: np.ndarray) -> dict:
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    return {
        "pixels": {"image": img, "image2": img},
        "robot_state": {"eef": {"pos": _POS, "quat": quat}, "gripper": {"qpos": _QPOS}},
    }


def _dataset_state(quat: np.ndarray) -> torch.Tensor:
    """The raw libero ``observation.state`` of one eval obs."""
    aa = quat_xyzw_to_axisangle(quat.astype(np.float32))
    return torch.as_tensor(np.concatenate([_POS, aa, _QPOS]), dtype=torch.float32)


def _libero(
    leaf_maker: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[Any, Path]:
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    leaf = leaf_maker("libero")
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    return robot_dataset("libero", **overrides), leaf


def test_recenter_removes_the_roll_wrap() -> None:
    state = torch.zeros(len(_ROT), 8)
    state[:, 3:6] = torch.as_tensor(_ROT.as_rotvec(), dtype=torch.float32)
    plain = axisangle_to_euler(state)
    assert plain[:, 3].diff().abs().max() > np.pi, "the plain roll jumps -π ↔ π"

    shifted = recenter_euler_roll(plain.clone())
    assert shifted[:, 3].diff().abs().max() < 0.02
    expected = (_ROT * R.from_euler("x", np.pi)).as_euler("xyz")
    np.testing.assert_allclose(shifted[:, 3:6].numpy(), expected, atol=1e-5)

    mirrored = mirror_vector(state, perm=EE_EULER_STATE_PERM, sign=EE_EULER_STATE_SIGN)
    torch.testing.assert_close(
        recenter_euler_roll(axisangle_to_euler(mirrored))[:, 3:6],
        shifted[:, 3:6] * torch.tensor(EE_EULER_STATE_SIGN[3:6], dtype=torch.float32),
        rtol=0,
        atol=1e-5,
    )


def test_training_and_eval_agree(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, leaf = _libero(tiny_lerobot_leaf, monkeypatch)
    adapter = LiberoAdapter(None, None)
    adapter.configure(ds.cfg)
    assert adapter.recenter_roll
    for quat in _ROT.as_quat():
        item = {OBS_STATE: _dataset_state(quat)}
        ds._state_action_to_model_layout(item, 0)
        env = libero_obs_to_envobs(
            _Env(), _raw_obs(quat), convert_rot_to_euler=True, recenter_roll=adapter.recenter_roll
        )
        np.testing.assert_allclose(item[OBS_STATE].numpy(), env.state_raw, rtol=0, atol=1e-6)
        assert abs(env.state_raw[3]) < 0.25

    sidecar = json.loads((leaf / "meta" / "norm_stats_euler_roll.json").read_text())
    train_state = ds.stats[0][OBS_STATE]
    np.testing.assert_allclose(train_state["qk01"][:6], sidecar[OBS_STATE]["qk01"][:6])
    eval_state, _ = leaf_norm_stats("libero", leaf, ds.cfg)
    for k, v in eval_state.items():
        torch.testing.assert_close(v, train_state[k].to(torch.float32), rtol=0, atol=0)


def test_mirrored_stats_follow_the_shift(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, _ = _libero(
        tiny_lerobot_leaf,
        monkeypatch,
        mirror_prob=0.5,
        state_encoding="adaln",
        state_mlp_embodiments={"single_arm_ee_euler": 8, "single_arm_ee_euler_mirrored": 8},
    )
    base, mirrored = ds.stats[0][OBS_STATE], ds.mirror_stats[0][OBS_STATE]
    torch.testing.assert_close(mirrored["qk01"][3], -base["qk99"][3])
    torch.testing.assert_close(mirrored["qk99"][3], -base["qk01"][3])
