"""drop_static_wrist_camera (on in CPT): the wrist cam of a static arm is dropped."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch
from conftest import robot_dataset
from lerobot.utils.constants import ACTION

from reki.data.robotics.cameras import (
    CAMERA_KEYS,
    WRIST_LEFT,
    WRIST_RIGHT,
    reduce_cameras,
    wrist_static_flags,
)
from reki.data.robotics.registry import DATASETS, SINGLE_ARM

pytestmark = pytest.mark.cpu


def _chunk(dim: int, moving_dims: list[int], T: int = 10) -> torch.Tensor:
    x = torch.full((T, dim), 0.5)
    for d in moving_dims:
        x[:, d] = torch.linspace(0.0, 1.0, T)
    return x


def test_static_wrist_dropped_moving_wrist_kept():
    """aloha [La6,Lg,Ra6,Rg]: a static left arm drops its wrist cam, the moving right keeps its."""
    stats = {ACTION: {"mean": torch.zeros(14), "std": torch.ones(14)}}
    item = {ACTION: _chunk(14, moving_dims=[7, 8, 9]), "chunk_size": 10}
    left_static, right_static = wrist_static_flags(
        item,
        10,
        stats,
        arm_action_dims=(list(range(7)), list(range(7, 14))),
        control_mode="absolute joint",
        gripper_dims=[6, 13],
        threshold=0.1,
        dataset_name="ds",
    )
    ci, frames, keys = [0, 1, 2], ["top", "wrist_l", "wrist_r"], ["kt", "kl", "kr"]
    reduce_cameras(ci, frames, keys, False, True, left_static, right_static)
    assert ci == [0, 2] and frames == ["top", "wrist_r"] and keys == ["kt", "kr"]


@pytest.mark.parametrize("arm_action_dims", [None, SINGLE_ARM])
def test_single_arm_never_static(arm_action_dims: Literal["single_arm"] | None):
    """Without left/right arm_action_dims neither wrist is judged static."""
    stats = {ACTION: {"mean": torch.zeros(7), "std": torch.ones(7)}}
    item = {ACTION: _chunk(7, moving_dims=[]), "chunk_size": 10}
    flags = wrist_static_flags(
        item,
        10,
        stats,
        arm_action_dims=arm_action_dims,
        control_mode="delta end-effector",
        gripper_dims=None,
        threshold=0.1,
        dataset_name="ds",
    )
    assert flags == (False, False)


def test_arm_action_dims_iff_two_wrist_cameras():
    """Every dataset that maps both wrist cameras sets arm_action_dims, and no other does."""
    wrists = {CAMERA_KEYS[WRIST_LEFT], CAMERA_KEYS[WRIST_RIGHT]}
    bad = [
        name
        for name, spec in DATASETS.items()
        if (spec.arm_action_dims is not None) != (wrists <= set(spec.cameras.values()))
    ]
    assert not bad


def test_two_wrists_without_arm_dims_raises(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
):
    """drop_static_wrist_camera rejects a two-wrist dataset that can't be judged static."""
    tiny_lerobot_leaf("robotwin2_clean")
    spec = dataclasses.replace(DATASETS["robotwin2_clean"], arm_action_dims=None)
    monkeypatch.setitem(DATASETS, "robotwin2_clean", spec)
    robot_dataset("robotwin2_clean")
    with pytest.raises(ValueError, match="needs arm_action_dims"):
        robot_dataset(
            "robotwin2_clean", drop_static_wrist_camera=True, drop_extra_side_cameras=True
        )
