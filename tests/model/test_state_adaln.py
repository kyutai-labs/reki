"""Per-frame proprio under register memory (``state_encoding="adaln"``): group member f carries
frame f's OWN state, not the anchor's. The streaming parity test only compares the last frame,
which IS the anchor, so an anchor-state leak into frames 0..K-2 is invisible there."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from conftest import robot_dataset
from lerobot.utils.constants import OBS_STATE

pytestmark = pytest.mark.hf


def test_each_frame_carries_its_own_state(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    tiny_lerobot_leaf("libero")
    ds = robot_dataset(
        "libero",
        state_encoding="adaln",
        action_expert_num_registers=8,
        history_k=3,
        image_size=224,
    )
    members = ds[25]["group_items"]
    assert [int(m["frame_index"]) for m in members] == [5, 15, 25]
    for m in members:
        own = ds[int(m["frame_index"])]["group_items"][-1]
        assert int(own["frame_index"]) == int(m["frame_index"])
        assert torch.equal(m[OBS_STATE], own[OBS_STATE])
    assert not torch.equal(members[0][OBS_STATE], members[-1][OBS_STATE])
