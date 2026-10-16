"""Dataset normalize → eval decode is the identity inside the stats band and clamps outside it.

libero covers the euler sidecar (axis-angle → euler at load, euler → axis-angle at decode) and
``GRIPPER_FLIP["libero"]["action"]``; robotwin covers plain abs-joint quantile normalization.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import robot_dataset
from lerobot.utils.constants import ACTION

from reki.eval.rollout.libero import LiberoAdapter
from reki.eval.rollout.remote import RoboTwinAdapter

pytestmark = pytest.mark.hf


def _decoder(name: str, leaf: Path) -> Callable[[torch.Tensor], np.ndarray]:
    """The decode each eval applies to one normalized action, with the stats it loads."""
    import json

    if name == "libero":
        src = json.loads((leaf / "meta" / "norm_stats_euler.json").read_text())[ACTION]
        stats = {k: torch.as_tensor(v) for k, v in src.items()}
        libero = LiberoAdapter(None, None)
        libero.rot_to_euler = True
        return lambda a: libero.decode(a, stats)
    src = json.loads((leaf / "meta" / "stats.json").read_text())[ACTION]
    stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in src.items()}
    robotwin = RoboTwinAdapter("task", "ckpt", "127.0.0.1", 0)
    return lambda a: robotwin.decode(a, stats)


@pytest.mark.parametrize("name", ["libero", "robotwin2_clean"])
def test_normalize_then_decode_round_trips(
    name: str, tiny_lerobot_leaf: Callable[..., Path]
) -> None:
    leaf = tiny_lerobot_leaf(name)
    ds = robot_dataset(
        name,
        state_encoding="adaln",
        action_expert_num_registers=8,
        image_size=224,
    )
    member = ds[3]["group_items"][-1]
    norm = member[ACTION]
    assert not member["action_is_pad"].any()
    assert norm.abs().max() <= 1.0

    raw = pd.read_parquet(leaf / "data" / "chunk-000" / "file-000.parquet")
    ep = raw[raw.episode_index == int(member["episode_index"])]
    f0 = int(member["frame_index"])
    expected = np.stack(ep.action.to_numpy()[f0 : f0 + norm.shape[0]])

    decode = _decoder(name, leaf)
    got = np.stack([decode(row) for row in norm])
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-5)

    hi = decode(torch.full_like(norm[0], 1.5))
    np.testing.assert_allclose(hi, decode(torch.ones_like(norm[0])), rtol=0, atol=0)
