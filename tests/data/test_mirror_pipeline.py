"""Mirror augmentation on one synthetic item of each mirrorable dataset. A half-applied mirror
still trains without error, so every channel (state, action, cameras, embodiment, prompt
marker, stats) is checked on the same draw."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import TASK, robot_dataset
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
from reki.data.robotics.mirror import (
    ALOHA_ARX5_SPEC,
    EE_EULER_ACTION_SIGN,
    EE_EULER_STATE_SIGN,
    MIRROR_SPECS,
    mirror_aloha,
)
from reki.data.robotics.prompt import MIRROR_MARKER, normalize_task
from reki.data.robotics.registry import DATASETS, mirrored_name

pytestmark = pytest.mark.hf


def _item(
    name: str, mirror_prob: float, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[dict, list, Any]:
    """(anchor member, the camera images the member was built from, dataset)."""
    seen: list[list[torch.Tensor]] = []
    build = CombinedLeRobotDataset._build_register_group

    def spy(self: Any, item: dict, ds_idx: int, images: list, *a: Any) -> dict:
        seen.append([im.clone() for im in images])
        return build(self, item, ds_idx, images, *a)

    monkeypatch.setattr(CombinedLeRobotDataset, "_build_register_group", spy)
    ds = robot_dataset(
        name,
        state_encoding="adaln",
        action_expert_num_registers=8,
        image_size=224,
        mirror_prob=mirror_prob,
        **overrides,
    )
    member = ds[5]["group_items"][-1]
    monkeypatch.setattr(CombinedLeRobotDataset, "_build_register_group", build)
    return member, seen[-1], ds


def _sliced_mirror(name: str, x: torch.Tensor, feature: str) -> torch.Tensor:
    """The dataset's mirror on its sliced layout (the slice is closed under the permutation)."""
    ds_spec = DATASETS[name]
    assert ds_spec.mirror is not None
    spec = MIRROR_SPECS[ds_spec.mirror]
    keep = ds_spec.state_slice if feature == "state" else ds_spec.action_slice
    assert keep is not None and None not in keep
    keep = [i for i in keep if i is not None]
    perm, sign = (
        (spec.state_perm, spec.state_sign)
        if feature == "state"
        else (spec.action_perm, spec.action_sign)
    )
    pos = {raw: j for j, raw in enumerate(keep)}
    idx = [pos[perm[raw]] for raw in keep]
    return torch.tensor([sign[raw] for raw in keep], dtype=x.dtype) * x[..., idx]


@pytest.mark.parametrize("name", ["robotwin2_clean", "libero", "robocasa365"])
def test_every_channel_mirrors_together(
    name: str, tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_lerobot_leaf(name)
    plain, plain_imgs, ds = _item(name, 0.0, monkeypatch)
    mirr, mirr_imgs, _ = _item(name, 1.0, monkeypatch)

    assert mirr["is_mirrored"] is True and "is_mirrored" not in plain
    assert mirr["state_mlp_embodiment"] == mirrored_name(plain["state_mlp_embodiment"])

    def text(m: dict) -> str:
        return ds._preprocessor.tokenizer.decode(m[OBS_LANGUAGE_TOKENS])

    assert MIRROR_MARKER in text(mirr) and MIRROR_MARKER not in text(plain)
    assert MIRROR_MARKER + normalize_task(TASK) in text(mirr)

    if name == "robotwin2_clean":
        top, wl, wr = plain_imgs
        expected_imgs = [top.flip(-1), wr.flip(-1), wl.flip(-1)]
        torch.testing.assert_close(
            mirr[OBS_STATE], mirror_aloha(plain[OBS_STATE]), rtol=0, atol=1e-5
        )
        torch.testing.assert_close(mirr[ACTION], mirror_aloha(plain[ACTION]), rtol=0, atol=1e-5)
    elif name == "robocasa365":
        top, wrist, side = plain_imgs
        expected_imgs = [side.flip(-1), wrist.flip(-1), top.flip(-1)]
        for key, feature in ((OBS_STATE, "state"), (ACTION, "action")):
            torch.testing.assert_close(
                mirr[key], _sliced_mirror(name, plain[key], feature), rtol=0, atol=1e-5
            )
    else:
        expected_imgs = [im.flip(-1) for im in plain_imgs]
        s_sign = torch.tensor(EE_EULER_STATE_SIGN[:6], dtype=torch.float32)
        torch.testing.assert_close(
            mirr[OBS_STATE][:6], s_sign * plain[OBS_STATE][:6], rtol=0, atol=1e-5
        )
        torch.testing.assert_close(mirr[OBS_STATE][6:], plain[OBS_STATE][6:], rtol=0, atol=1e-5)
        a_sign = torch.tensor(EE_EULER_ACTION_SIGN, dtype=torch.float32)
        torch.testing.assert_close(mirr[ACTION], a_sign * plain[ACTION], rtol=0, atol=1e-5)
    assert len(mirr_imgs) == len(expected_imgs)
    for got, want in zip(mirr_imgs, expected_imgs):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_capped_pair_partner_keeps_its_history(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``max_cameras=2`` the robocasa365 side camera is outside the cap, but the mirror
    swaps it into the top slot: it must be fetched with the same K frames as the top camera."""
    tiny_lerobot_leaf("robocasa365")
    plain, plain_imgs, _ = _item("robocasa365", 0.0, monkeypatch, history_k=2, max_cameras=2)
    mirr, mirr_imgs, _ = _item("robocasa365", 1.0, monkeypatch, history_k=2, max_cameras=2)
    assert len(plain_imgs) == len(mirr_imgs) == 2
    for got, want in zip(mirr_imgs, plain_imgs):
        assert got.shape == want.shape


@pytest.mark.parametrize(
    ("item", "match"),
    [
        (
            {
                ALOHA_ARX5_SPEC.paired_cameras[0][0]: torch.zeros(2, 3, 8, 8),
                ALOHA_ARX5_SPEC.paired_cameras[0][1]: torch.zeros(3, 8, 8),
            },
            "different shapes",
        ),
        ({ALOHA_ARX5_SPEC.paired_cameras[0][0]: torch.zeros(3, 8, 8)}, "only one is present"),
    ],
)
def test_paired_cameras_that_cannot_swap_raise(item: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CombinedLeRobotDataset._mirror_sample(None, item, ALOHA_ARX5_SPEC)  # type: ignore[arg-type]
