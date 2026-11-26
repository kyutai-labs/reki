"""The pretraining-only paths of the forward, on a tiny random Qwen3.5 (``tiny_qwen``):

- an assistant-turn subtask (``predict_subtask``) does not move the action expert: the flow
  loss of each frame is the one of the same group without the subtask turn;
- a batch of VL samples only gives the text CE, and every action and state parameter still gets
  a (zero) gradient, so the DDP unused-parameter set is the same on every rank.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS
from test_register_memory_gpu import (
    DEVICE,
    SEED,
    _frame_bounds,
    _per_frame_losses,
    _policy_and_batch,
)

from reki.policy.constants import IGNORE_INDEX, RegionTag

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("tiny_qwen")]


def _subtask_policy(synthetic_group: Callable[..., Any], tmp_path: Path) -> tuple[Any, dict, dict]:
    """A K=3 policy and batch with an assistant-turn subtask on every frame."""
    lookup = [{"episode_index": e, "intervals": [[0, 40, "pick up the white mug"]]} for e in (0, 1)]
    meta = tmp_path / "HuggingFaceVLA" / "libero" / "meta"
    (meta / "subtask_lookup.jsonl").write_text("\n".join(map(json.dumps, lookup)))
    return _policy_and_batch(
        synthetic_group, predict_subtask=True, subtask_lookup_filename="subtask_lookup.jsonl"
    )


def _without_subtask(member: dict) -> dict:
    """The frame without its subtask turn (the suffix after the user turn)."""
    n = int((member["region_tags"] == RegionTag.USER).sum())
    assert bool((member["region_tags"][:n] == RegionTag.USER).all())
    out = {k: v for k, v in member.items() if k != "labels"}
    out[OBS_LANGUAGE_TOKENS] = member[OBS_LANGUAGE_TOKENS][..., :n]
    out["region_tags"] = member["region_tags"][:n]
    vision = dict(member["vision_inputs"])
    vision["mm_token_type_ids"] = vision["mm_token_type_ids"][..., :n]
    out["vision_inputs"] = vision
    return out


def test_assistant_subtask_does_not_move_the_expert(
    synthetic_group: Callable[..., Any], tmp_path: Path
) -> None:
    from reki.data.robotics.collate import combined_collate
    from reki.policy.utils import to_device

    policy, batch, group = _subtask_policy(synthetic_group, tmp_path)
    assert bool((batch["region_tags"] == RegionTag.SUBTASK).any())
    plain = combined_collate([{"group_items": [_without_subtask(m) for m in group["group_items"]]}])
    plain = to_device(plain, DEVICE)
    actions, cu = _frame_bounds(policy, batch)
    torch.manual_seed(SEED)
    noise = torch.randn(actions.shape, device=DEVICE)
    time = torch.rand(cu.shape[0] - 1, device=DEVICE)
    with_subtask = _per_frame_losses(policy, batch, noise, time)
    without = _per_frame_losses(policy, plain, noise, time)
    torch.testing.assert_close(with_subtask, without, rtol=1e-3, atol=1e-5)


def test_vl_only_batch(synthetic_group: Callable[..., Any]) -> None:
    from reki.data.robotics.collate import combined_collate
    from reki.policy.utils import to_device

    policy, _, group = _policy_and_batch(synthetic_group)
    vl_items = []
    for member in group["group_items"][:2]:
        tokens = member[OBS_LANGUAGE_TOKENS]
        labels = tokens.clone()
        labels[..., : tokens.shape[-1] // 2] = IGNORE_INDEX
        vl_items.append(
            {
                OBS_LANGUAGE_TOKENS: tokens,
                "vision_inputs": member["vision_inputs"],
                "n_image_tokens": member["n_image_tokens"],
                "labels": labels,
                "camera_indices": torch.tensor([], dtype=torch.long),
            }
        )
    batch = to_device(combined_collate(vl_items), DEVICE)
    assert batch[ACTION] == [None, None] and "region_tags" not in batch
    policy.zero_grad(set_to_none=True)
    loss, loss_dict = policy.forward(batch)
    assert "flow_loss" not in loss_dict
    torch.testing.assert_close(loss_dict["ce_loss"], loss_dict["text_loss"], rtol=0, atol=0)
    ce_term = (policy.config.ce_loss_weight * loss_dict["ce_loss"]).float()
    torch.testing.assert_close(loss.detach(), ce_term, rtol=0, atol=0)
    loss.backward()
    model = policy.model
    action_side = [
        model.action_in_proj,
        model.action_out_proj,
        model.time_mlp_in,
        model.time_mlp_out,
        model.vlm_with_expert.action_expert,
        model.state_adaln_proj,
        *model.vlm_with_expert.state_mlps.values(),
    ]
    for module in action_side:
        for p in module.parameters():
            assert p.grad is not None and not p.grad.any()
    lm_head = model.vlm_with_expert.vlm.lm_head.weight
    assert lm_head.grad is not None and lm_head.grad.any()
