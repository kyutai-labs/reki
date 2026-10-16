"""``load_pretrained_weights`` tolerates missing keys only for the proprio modules (a warm-start
onto a new embodiment adds a fresh state MLP); any other missing key raises."""

from pathlib import Path

import pytest
import safetensors.torch as st
import torch.nn as nn

from reki.policy.checkpoint import load_pretrained_weights

pytestmark = pytest.mark.cpu


def _policy() -> nn.Module:
    policy = nn.Module()
    policy.model = nn.Module()
    policy.model.vlm_with_expert = nn.Module()
    policy.model.vlm_with_expert.state_mlps = nn.ModuleDict(
        {"single_arm_ee_euler": nn.Linear(4, 8), "aloha": nn.Linear(4, 8)}
    )
    policy.model.state_adaln_proj = nn.Linear(8, 8)
    policy.model.action_in_proj = nn.Linear(8, 8)
    return policy


def _ckpt(tmp_path: Path, drop: str) -> Path:
    sd = {k: v for k, v in _policy().state_dict().items() if not k.startswith(drop)}
    st.save_file(sd, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_missing_state_mlp_is_a_notice(tmp_path: Path) -> None:
    load_pretrained_weights(_policy(), _ckpt(tmp_path, "model.vlm_with_expert.state_mlps.aloha."))


def test_missing_non_proprio_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model.action_in_proj.weight"):
        load_pretrained_weights(_policy(), _ckpt(tmp_path, "model.action_in_proj."))
