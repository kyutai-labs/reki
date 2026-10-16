"""A checkpoint exported with the EMA merged (what ships on the hub) reloads with every key and
predicts exactly what the source checkpoint predicts with ``use_ema=True``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch as st
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("tiny_qwen")]


def _batch(member: dict) -> dict:
    from reki.policy.utils import to_device

    m = to_device(dict(member), torch.device("cuda"))
    return {
        OBS_LANGUAGE_TOKENS: m[OBS_LANGUAGE_TOKENS],
        "vision_inputs": [m["vision_inputs"]],
        "k_eff": [1],
        "action_dim": m[ACTION].shape[-1],
        "state_mlp_embodiment": [m["state_mlp_embodiment"]],
        OBS_STATE: [m[OBS_STATE]],
    }


def _load(ckpt: Path, use_ema: bool) -> Any:
    from reki.policy.checkpoint import load_pretrained_weights
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.train_config import TrainConfig

    cfg = TrainConfig.from_yaml(ckpt / "train_config.yaml")
    torch.manual_seed(1)
    policy = RekiModel(RekiConfig.from_train_config(cfg, device="cuda"))
    load_pretrained_weights(policy, ckpt, use_ema=use_ema)
    return policy.to("cuda").eval()


def test_export_then_reload_predicts_identically(
    synthetic_group: Callable[..., tuple[Any, dict]], tmp_path: Path
) -> None:
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.optim import EMAModel
    from scripts.export_checkpoint import export_checkpoint

    cfg, group = synthetic_group(k=1)
    torch.manual_seed(0)
    policy = RekiModel(RekiConfig.from_train_config(cfg, device="cuda"))
    ema = EMAModel(policy, decay=0.999)
    with torch.no_grad():
        for t in ema.shadow.values():
            t.add_(torch.randn_like(t) * 1e-2)

    ckpt = tmp_path / "checkpoint-1"
    policy.save_pretrained(ckpt)
    st.save_file(ema.state_dict(), str(ckpt / "ema.safetensors"))
    cfg.save_yaml(str(ckpt / "train_config.yaml"))
    out = tmp_path / "export"
    export_checkpoint(ckpt, out, merge_ema=True, source_rev="test")

    source, exported = _load(ckpt, use_ema=True), _load(out, use_ema=False)
    src_sd, exp_sd = source.state_dict(), exported.state_dict()
    assert set(src_sd) == set(exp_sd)
    for k, v in src_sd.items():
        assert torch.equal(v, exp_sd[k]), k

    batch = _batch(group["group_items"][-1])
    actions = []
    for policy_ in (source, exported):
        torch.manual_seed(0)
        actions.append(policy_.predict_action_chunk(batch))
    assert torch.equal(actions[0], actions[1])
