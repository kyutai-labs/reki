"""``RekiConfig.from_train_config`` copies every field the two configs share, so no
TrainConfig flag silently builds a model with the bare RekiConfig default."""

from __future__ import annotations

import dataclasses as dc
from typing import Any

import pytest

from reki.policy.configuration import RekiConfig
from reki.policy.train_config import TrainConfig

pytestmark = pytest.mark.cpu

_EXPERT_DIMS: dict[str, Any] = dict(
    action_expert_hidden_size=768,
    action_expert_intermediate_size=3072,
    action_expert_num_layers=10,
    action_expert_num_heads=12,
)


def test_from_train_config_no_field_falls_back_to_rekiconfig_default():
    """Any shared field whose TrainConfig value differs from RekiConfig's bare
    default must be copied; otherwise it silently takes the RekiConfig default."""
    cfg = TrainConfig(**_EXPERT_DIMS, action_expert_num_registers=32, history_k=5)
    vc = RekiConfig.from_train_config(cfg, device="cpu")
    shared = {f.name for f in dc.fields(RekiConfig) if f.init} & {
        f.name for f in dc.fields(TrainConfig)
    }
    reki_defaults = {f.name: f.default for f in dc.fields(RekiConfig)}
    checked = []
    for name in sorted(shared - {"device"}):
        tval = getattr(cfg, name)
        if tval != reki_defaults.get(name):
            assert getattr(vc, name) == tval, f"{name} not threaded from TrainConfig"
            checked.append(name)
    assert "history_k" in checked
