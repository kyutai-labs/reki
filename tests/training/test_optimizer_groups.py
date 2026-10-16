"""``build_optimizer_groups`` under the shipped scales: every trainable parameter lands in
exactly one group at its cohort's LR (a parameter missing from every group is silently frozen)."""

import pytest
import torch.nn as nn

from reki.policy.optim import build_optimizer_groups

pytestmark = pytest.mark.cpu

LR, BACKBONE = 1e-4, 0.1


class _VLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.vision = nn.Linear(8, 8)
        self.frozen = nn.Linear(8, 8).requires_grad_(False)
        self.action_expert = nn.Sequential(nn.Linear(8, 8), nn.RMSNorm(8))
        self.state_mlps = nn.ModuleDict({"single_arm_ee_euler": nn.Linear(8, 8)})


class _Policy(nn.Module):
    def __init__(self, vlm: _VLM, adaln_proj: nn.Linear) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vlm_with_expert = vlm
        self.model.state_adaln_proj = adaln_proj


def test_every_trainable_param_in_one_group_at_its_lr() -> None:
    vlm, adaln_proj = _VLM(), nn.Linear(8, 8)
    policy = _Policy(vlm, adaln_proj)
    groups, _ = build_optimizer_groups(
        policy,  # type: ignore[arg-type]
        vlm,  # type: ignore[arg-type]
        lr=LR,
        weight_decay=1e-8,
        backbone_lr_scale=BACKBONE,
    )
    lr_of: dict[int, float] = {}
    for g in groups:
        for p in g["params"]:
            assert id(p) not in lr_of, "a parameter sits in two groups"
            lr_of[id(p)] = g["lr"]

    expected = {
        vlm.backbone: LR * BACKBONE,
        vlm.vision: LR * BACKBONE,
        vlm.action_expert: LR,
        vlm.state_mlps: LR,
        adaln_proj: LR,
    }
    for module, lr in expected.items():
        for p in module.parameters():
            assert lr_of.pop(id(p)) == pytest.approx(lr)
    assert not lr_of, "unexpected parameters in the optimizer"


def test_none_proprio_modules_and_missing_attribute() -> None:
    vlm = _VLM()
    vlm.state_mlps = None  # type: ignore[assignment]
    policy = _Policy(vlm, None)  # type: ignore[arg-type]
    groups, _ = build_optimizer_groups(
        policy,  # type: ignore[arg-type]
        vlm,  # type: ignore[arg-type]
        lr=LR,
        weight_decay=0.0,
        backbone_lr_scale=BACKBONE,
    )
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert sum(len(g["params"]) for g in groups) == len(trainable)
    del vlm.action_expert
    with pytest.raises(AttributeError, match="action_expert"):
        build_optimizer_groups(policy, vlm, lr=LR, weight_decay=0.0)  # type: ignore[arg-type]
