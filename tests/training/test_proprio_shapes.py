"""``check_proprio_shapes``: a state MLP whose shape differs from the checkpoint's raises."""

from __future__ import annotations

import pytest
import torch

from reki.policy.checkpoint import check_proprio_shapes

pytestmark = pytest.mark.cpu

BANK_W = "model.vlm_with_expert.state_mlps.franka.0.weight"


def test_resized_bank_raises() -> None:
    with pytest.raises(ValueError, match="state MLP shapes differ"):
        check_proprio_shapes({BANK_W: torch.randn(8, 7)}, {BANK_W: torch.empty(8, 9)})


def test_same_shapes_pass() -> None:
    check_proprio_shapes({BANK_W: torch.randn(8, 7)}, {BANK_W: torch.empty(8, 7)})
