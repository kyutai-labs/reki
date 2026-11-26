"""``val_flow_loss`` seeds a forked RNG: validation is reproducible and the training RNG is untouched."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from reki.policy.val import val_flow_loss

pytestmark = pytest.mark.cpu

SEED = 7


class _NoisyFlow(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[dict, dict[str, Any]]:
        tau = torch.rand(batch["x"].shape[0], 1)
        noise = torch.rand_like(batch["x"])
        return {}, {"flow_loss": ((tau * noise - batch["x"]) ** 2).mean()}


def test_val_flow_loss_does_not_reseed_global_rng() -> None:
    policy, batches, device = _NoisyFlow(), [{"x": torch.ones(4, 3)}] * 2, torch.device("cpu")
    torch.manual_seed(SEED)
    after_seed = torch.rand(8)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    first = val_flow_loss(policy, batches, device, seed=SEED)
    assert torch.equal(torch.get_rng_state(), state)
    second = val_flow_loss(policy, batches, device, seed=SEED)
    assert first is not None and second is not None and torch.equal(first, second)
    assert not torch.equal(torch.rand(8), after_seed)
