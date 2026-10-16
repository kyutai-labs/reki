import pytest
import torch

from reki.policy.checkpoint import seed_mirrored_state_mlp

pytestmark = pytest.mark.cpu


def test_seed_mirrored_state_mlp():
    prefix = "model.vlm_with_expert.state_mlps."
    aloha = torch.tensor([1.0])
    panda_omron_mirrored = torch.tensor([3.0])
    state = {
        f"{prefix}aloha.weight": aloha,
        f"{prefix}panda_omron.weight": torch.tensor([2.0]),
        f"{prefix}panda_omron_mirrored.weight": panda_omron_mirrored,
    }
    model_keys = {f"{prefix}aloha_mirrored.weight", f"{prefix}panda_omron_mirrored.weight"}

    assert seed_mirrored_state_mlp(state, model_keys) == 1
    assert torch.equal(state[f"{prefix}aloha_mirrored.weight"], aloha)
    assert state[f"{prefix}aloha_mirrored.weight"] is not aloha
    assert state[f"{prefix}panda_omron_mirrored.weight"] is panda_omron_mirrored
    assert set(state) == {
        f"{prefix}aloha.weight",
        f"{prefix}aloha_mirrored.weight",
        f"{prefix}panda_omron.weight",
        f"{prefix}panda_omron_mirrored.weight",
    }
