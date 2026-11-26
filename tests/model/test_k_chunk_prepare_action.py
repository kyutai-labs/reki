"""``RekiModel.prepare_action`` K-chunk unrolling: the target of a chunk depends only on its
frame, padded episode-start chunks are dropped, time-padded steps and invalid dims
(``ACTION_DIM_VALID``) get no supervision."""

from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import ACTION

from reki.policy.constants import ACTION_DIM_VALID, ACTION_IS_PAD
from reki.policy.modeling import RekiModel

pytestmark = pytest.mark.cpu


def _fake_policy(max_action_dim: int = 8) -> SimpleNamespace:
    """Minimal stub exposing the config fields ``prepare_action`` reads."""
    return SimpleNamespace(config=SimpleNamespace(max_action_dim=max_action_dim))


def _batch(
    actions: list[torch.Tensor],
    k_effs: list[int],
    chunk: int,
    pads: list[torch.Tensor | None] | None = None,
) -> dict:
    return {
        ACTION: actions,
        "k_eff": k_effs,
        "chunk_size": [chunk] * len(actions),
        ACTION_IS_PAD: pads if pads is not None else [None] * len(actions),
    }


class TestPrepareAction:
    def test_drops_pad_replicated_prefix_chunks(self) -> None:
        chunk, K_max, k_eff = 4, 6, 3
        per_chunk = [torch.full((chunk, 2), float(k)) for k in range(K_max)]
        a = torch.cat(per_chunk, dim=0)
        batch = _batch([a], [k_eff], chunk)
        actions, _, _, anchor, _, _ = RekiModel.prepare_action(_fake_policy(4), batch)  # type: ignore[arg-type]
        assert actions.shape == (k_eff * chunk, 4)
        assert anchor.tolist() == [0, 1, 2]
        for k in range(k_eff):
            expected = float(K_max - k_eff + k)
            assert (actions[k * chunk : (k + 1) * chunk, :2] == expected).all()

    def test_variable_k_eff_packing(self) -> None:
        chunk = 4
        a0 = torch.cat([torch.full((chunk, 2), float(k)) for k in range(3)], dim=0)
        a1 = torch.cat([torch.full((chunk, 2), float(k + 100)) for k in range(2)], dim=0)
        batch = _batch([a0, a1], [3, 2], chunk)
        actions, _, owner, anchor, cs, _ = RekiModel.prepare_action(_fake_policy(4), batch)  # type: ignore[arg-type]
        assert actions.shape == (5 * chunk, 4)
        assert owner.tolist() == [0, 0, 0, 1, 1]
        assert anchor.tolist() == [0, 1, 2, 0, 1]
        assert cs.tolist() == [chunk] * 5

    def test_time_pad_truncates_padded_steps(self) -> None:
        """action_is_pad is a contiguous suffix → padded steps are truncated away
        (variable per-chunk cs), leaving only fully-valid rows in the packed stream."""
        chunk, K = 4, 2
        a = torch.ones(K * chunk, 2)
        pad = torch.tensor([False] * chunk + [False, False, True, True])
        batch = _batch([a], [K], chunk, pads=[pad])
        actions, mask, _, anchor, cs, _ = RekiModel.prepare_action(_fake_policy(4), batch)  # type: ignore[arg-type]
        assert anchor.tolist() == [0, 1]
        assert cs.tolist() == [chunk, 2]
        assert actions.shape == (chunk + 2, 4)
        assert mask[:, :2].all()
        assert not mask[:, 2:].any()

    def test_invalid_dims_leave_the_loss_mask_only(self) -> None:
        """An untracked hand: its dims are zeroed and out of the loss mask, but keep the noise
        mask (the expert input looks like inference); padded time steps cut both."""
        chunk, K = 4, 2
        a = torch.full((K * chunk, 3), 5.0)
        valid = torch.ones(K * chunk, 3, dtype=torch.bool)
        valid[1:3, 1] = False
        valid[6, 0] = False
        pad = torch.tensor([False] * chunk + [False, False, True, True])
        batch = {**_batch([a], [K], chunk, pads=[pad]), ACTION_DIM_VALID: [valid]}
        actions, mask, _, _, cs, loss_mask = RekiModel.prepare_action(_fake_policy(4), batch)  # type: ignore[arg-type]
        assert cs.tolist() == [chunk, 2]
        assert mask[:, :3].all() and not mask[:, 3].any()
        expected = valid[:6].clone()
        assert torch.equal(loss_mask[:, :3], expected) and not loss_mask[:, 3].any()
        assert torch.equal(actions[:, :3], torch.where(expected, 5.0, 0.0))
        no_key = _batch([a], [K], chunk, pads=[pad])
        _, _, _, _, _, all_valid = RekiModel.prepare_action(_fake_policy(4), no_key)  # type: ignore[arg-type]
        assert torch.equal(all_valid, mask)
