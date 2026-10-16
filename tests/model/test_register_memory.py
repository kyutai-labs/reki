"""Register memory structure (CPU): collate group explosion, the memory index set each frame
reads, the owner-block layout of the expert suffix, and the inference register cache."""

from __future__ import annotations

from typing import Any

import pytest
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS

from reki.data.robotics.collate import combined_collate

pytestmark = pytest.mark.cpu


def _tok_item(n_tokens: int, n_action: int | None = None, n_img: int = 0) -> dict:
    import torch

    item: dict = {
        OBS_LANGUAGE_TOKENS: torch.arange(n_tokens),
        "vision_inputs": {},
        "camera_indices": torch.tensor([0]),
        "k_eff": 1,
        "n_image_tokens": n_img,
    }
    if n_action is not None:
        item[ACTION] = torch.zeros(n_action, 4)
        item["chunk_size"] = n_action
    return item


def _tl(x: Any) -> list:
    return x.tolist()


def test_collate_explodes_groups():
    batch = [
        {"group_items": [_tok_item(10, 5), _tok_item(12, 5), _tok_item(11, 5)]},
        _tok_item(7),
        {"group_items": [_tok_item(9, 5)]},
    ]
    out = combined_collate(batch)
    assert _tl(out["seq_lens"]) == [10, 12, 11, 7, 9]
    assert _tl(out["group_index"]) == [0, 0, 0, 1, 2]
    assert _tl(out["frame_pos"]) == [0, 1, 2, 0, 0]
    assert len(out[OBS_LANGUAGE_TOKENS]) == 10 + 12 + 11 + 7 + 9


def test_register_memory_indices_multi_group_and_draws():
    import torch

    from reki.policy.modeling import build_register_memory_indices

    fp = torch.tensor([0, 1, 0])
    gi = torch.tensor([0, 0, 1])
    cs_expert = torch.tensor([5, 5, 5, 5, 5, 5])
    flat, counts = build_register_memory_indices(cs_expert, fp, gi, n_reg=2, n_draws=2)
    assert counts.tolist() == [2, 2, 4, 4, 2, 2]
    got = flat.tolist()
    assert got[:2] == [0, 1]
    assert got[2:4] == [5, 6]
    assert got[4:8] == [0, 1, 10, 11]
    assert got[8:12] == [5, 6, 15, 16]
    assert got[12:14] == [20, 21] and got[14:16] == [25, 26]


def test_prepend_shared_register_prefix_layout():
    """Owner blocks [R ∥ draw_0..draw_{k-1}]: registers once per owner, action rows
    kept in draw-tiled order, adarms zero on register rows."""
    import torch

    from reki.policy.modeling import _prepend_registers

    n_reg, k = 2, 2
    cs_action = torch.tensor([3, 3, 4, 4], dtype=torch.int32)
    n_act = int(cs_action.sum())
    hidden = 8
    action_emb = torch.arange(n_act, dtype=torch.float32).unsqueeze(1).expand(n_act, hidden)
    adarms = torch.ones(n_act, hidden)
    reg = torch.full((n_reg, hidden), -1.0)

    suffix, adarms_out, cs_e, is_register, act_idx = _prepend_registers(
        action_emb, adarms, reg, cs_action, k
    )
    assert cs_e.tolist() == [2 + 6, 2 + 8]
    assert is_register.tolist() == [True] * 2 + [False] * 6 + [True] * 2 + [False] * 8
    assert torch.equal(suffix[act_idx], action_emb)
    assert torch.equal(suffix[is_register], reg.repeat(2, 1))
    assert torch.equal(adarms_out[act_idx], adarms)
    assert torch.equal(adarms_out[is_register], torch.zeros(4, hidden))


def test_register_cache_ring_buffer():
    """RegisterCache keeps the last `capacity` PREVIOUS frames, right-aligned and oldest first,
    in fixed-size buffers updated in place; the memory set for a frame = buffer + current ≤
    history_k. Every rollout longer than K evicts."""
    import torch

    from reki.vlm.common import RegisterCache

    R, H, D, n_layers = 2, 1, 4, 3
    cache = RegisterCache(n_layers, n_reg=R, capacity=2, n_heads=H, head_dim=D, dtype=torch.float32)
    k_buf, v_buf = cache.k, cache.v
    assert k_buf.shape == (n_layers, 2 * R, H, D)

    def frame(val: float):
        return torch.full((n_layers, R, H, D), val), torch.full((n_layers, R, H, D), -val)

    def frames() -> list[float]:
        n = cache.n_frames
        assert int(cache.n_valid) == n
        valid = cache.k[0, cache.k.shape[1] - n * R :, 0, 0]
        assert torch.equal(cache.v[0, cache.k.shape[1] - n * R :, 0, 0], -valid)
        return valid[::R].tolist()

    assert frames() == []
    cache.append_frame(*frame(1.0))
    assert frames() == [1.0]
    cache.append_frame(*frame(2.0))
    assert frames() == [1.0, 2.0]
    cache.append_frame(*frame(3.0))
    assert frames() == [2.0, 3.0]
    assert cache.k is k_buf and cache.v is v_buf
    cache.reset()
    assert frames() == []
    cache.append_frame(*frame(4.0))
    assert frames() == [4.0]


def test_modulated_norm_matches_adarms():
    """Inference applies AdaRMS with the modulation precomputed once per step and broadcast
    over rows; it must equal the norm called with the per-row cond."""
    import torch

    from reki.vlm.adarms import RegisterAdaRMSNorm, modulated_norm

    torch.manual_seed(0)
    norm = RegisterAdaRMSNorm(16, cond_dim=8).to(torch.bfloat16)
    torch.nn.init.normal_(norm.dense.weight)  # type: ignore[union-attr]
    torch.nn.init.normal_(norm.register_weight)
    x = torch.randn(6, 16, dtype=torch.bfloat16)
    cond = torch.randn(1, 8, dtype=torch.bfloat16)
    is_register = torch.arange(6) < 2
    mod = norm.dense(cond)  # type: ignore[misc]
    for prefix in (None, is_register):
        want = norm(
            x,
            cond=cond.expand(6, 8),
            is_register=torch.zeros(6, dtype=torch.bool) if prefix is None else prefix,
        )
        got = modulated_norm(norm, x, mod, prefix)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1].expand(6, 16), want[1])
