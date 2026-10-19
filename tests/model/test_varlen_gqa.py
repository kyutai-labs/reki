"""``varlen_attention`` with fewer KV heads than query heads (the Qwen3.5 text GQA layers) matches
the explicit ``repeat_interleave`` of K/V in the output and in dq/dk/dv. Guards torch's
undocumented GQA support in ``varlen_attn`` across torch upgrades."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

DEVICE = torch.device("cuda")


@pytest.mark.parametrize("is_causal", [True, False])
def test_varlen_gqa_matches_repeat(is_causal: bool) -> None:
    from torch.nn.attention.varlen import varlen_attn

    from reki.vlm.common import cu_seqlens_from, varlen_attention

    torch.manual_seed(0)
    lens, H, H_kv, D = [169, 700, 1, 33], 8, 2, 256
    T, rep = sum(lens), H // H_kv
    cu = cu_seqlens_from(lens, DEVICE)
    q, k, v = (
        torch.randn(T, h, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        for h in (H, H_kv, H_kv)
    )
    grad = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)

    out = varlen_attention(q, k, v, cu, cu, max(lens), max(lens), is_causal=is_causal)
    ref = varlen_attn(
        q,
        k.repeat_interleave(rep, dim=1),
        v.repeat_interleave(rep, dim=1),
        cu,
        cu,
        max(lens),
        max(lens),
        is_causal=is_causal,
    )
    assert isinstance(ref, torch.Tensor)
    grads = torch.autograd.grad(out, (q, k, v), grad)
    ref_grads = torch.autograd.grad(ref, (q, k, v), grad)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for name, g, r in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert g.shape == r.shape, name
        rel = ((g.float() - r.float()).norm() / r.float().norm()).item()
        assert rel < 1e-3, f"{name} rel err {rel:.2e}"
