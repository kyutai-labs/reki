"""Expert register-memory self-attention matches a dense fp32 masked softmax in the output AND
in dq/dk/dv (the flash kernel's backward drops the LSE gradient, so a two-call LSE merge is exact
only in the forward)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import torch

pytestmark = pytest.mark.gpu

DEVICE = torch.device("cuda")


def test_register_sa_output_and_grads_match_dense() -> None:
    from reki.policy.modeling import build_register_memory_indices
    from reki.vlm.qwen3_5.modeling import ExpertForwardState, Qwen3_5VLMWithExpert

    R, draws, H, D = 4, 3, 4, 64
    # groups of 3, 1 and 2 frames; chunk size differs across groups
    fp = torch.tensor([0, 1, 2, 0, 0, 1], device=DEVICE)
    gi = torch.tensor([0, 0, 0, 1, 2, 2], device=DEVICE)
    cs = torch.tensor([5, 5, 5, 7, 3, 3], device=DEVICE)
    cs_e = R + cs * draws
    reg_idx, reg_counts = build_register_memory_indices(cs_e, fp, gi, R, 1)
    N = int(cs_e.sum())

    owner = torch.arange(cs_e.shape[0], device=DEVICE).repeat_interleave(cs_e)
    starts = torch.cat([cs_e.new_zeros(1), cs_e.cumsum(0)[:-1]])
    off = torch.arange(N, device=DEVICE) - starts[owner]
    is_reg = off < R
    draw = torch.where(is_reg, -1, (off - R) // cs[owner])
    qo, ko = owner[:, None], owner[None, :]
    allowed = (gi[qo] == gi[ko]) & (
        (is_reg[None, :] & (fp[ko] <= fp[qo]))
        | (~is_reg[None, :] & (qo == ko) & (draw[:, None] == draw[None, :]))
    )

    fw = ExpertForwardState(
        reg_kv_flat_idx=reg_idx,
        reg_kv_counts=reg_counts,
        expert_is_register=is_reg,
        cs_per_sample=cs_e,
        flow_draws=draws,
    )
    obj = SimpleNamespace(fwd=fw, num_registers=R)
    e_attn = SimpleNamespace(o_proj=lambda x: x)

    def model_sa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return Qwen3_5VLMWithExpert.register_memory_attention(
            cast(Qwen3_5VLMWithExpert, obj),
            e_attn,
            q,
            k,
            v,
            D,
            N,
        )

    def dense_sa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        s = torch.einsum("qhd,khd->hqk", q, k) * D**-0.5
        p = s.masked_fill(~allowed, float("-inf")).softmax(-1)
        return torch.einsum("hqk,khd->qhd", p, v).reshape(N, -1)

    torch.manual_seed(0)
    qkv = [torch.randn(N, H, D, device=DEVICE).bfloat16() for _ in range(3)]
    dy = torch.randn(N, H * D, device=DEVICE)

    def run(fn: Callable[..., torch.Tensor], dtype: torch.dtype) -> list[torch.Tensor]:
        ts = [t.detach().to(dtype).requires_grad_() for t in qkv]
        y = fn(*ts)
        y.float().backward(dy)
        out = [y.detach().float()]
        for t in ts:
            assert t.grad is not None
            out.append(t.grad.float())
        return out

    got = run(model_sa, torch.bfloat16)
    ref = run(dense_sa, torch.float32)
    for name, a, b in zip(("y", "dq", "dk", "dv"), got, ref):
        rel = float((a - b).norm() / b.norm())
        assert rel < 0.02, f"{name}: rel L2 error {rel:.4f} vs dense fp32"
