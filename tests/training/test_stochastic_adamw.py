"""StochasticRoundingAdamW under ``optimizer_master_weights=True`` (every shipped recipe): the
fp32 master trajectory equals fp32 AdamW, and a resume keeps the fp32 moments."""

from __future__ import annotations

import pytest
import torch

from reki.policy.optim import StochasticRoundingAdamW

pytestmark = pytest.mark.cpu


def test_load_state_dict_preserves_fp32_moments():
    """Base Optimizer.load_state_dict casts state to the param's dtype, silently
    demoting fp32 moments to bf16 on plain-DDP resume. The override re-injects fp32."""
    p = torch.nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
    opt = StochasticRoundingAdamW([p], lr=1e-3, weight_decay=0.01)
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = StochasticRoundingAdamW([p2], lr=1e-3, weight_decay=0.01)
    opt2.load_state_dict(sd)

    for key in ("exp_avg", "exp_avg_sq"):
        restored = opt2.state[p2][key]
        assert restored.dtype == torch.float32
        assert torch.equal(restored, opt.state[p][key])
        assert restored.data_ptr() != sd["state"][0][key].data_ptr()


def test_master_weights_matches_fp32_adamw():
    """master_weights=True reproduces torch.optim.AdamW: the fp32 master accumulates exactly, so
    bf16 storage costs only the write-back rounding (a few ulp)."""
    torch.manual_seed(0)
    w0 = torch.randn(4096) * 1e-2
    grads = [torch.randn(4096) * 1e-3 for _ in range(100)]
    kw = dict(lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-10)

    def run(dtype: torch.dtype, **extra: object) -> torch.Tensor:
        p = torch.nn.Parameter(w0.clone().to(dtype))
        opt = (
            torch.optim.AdamW([p], **kw)  # type: ignore[arg-type]
            if extra.pop("stock", False)
            else StochasticRoundingAdamW([p], **kw, **extra)  # type: ignore[arg-type]
        )
        for g in grads:
            p.grad = g.clone().to(dtype)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return p.detach().float()

    ref = run(torch.float32, stock=True)
    master = run(torch.bfloat16, master_weights=True)
    sr = run(torch.bfloat16, master_weights=False)

    ulp = (w0.abs() * 2**-8).mean()
    master_err, sr_err = (master - ref).abs().mean(), (sr - ref).abs().mean()
    assert master_err < 0.25 * sr_err, f"master no better than SR ({master_err} vs {sr_err})"
    assert master_err < ulp, f"master error {master_err} exceeds the bf16 write-back floor {ulp}"
