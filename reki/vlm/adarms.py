"""τ-AdaRMS norms of the action expert.

Only torch, so it imports on a CPU-only host (CPU dry run, most of the tests).
The ``qwen3_5`` model modules (``modeling``, ``inference``, ``hf_patches``) import the CUDA kernels
at module level and do not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from reki.vlm.adarms_base import PiGemmaRMSNorm as AdaRMSNorm
from reki.vlm.adarms_base import _gated_residual

gated_residual = cast(
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor], _gated_residual
)


class RegisterAdaRMSNorm(AdaRMSNorm):
    """AdaRMS for action rows and plain RMSNorm without cond (gate 1) for the register rows
    (``is_register``)."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None) -> None:
        super().__init__(dim, eps=eps, cond_dim=cond_dim)
        self.register_weight = nn.Parameter(torch.zeros(dim))

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        is_register: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed, gate = super().forward(x, cond=cond)
        plain = (self._norm(x) * (1.0 + self.register_weight.float())).type_as(x)
        sel = is_register.unsqueeze(-1)
        normed = torch.where(sel, plain, normed)
        if gate is not None:
            gate = torch.where(sel, torch.ones_like(gate), gate)
        return normed, gate


def modulated_norm(
    norm: AdaRMSNorm,
    x: torch.Tensor,
    modulation: torch.Tensor,
    is_register: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``RegisterAdaRMSNorm.forward`` with ``modulation = norm.dense(cond)`` computed by the
    caller, which may broadcast over the rows of ``x``. The modulation depends only on the cond,
    so inference computes it for all Euler steps at once. ``is_register=None``: no register
    rows (inference after the first Euler step)."""
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normed = (norm._norm(x) * (1 + scale.float()) + shift.float()).to(x.dtype)
    gate = gate.to(x.dtype)
    if is_register is None:
        return normed, gate
    plain = (norm._norm(x) * (1.0 + norm.register_weight.float())).type_as(x)  # type: ignore[attr-defined]
    sel = is_register.unsqueeze(-1)
    return torch.where(sel, plain, normed), torch.where(sel, torch.ones_like(gate), gate)
