# Copied verbatim from LeRobot (github.com/huggingface/lerobot @ aa9cc9bd43e9eba92a32aaadde2a09c90b5836cf),
# src/lerobot/policies/pi_gemma.py (_gated_residual, PiGemmaRMSNorm). Kept byte-identical:
# checkpoints and bit-exact parity depend on this math.
#
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pyright: reportMissingParameterType=false
"""AdaRMS norm (scale/shift/gate from a conditioning vector) and its gated residual."""

from __future__ import annotations

import torch
from torch import nn


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """Gated residual: x + y when gate is None, else x + y * gate."""
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


class PiGemmaRMSNorm(nn.Module):
    """
    Adaptive RMSNorm for PI Gemma (AdaRMS).
    When cond_dim is set, uses cond to modulate scale/shift/gate; otherwise behaves like standard GemmaRMSNorm.
    forward(x, cond=None) returns (output, gate) for use with _gated_residual.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is not None:
            return f"dim={self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return f"dim={self.dim}, eps={self.eps}"
