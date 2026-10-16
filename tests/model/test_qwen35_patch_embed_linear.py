"""Parity: the folded Linear patch embed on single-frame (t=1) patches equals the native
Conv3d on the frame duplicated across t (``W.sum(dim=2)`` folds the temporal kernels)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.gpu


class _FakeConvPatchEmbed(nn.Module):
    """Mirror of transformers' ``Qwen3_5VisionPatchEmbed``: a Conv3d with
    kernel == stride == ``[t, p, p]`` and the ``view(-1, C, t, p, p)`` forward."""

    def __init__(self, in_channels: int, t: int, p: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels, self.t, self.p, self.embed_dim = in_channels, t, p, embed_dim
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=(t, p, p), stride=(t, p, p), bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(-1, self.in_channels, self.t, self.p, self.p)
        return self.proj(x).view(-1, self.embed_dim)


def test_folded_linear_matches_conv3d_on_duplicated_input() -> None:
    """folded-Linear(single-frame C*p*p patch) == Conv3d(frame duplicated across t)."""
    from reki.vlm.qwen3_5.hf_patches import LinearPatchEmbed

    torch.manual_seed(0)
    C, t, p, embed_dim, n_patches = 3, 2, 16, 128, 7
    conv_pe = _FakeConvPatchEmbed(C, t, p, embed_dim).to(torch.float64)
    lin_pe = LinearPatchEmbed.from_conv3d(conv_pe).to(torch.float64)

    assert lin_pe.in_features == C * p * p

    single = torch.randn(n_patches, C * p * p, dtype=torch.float64)
    dup = (
        single.view(n_patches, C, 1, p, p)
        .expand(n_patches, C, t, p, p)
        .reshape(n_patches, C * t * p * p)
    )

    out_conv = conv_pe(dup)
    out_lin = lin_pe(single)

    assert out_conv.shape == out_lin.shape == (n_patches, embed_dim)
    max_err = (out_conv - out_lin).abs().max().item()
    assert torch.allclose(out_conv, out_lin, atol=1e-9, rtol=1e-7), f"max abs err {max_err}"
