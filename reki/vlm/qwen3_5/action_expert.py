"""The action expert: a DiT after the VLM whose depth, heads and width are independent of the
VLM. Every block runs self-attention over the packed expert rows, cross-attention to the
final-layer VLM output, then a SwiGLU MLP.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm


def _init_expert_linears(module: nn.Module, std: float) -> None:
    """Qwen-style init on a module's Linear children: normal(0, std) weights (HF Qwen's
    ``initializer_range``), zero biases. Torch's default init is not Qwen's."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ActionExpertMLP(nn.Module):
    """The SwiGLU feed-forward sublayer of an action-expert block."""

    def __init__(self, expert_hidden: int, expert_intermediate: int, hidden_act: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(expert_hidden, expert_intermediate, bias=False)
        self.up_proj = nn.Linear(expert_hidden, expert_intermediate, bias=False)
        self.down_proj = nn.Linear(expert_intermediate, expert_hidden, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ActionExpertSelfAttention(nn.Module):
    """Self-attention projections of one expert block over the packed expert rows
    (registers ∥ actions). MHA at head_dim = hidden // heads."""

    def __init__(self, vlm_text_config: Any, expert_hidden: int, num_heads: int) -> None:
        super().__init__()
        assert expert_hidden % num_heads == 0, (expert_hidden, num_heads)
        self.head_dim = expert_hidden // num_heads
        self.n_q_heads = num_heads
        self.n_kv_heads = num_heads
        eps = vlm_text_config.rms_norm_eps

        self.q_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.k_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.v_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.o_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)


class ActionExpertCrossAttention(nn.Module):
    """Cross-attention projections of one expert block. Keys and values are projections of the
    final-layer VLM hidden states (vlm_hidden → expert_hidden)."""

    def __init__(self, vlm_text_config: Any, expert_hidden: int, num_heads: int) -> None:
        super().__init__()
        assert expert_hidden % num_heads == 0, (expert_hidden, num_heads)
        self.head_dim = expert_hidden // num_heads
        self.n_q_heads = num_heads
        self.n_kv_heads = num_heads
        eps = vlm_text_config.rms_norm_eps

        vlm_hidden = vlm_text_config.hidden_size
        self.q_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.k_proj = nn.Linear(vlm_hidden, expert_hidden, bias=False)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.v_proj = nn.Linear(vlm_hidden, expert_hidden, bias=False)
        self.o_proj = nn.Linear(expert_hidden, expert_hidden, bias=False)


class ActionExpertLayer(nn.Module):
    """Sequential pre-norm self-attention, cross-attention, and SwiGLU sublayers.

    Norms are plain RMSNorm at construction; `_replace_expert_norms` swaps them
    for τ-AdaRMS (input_layernorm for SA, cross_attn_layernorm for CA,
    post_attention_layernorm for the MLP)."""

    def __init__(
        self, vlm_text_config: Any, expert_hidden: int, expert_intermediate: int, num_heads: int
    ) -> None:
        super().__init__()
        eps = vlm_text_config.rms_norm_eps
        # Registration order (self_attn, cross_attn, mlp, norms) fixes the parameter order,
        # which optimizer-state resume relies on (it is positional), and the order of the init
        # RNG draws. Do not reorder.
        self.self_attn = ActionExpertSelfAttention(vlm_text_config, expert_hidden, num_heads)
        self.cross_attn = ActionExpertCrossAttention(vlm_text_config, expert_hidden, num_heads)
        self.mlp = ActionExpertMLP(expert_hidden, expert_intermediate, vlm_text_config.hidden_act)
        _init_expert_linears(self.self_attn, vlm_text_config.initializer_range)
        _init_expert_linears(self.cross_attn, vlm_text_config.initializer_range)
        _init_expert_linears(self.mlp, vlm_text_config.initializer_range)
        self.input_layernorm = Qwen3_5RMSNorm(expert_hidden, eps=eps)
        self.cross_attn_layernorm = Qwen3_5RMSNorm(expert_hidden, eps=eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(expert_hidden, eps=eps)


class Qwen3_5ActionExpert(nn.Module):
    """The expert stack: every block cross-attends the final-layer VLM output.
    ``registers`` is the learnable per-frame register bank prepended to each frame's noisy
    chunk (clean rows, no timestep conditioning)."""

    def __init__(
        self,
        vlm_text_config: Any,
        expert_hidden: int,
        expert_intermediate: int,
        num_layers: int,
        num_heads: int,
        num_registers: int,
    ) -> None:
        super().__init__()
        if num_registers <= 0:
            raise ValueError(
                f"num_registers must be > 0 (got {num_registers}); "
                "the expert always runs register memory"
            )
        self.registers = nn.Parameter(torch.empty(num_registers, expert_hidden))
        nn.init.trunc_normal_(self.registers, std=0.02)
        self.layers = nn.ModuleList(
            [
                ActionExpertLayer(vlm_text_config, expert_hidden, expert_intermediate, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.norm: nn.Module = Qwen3_5RMSNorm(expert_hidden, eps=vlm_text_config.rms_norm_eps)
