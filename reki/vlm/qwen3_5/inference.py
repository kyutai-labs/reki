# _rope1 adapted from Transformers v5.3.0, src/transformers/models/qwen3_5/modeling_qwen3_5.py
# (apply_rotary_pos_emb), changed to rotate one tensor. Copyright 2025 The Qwen Team and The
# HuggingFace Inc. team. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). See NOTICE.
"""Inference of the action expert under register memory: the per-env-step conditioning
(``CrossAttnCache``) and the methods of ``Qwen3_5VLMWithExpert`` that run the Euler steps
(``Qwen3_5ExpertInference``, its base class). The VLM pass is ``Qwen3_5VLMWithExpert.forward``
with ``prefill_only=True``; the VLM keeps no KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, rotate_half

from ..adarms import gated_residual as _gated_residual
from ..adarms import modulated_norm
from ..adarms_base import PiGemmaRMSNorm as AdaRMSNorm
from ..common import cu_seqlens_from, cu_single

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PreTrainedModel

    from ..common import RegisterCache
    from .action_expert import Qwen3_5ActionExpert


# One Euler step's AdaRMS modulations: per expert layer (SA, CA, MLP norm), and the final norm.
ExpertModulation = tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]


class CrossAttnCache:
    """Per-env-step conditioning of the action expert at inference (no VLM K/V).

    The expert conditions only on the prefill's final-layer ``vlm_out`` (the cross-attention
    source) and packing metadata. The VLM pass is rerun every env step, so no recurrent state
    (K/V, gated delta net state) is carried. The prefill (``RekiFlowModel._prefill`` or its
    CUDA-graph replay) fills ``vlm_out``/``vlm_position_ids``;
    ``register_prepare`` builds the per-layer cross-attention K/V (``cross_attn_kv``), which
    does not depend on τ, so every Euler step reuses it.
    """

    def __init__(self) -> None:
        self.expert_start: int | None = None
        self.vlm_out: torch.Tensor | None = None
        self.vlm_position_ids: torch.Tensor | None = None
        # Fixed-size CA source and its true length / position start (``gather_cross_attn_source``).
        self.ca_src: torch.Tensor | None = None
        self.ca_src_pos: torch.Tensor | None = None
        self.ca_cu: torch.Tensor | None = None
        self.ca_start: torch.Tensor | None = None
        self.cross_attn_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        # Euler-step invariants (see ``register_prepare``).
        self.register_step: dict[str, Any] | None = None


def _rope1(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate a single [M, H, D] tensor with per-row cos/sin [M, rotary_dim]: the
    one-tensor body of ``apply_rotary_pos_emb`` (dims past ``rotary_dim`` pass through)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


def _cached_cross_attn(
    c_attn: Any,
    q_ca_hidden: torch.Tensor,
    cos_ca: torch.Tensor,
    sin_ca: torch.Tensor,
    k_ca: torch.Tensor,
    v_ca: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    head_dim: int,
    n_rows: int,
) -> torch.Tensor:
    """Single-sample CA to the precomputed source K/V (the cached analog of
    the CA sublayer in ``_action_expert_stack``). ``cu_q``/``cu_k`` are
    ``[0, n_rows]``/``[0, S]`` with ``S`` the true source length: the padding rows of
    ``k_ca``/``v_ca`` past ``S`` are never read (``gather_cross_attn_source``)."""
    q_ca = c_attn.q_norm(c_attn.q_proj(q_ca_hidden).view(n_rows, c_attn.n_q_heads, head_dim))
    q_ca = _rope1(q_ca, cos_ca, sin_ca)
    y_ca = flash_attn_varlen_func(
        q_ca,
        k_ca,
        v_ca,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=n_rows,
        max_seqlen_k=k_ca.shape[0],
        softmax_scale=head_dim**-0.5,
        causal=False,
    ).view(n_rows, -1)
    return c_attn.o_proj(y_ca)


class Qwen3_5ExpertInference:
    """The inference methods of ``Qwen3_5VLMWithExpert`` (its base class). Per env step:
    ``gather_cross_attn_source``, ``expert_modulations``, ``register_prepare``, then
    ``denoise_first_step_register`` and one ``denoise_forward_register`` per further Euler step,
    each with its row of the modulations (``split_modulation``)."""

    if TYPE_CHECKING:
        vlm: PreTrainedModel
        action_expert: Qwen3_5ActionExpert
        num_registers: int

        @property
        def text_backbone(self) -> nn.Module: ...

        def expert_layer(self, layer_idx: int) -> Any: ...

    def _expert_pe_rows(
        self, start: torch.Tensor, n: int, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-sample inference version of ``_expert_position_embeddings``: cos/sin
        ``[n, head_dim]`` for positions ``start .. start + n - 1`` on all three MRoPE axes.
        ``start`` is a one-element device tensor, so a compiled expert does not guard on it."""
        positions = start + torch.arange(n, device=ref.device, dtype=torch.long)
        mrope_positions = positions.view(1, -1).unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.text_backbone.rotary_emb(ref, mrope_positions)  # type: ignore[attr-defined]
        return cos.squeeze(0), sin.squeeze(0)

    def _register_sa_pe_rows(
        self, frame_idx: torch.Tensor, slot_idx: torch.Tensor, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MRoPE cos/sin for explicit (frame, slot) rows: the inference counterpart of
        ``_register_sa_position_embeddings`` (which derives (frame, slot) from a
        packed cs_per_sample). Frame on the temporal axis, slot on both spatial
        axes. Returns cos/sin ``[M, head_dim]``."""
        mrope = torch.stack([frame_idx, slot_idx, slot_idx], dim=0).unsqueeze(1).long()
        cos, sin = self.text_backbone.rotary_emb(ref, mrope)  # type: ignore[attr-defined]
        return cos.squeeze(0), sin.squeeze(0)

    def gather_cross_attn_source(self, past: CrossAttnCache, bucket: int) -> None:
        """Gather the CA source rows (the prefill ``vlm_out`` rows) and their positions into
        fixed-size tensors on ``past``, so the expert shapes do not depend on the prompt. Sets
        ``ca_src`` ``[rows, hidden]``, ``ca_src_pos`` ``[3, 1, rows]``, and as device data
        ``ca_cu`` = ``[0, S]`` and ``ca_start`` (the CA position start of the expert). ``rows``
        is the source length S rounded up to a multiple of ``bucket``. The padding repeats
        row 0 and is never attended. Runs eagerly, outside a compiled expert, once per env
        step."""
        vlm_out = past.vlm_out
        pos_ids = past.vlm_position_ids
        assert vlm_out is not None and pos_ids is not None, (
            "action expert cross-attention requires prefill vlm_out/position_ids "
            "stashed on the cache (set by RekiFlowModel._prefill)"
        )
        assert past.expert_start is not None
        vlm_flat = vlm_out.reshape(-1, vlm_out.shape[-1])
        S = vlm_flat.shape[0]
        idx = F.pad(torch.arange(S, device=vlm_flat.device), (0, -S % bucket))
        past.ca_src = vlm_flat[idx]
        past.ca_src_pos = pos_ids[:, idx].unsqueeze(1)
        past.ca_cu = cu_single(S, vlm_flat.device)
        past.ca_start = torch.full(
            (1,), past.expert_start, dtype=torch.long, device=vlm_flat.device
        )

    def _build_cross_attn_kv(self, past: CrossAttnCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build the CA K/V of each expert block over the padded CA source
        (``gather_cross_attn_source``), once per env step. The keys get the native MRoPE
        rotation at projection. The K/V do not depend on τ, so every Euler step reuses them."""
        src, src_pos = past.ca_src, past.ca_src_pos
        assert src is not None and src_pos is not None, "run gather_cross_attn_source first"
        s_cos, s_sin = self.text_backbone.rotary_emb(src, src_pos)  # type: ignore[attr-defined]
        s_cos, s_sin = s_cos.squeeze(0), s_sin.squeeze(0)
        S = src.shape[0]
        src_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(self.action_expert.layers)):  # type: ignore[arg-type]
            c_attn = self.expert_layer(layer_idx).cross_attn
            D = c_attn.head_dim
            k_ca = c_attn.k_norm(c_attn.k_proj(src).view(S, c_attn.n_kv_heads, D))
            k_ca = _rope1(k_ca, s_cos, s_sin)
            v_ca = c_attn.v_proj(src).view(S, c_attn.n_kv_heads, D)
            src_kv.append((k_ca, v_ca))
        past.cross_attn_kv = src_kv
        return src_kv

    def register_prepare(
        self,
        past: CrossAttnCache,
        reg_cache: RegisterCache,
        n_act: int,
    ) -> None:
        """Per-env-step expert constants under register memory (inference). Builds the CA
        source K/V and stores on ``past.register_step`` what the Euler steps reuse: cos/sin and
        cu_seqlens for the first step over ``[R registers ∥ n_act actions]``
        (``denoise_first_step_register``) and for the action-only steps 2..N, and the cos/sin
        that rotate the cached pre-RoPE memory keys to their window-local frames.

        The cache capacity fixes every shape. The fill level ``n = reg_cache.n_valid`` is device
        data (positions and cu_seqlens), so one graph serves every fill level. The buffer has
        ``P`` rows and is right-aligned: rows ``[P - nR, P)`` are frames ``0 .. n-1``, and the
        current frame is ``n``. In the keys, the stale buffer rows are in front of each memory
        copy. cu_seqlens_k starts each segment at its first valid row. Flash varlen reads the
        keys of segment ``i`` from ``[cu_k[i], cu_k[i + 1])`` only, so ``cu_k[0]`` can be > 0.
        """
        past.cross_attn_kv = None
        past.register_step = None
        self._build_cross_attn_kv(past)
        R = self.num_registers
        reg_param = self.action_expert.registers
        ref = reg_param.to(dtype=self.vlm.dtype)  # cos/sin read only its dtype/device
        device = ref.device
        n = reg_cache.n_valid.to(device=device, dtype=torch.long)
        P = reg_cache.k.shape[1]
        n_rows = R + n_act
        # The current frame is at window-local index n, before and after append_frame.
        sa_cos, sa_sin = self._register_sa_pe_rows(
            n.expand(n_rows), torch.arange(n_rows, device=device), ref
        )
        assert past.ca_start is not None and past.ca_cu is not None
        ca_cos, ca_sin = self._expert_pe_rows(past.ca_start, n_rows, ref)
        # Buffer row r holds frame r // R - (capacity - n); stale rows (< 0) are never read.
        rows = torch.arange(P, device=device)
        prev_pe = self._register_sa_pe_rows(
            (rows // R - (reg_cache.capacity - n)).clamp_min(0), rows % R, ref
        )
        # Step 1 keys [buffer ∥ R ∥ buffer ∥ R ∥ n_act]: register queries attend
        # [P - nR, P + R), an empty query segment skips the stale rows of the second copy,
        # action queries attend [2P + R - nR, 2P + 2R + n_act).
        n_mem_prev = (n * R).to(torch.int32)
        even = torch.arange(4, device=device) % 2 == 0
        cu_first_kv = cu_seqlens_from([P, R, P, R + n_act], device)[1:] - even * n_mem_prev
        past.register_step = {
            "first_sa_pe": (sa_cos, sa_sin),
            "first_ca_pe": (ca_cos, ca_sin),
            "prev_pe": prev_pe,
            "sa_pe": (sa_cos[R:], sa_sin[R:]),
            "ca_pe": (ca_cos[R:], ca_sin[R:]),
            "cu_first_q": cu_seqlens_from([R, 0, n_act], device),
            "cu_first_kv": cu_first_kv,
            "cu_rows": cu_single(n_rows, device),
            "cu_act": cu_single(n_act, device),
            # Steps 2..N keys [buffer ∥ R ∥ n_act]: [P - nR, P + R + n_act).
            "cu_kv": cu_first_kv[2:] - (P + R),
            "cu_src": past.ca_cu,
            "mem_kv": None,  # set from the first step's output
        }

    def _modulation_norms(self) -> list[Any]:
        """The AdaRMS expert norms in modulation-row order: (SA, CA, MLP) per layer, then the
        final norm."""
        norms = [
            norm
            for layer in self.action_expert.layers
            for norm in (
                layer.input_layernorm,
                layer.cross_attn_layernorm,
                layer.post_attention_layernorm,
            )
        ]
        return [*norms, self.action_expert.norm]

    def expert_modulations(self, cond: torch.Tensor) -> torch.Tensor:
        """AdaRMS modulations ``norm.dense(cond)`` of every expert norm for all Euler steps at
        once. ``cond`` is ``[n_steps, cond_dim]``: one row per step, shared by every action row.
        Returns ``[n_steps, sum of 3 * hidden]``, the norms concatenated in ``_modulation_norms``
        order. ``split_modulation`` splits the ``[1, ...]`` row of one step."""
        return torch.cat([norm.dense(cond) for norm in self._modulation_norms()], dim=-1)

    def split_modulation(self, row: torch.Tensor) -> ExpertModulation:
        """One step's ``[1, ...]`` row of ``expert_modulations`` as per-layer (SA, CA, MLP)
        and final modulations, each ``[1, 3 * hidden]`` (views)."""
        parts = row.split([norm.dense.out_features for norm in self._modulation_norms()], -1)
        per_layer = [tuple(parts[i : i + 3]) for i in range(0, len(parts) - 1, 3)]
        return per_layer, parts[-1]  # type: ignore[return-value]

    def denoise_first_step_register(
        self,
        expert_hidden: torch.Tensor,
        past: CrossAttnCache,
        mods: ExpertModulation,
        prev_k: torch.Tensor,
        prev_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """First Euler step under register memory. The R register rows of the current frame and
        the action rows go through the expert as one sequence, as in training
        (``_action_expert_stack``). Register rows use the plain norms (gate 1) and SA over
        ``[memory ∥ current registers]``. Action rows use the AdaRMS cond and SA over
        ``[memory ∥ current registers ∥ own actions]``.

        ``mods`` is the AdaRMS modulation of this step (``split_modulation``).
        ``prev_k``/``prev_v`` are the fixed-size pre-RoPE k/v buffers of the register cache
        ``[n_layers, P, H, D]``; the cu_seqlens of ``register_prepare`` select the valid rows.
        Pure. Returns the action output, the pre-RoPE register k and v of this frame
        ``[n_layers, R, H, D]`` (for ``RegisterCache.append_frame``) and the rotated memory K/V
        per layer, for steps 2..N.
        """
        src_kv, step = past.cross_attn_kv, past.register_step
        assert src_kv is not None and step is not None
        R = self.num_registers
        n_act = expert_hidden.shape[0]
        n_rows = R + n_act
        reg_param = self.action_expert.registers
        hidden = torch.cat([reg_param.to(expert_hidden.dtype), expert_hidden], dim=0)
        # Register rows take the plain norm and gate 1 (is_register), whatever the modulation.
        is_register = torch.arange(n_rows, device=hidden.device) < R
        cos, sin = step["first_sa_pe"]
        cos_ca, sin_ca = step["first_ca_pe"]
        cu_q, cu_kv, cu_rows, cu_src = (
            step["cu_first_q"],
            step["cu_first_kv"],
            step["cu_rows"],
            step["cu_src"],
        )
        # All layers at once: the cached keys move to their window-local frames.
        prev_rot = _rope1(prev_k, *step["prev_pe"])
        n_mem = prev_k.shape[1] + R

        cur_k: list[torch.Tensor] = []
        cur_v: list[torch.Tensor] = []
        mem_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(self.action_expert.layers)):  # type: ignore[arg-type]
            e_layer = self.expert_layer(layer_idx)
            e_attn = e_layer.self_attn
            D = e_attn.head_dim
            mod_sa, mod_ca, mod_mlp = mods[0][layer_idx]
            e_norm, e_gate = modulated_norm(e_layer.input_layernorm, hidden, mod_sa, is_register)
            q = e_attn.q_norm(e_attn.q_proj(e_norm).view(n_rows, e_attn.n_q_heads, D))
            k = e_attn.k_norm(e_attn.k_proj(e_norm).view(n_rows, e_attn.n_kv_heads, D))
            v = e_attn.v_proj(e_norm).view(n_rows, e_attn.n_kv_heads, D)
            cur_k.append(k[:R])
            cur_v.append(v[:R])
            q, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            mem_k = torch.cat([prev_rot[layer_idx], k_rot[:R]], dim=0)
            mem_v = torch.cat([prev_v[layer_idx], v[:R]], dim=0)
            mem_kv.append((mem_k, mem_v))
            # One varlen call: register queries attend the memory, action queries attend
            # memory ∥ own actions (an empty query segment in between skips stale rows).
            y_sa = flash_attn_varlen_func(
                q,
                torch.cat([mem_k, mem_k, k_rot[R:]], dim=0),
                torch.cat([mem_v, mem_v, v[R:]], dim=0),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_kv,
                max_seqlen_q=max(R, n_act),
                max_seqlen_k=n_mem + n_act,
                softmax_scale=D**-0.5,
                causal=False,
            ).view(n_rows, -1)
            h = _gated_residual(hidden, e_attn.o_proj(y_sa), e_gate)

            k_ca, v_ca = src_kv[layer_idx]
            ca_norm, ca_gate = modulated_norm(e_layer.cross_attn_layernorm, h, mod_ca, is_register)
            out_ca = _cached_cross_attn(
                e_layer.cross_attn, ca_norm, cos_ca, sin_ca, k_ca, v_ca, cu_rows, cu_src, D, n_rows
            )
            h = _gated_residual(h, out_ca, ca_gate)

            h_norm2, gate2 = modulated_norm(
                e_layer.post_attention_layernorm, h, mod_mlp, is_register
            )
            hidden = _gated_residual(h, e_layer.mlp(h_norm2), gate2)
        final_norm = self.action_expert.norm
        assert isinstance(final_norm, AdaRMSNorm)
        out, _ = modulated_norm(final_norm, hidden[R:], mods[1])
        return out, torch.stack(cur_k), torch.stack(cur_v), mem_kv

    def denoise_forward_register(
        self, expert_hidden: torch.Tensor, past: CrossAttnCache, mods: ExpertModulation
    ) -> torch.Tensor:
        """Inference denoise steps 2..N under register memory: action rows SA over
        ``[memory registers ∥ own actions]``, sequential CA to the cached final-layer
        source, SwiGLU. Everything τ-independent comes from ``register_prepare`` (source
        K/V, cos/sin, cu_seqlens), ``split_modulation`` (``mods``: this step's AdaRMS
        modulation) and the first step (rotated memory K/V)."""
        src_kv, step = past.cross_attn_kv, past.register_step
        assert src_kv is not None and step is not None, (
            "register denoise requires the source K/V and step constants built by register_prepare"
        )
        cs = expert_hidden.shape[0]
        a_cos, a_sin = step["sa_pe"]
        cos_ca, sin_ca = step["ca_pe"]
        # The cos/sin were built against the register rows; rotary_emb only reads x's dtype.
        assert a_cos.shape[0] == cs and a_cos.dtype == expert_hidden.dtype
        cu_a, cu_kv, cu_src = step["cu_act"], step["cu_kv"], step["cu_src"]

        action_hidden = expert_hidden
        for layer_idx in range(len(self.action_expert.layers)):  # type: ignore[arg-type]
            e_layer = self.expert_layer(layer_idx)
            e_attn = e_layer.self_attn
            D = e_attn.head_dim
            mod_sa, mod_ca, mod_mlp = mods[0][layer_idx]
            a_norm, a_gate = modulated_norm(e_layer.input_layernorm, action_hidden, mod_sa)
            qa = e_attn.q_norm(e_attn.q_proj(a_norm).view(cs, e_attn.n_q_heads, D))
            ka = e_attn.k_norm(e_attn.k_proj(a_norm).view(cs, e_attn.n_kv_heads, D))
            va = e_attn.v_proj(a_norm).view(cs, e_attn.n_kv_heads, D)
            qa, ka = apply_rotary_pos_emb(qa, ka, a_cos, a_sin, unsqueeze_dim=1)

            # One softmax over the union of keys, as in training.
            mem_k, mem_v = step["mem_kv"][layer_idx]
            y_sa = (
                flash_attn_varlen_func(
                    qa,
                    torch.cat([mem_k, ka], dim=0),
                    torch.cat([mem_v, va], dim=0),
                    cu_seqlens_q=cu_a,
                    cu_seqlens_k=cu_kv,
                    max_seqlen_q=cs,
                    max_seqlen_k=mem_k.shape[0] + cs,
                    softmax_scale=D**-0.5,
                    causal=False,
                )
                .to(action_hidden.dtype)
                .view(cs, -1)
            )
            h = _gated_residual(action_hidden, e_attn.o_proj(y_sa), a_gate)

            k_ca, v_ca = src_kv[layer_idx]
            ca_norm, ca_gate = modulated_norm(e_layer.cross_attn_layernorm, h, mod_ca)
            out_ca = _cached_cross_attn(
                e_layer.cross_attn, ca_norm, cos_ca, sin_ca, k_ca, v_ca, cu_a, cu_src, D, cs
            )
            h = _gated_residual(h, out_ca, ca_gate)

            h_norm2, gate2 = modulated_norm(e_layer.post_attention_layernorm, h, mod_mlp)
            action_hidden = _gated_residual(h, e_layer.mlp(h_norm2), gate2)
        final_norm = self.action_expert.norm
        assert isinstance(final_norm, AdaRMSNorm)
        return modulated_norm(final_norm, action_hidden, mods[1])[0]
