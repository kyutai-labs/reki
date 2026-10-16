# Parts adapted from Transformers v5.3.0, src/transformers/models/qwen3_5/modeling_qwen3_5.py
# (_packed_gdn_forward, _PackedQwen3_5DecoderLayer.forward, the attention forward of
# install_vision_varlen), changed for sequence packing. Copyright 2025 The Qwen Team and The
# HuggingFace Inc. team. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). See NOTICE.
"""Patches to the HF Qwen3.5 model and the FLA kernels, applied by ``Qwen3_5VLMWithExpert``:
a ``Linear`` vision patch embed, one flash-attention varlen call per vision layer, decoder
layers that pass the packing kwargs to the gated delta net, and the fixed FLA kernel configs of
the inference prefill (``fla_inference_configs``)."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import types
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    apply_rotary_pos_emb_vision,
)

# FLA kernels of the GDN prefill whose config changes the output bits:
# (module, kernel) -> (meta-parameters, num_warps, num_stages).
# On the first call in a process, Triton times each config and keeps the fastest. For these
# kernels, several configs are equally fast but add in a different order. Without a fixed
# config, the actions differ between processes (up to ~1e-2). Each fixed config is the one
# that autotuning chose most often on H100. The other prefill kernels give the same bits
# with every config.
_FLA_INFERENCE_CONFIGS = {
    ("fla.modules.fused_norm_gate", "layer_norm_gated_fwd_kernel"): ({"BT": 16}, 16, 3),
    ("fla.modules.l2norm", "l2norm_fwd_kernel"): ({"BT": 8}, 4, 3),
    ("fla.ops.utils.solve_tril", "merge_16x16_to_64x64_inverse_kernel"): (
        {"DOT_PRECISION": "ieee"},
        4,
        3,
    ),
}
_fla_inference_tuners: list[tuple[Any, list[Any]]] | None = None  # (autotuner, [pinned config])


@contextlib.contextmanager
def fla_inference_configs() -> Iterator[None]:
    """Run each kernel in ``_FLA_INFERENCE_CONFIGS`` with its fixed config. The inference
    prefill then gives the same bits in every process. On exit, the full config lists come back,
    so training continues to autotune after a predict call."""
    global _fla_inference_tuners
    if _fla_inference_tuners is None:
        from triton.runtime.autotuner import Autotuner

        tuners = []
        for (module, name), (kwargs, num_warps, num_stages) in _FLA_INFERENCE_CONFIGS.items():
            tuner = getattr(importlib.import_module(module), name)
            while not isinstance(tuner, Autotuner):  # under @triton.heuristics
                tuner = tuner.fn
            pinned = [
                c
                for c in tuner.configs
                if c.kwargs == kwargs and c.num_warps == num_warps and c.num_stages == num_stages
            ]
            assert len(pinned) == 1, (
                f"{module}.{name}: no config {kwargs}, {num_warps}, {num_stages}"
            )
            tuners.append((tuner, pinned))
        _fla_inference_tuners = tuners
    saved = [(tuner, tuner.configs) for tuner, _ in _fla_inference_tuners]
    for tuner, pinned in _fla_inference_tuners:
        tuner.configs = pinned  # with one config, the autotuner does not time anything
    try:
        yield
    finally:
        for tuner, configs in saved:
            tuner.configs = configs


class LinearPatchEmbed(nn.Module):
    """Qwen3.5 vision patch embed as a ``Linear``, with the temporal axis folded.

    The native ``Qwen3_5VisionPatchEmbed`` is a ``Conv3d`` whose kernel and stride are the full
    ``[t, p, p]`` patch, so it is a linear projection of each flattened patch. As a ``Linear``,
    it runs as a GEMM. HF's image processor repeats each image over ``t=2`` identical temporal
    slices. The kernel summed over ``t`` is the exact single-frame projection over ``C*p*p``
    (see :meth:`from_conv3d`). ``FastQwenImageProcessor`` emits ``t=1`` patches, half the
    ``pixel_values``.

    :meth:`Qwen3_5VLMWithExpert.from_pretrained` installs it after loading
    (see :func:`linearize_vision_patch_embed`).
    """

    def __init__(self, in_features: int, embed_dim: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.proj = nn.Linear(in_features, embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.proj.weight
        assert hidden_states.shape[-1] == self.in_features, (
            f"patch-embed expected last dim {self.in_features} (C*p*p, temporally "
            f"folded), got {hidden_states.shape[-1]}; the image processor must emit "
            f"single-frame patches (temporal_patch_size=1)."
        )
        return self.proj(hidden_states.to(w.dtype))

    @classmethod
    def from_conv3d(cls, patch_embed: nn.Module) -> "LinearPatchEmbed":
        """Build from a native patch embed, folding the Conv3d weights.

        The conv weight is ``[O, C, t, p, p]``. HF's image processor repeats a
        single frame across the ``t`` temporal slices, so for an input row ``x``
        with ``x[c, 0, i, j] == x[c, 1, i, j] == img[c, i, j]`` the conv output is

            out[o] = Σ_{c,τ,i,j} W[o, c, τ, i, j] · x[c, τ, i, j]
                   = Σ_{c,i,j} (Σ_τ W[o, c, τ, i, j]) · img[c, i, j]

        This is a linear projection over ``C*p*p`` with the kernel ``W.sum(dim=2)``. That
        kernel and the single-frame patch from ``FastQwenImageProcessor`` both flatten
        row-major in ``(C, p, p)`` order, so the projection is exact up to fp accumulation
        order.
        """
        conv = patch_embed.proj
        assert isinstance(conv, nn.Conv3d)
        embed_dim = conv.weight.shape[0]
        folded = conv.weight.sum(dim=2)
        in_features = int(folded.numel() // embed_dim)
        module = cls(in_features, embed_dim).to(conv.weight.device, conv.weight.dtype)
        with torch.no_grad():
            module.proj.weight.copy_(folded.reshape(embed_dim, in_features))
            if conv.bias is not None:
                module.proj.bias.copy_(conv.bias)
        return module


def linearize_vision_patch_embed(vlm: Any) -> None:
    """Swap the vision ``Conv3d`` patch embed for the equivalent folded ``Linear``.

    A no-op if it is already a :class:`LinearPatchEmbed`. A REKI checkpoint stores the
    folded weights under the same ``visual.patch_embed.proj.*`` keys, so it loads on top of
    the swapped module. Raises if there is no ``Conv3d`` patch embed.
    """
    visual = vlm.model.visual
    pe = visual.patch_embed
    if isinstance(pe, LinearPatchEmbed):
        return
    if not isinstance(pe.proj, nn.Conv3d):
        raise TypeError(f"expected a Conv3d vision patch embed, got {type(pe.proj).__name__}")
    visual.patch_embed = LinearPatchEmbed.from_conv3d(pe)


def install_vision_varlen(vlm: nn.Module) -> None:
    """Patch the HF vision tower to launch fewer kernels and host ops. The math is the same.

    - Attention: one ``flash_attn_varlen_func`` call per layer replaces HF's per-image SDPA
      loop. HF's FA2 path crashes on the host ``cu_seqlens`` that the packed path passes.
      The device copy of ``cu_seqlens`` is made once per tower call.
    - Position tables: HF's ``rot_pos_emb`` / ``fast_pos_embed_interpolate`` are per-image
      Python loops. The patch computes them once per distinct grid and tiles the result. The
      rows of each image depend only on its own grid.
    """
    visual = getattr(getattr(vlm, "model", vlm), "visual", None)
    if visual is None:
        raise RuntimeError("install_vision_varlen: vision tower not found")
    # The patched forward re-implements HF's: refuse any layout it does not cover.
    expected_sig = [
        "self",
        "hidden_states",
        "cu_seqlens",
        "rotary_pos_emb",
        "position_embeddings",
        "kwargs",
    ]
    for blk in visual.blocks:
        attn = blk.attn
        sig = list(inspect.signature(type(attn).forward).parameters)
        children = sorted(n for n, _ in attn.named_children())
        if sig != expected_sig or children != ["proj", "qkv"]:
            raise RuntimeError(
                f"install_vision_varlen: unexpected {type(attn).__name__} layout "
                f"(forward params {sig}, children {children})"
            )
        if attn.attention_dropout != 0.0 or attn.is_causal:
            raise RuntimeError(
                f"install_vision_varlen: attention_dropout={attn.attention_dropout}, "
                f"is_causal={attn.is_causal}; the varlen path supports neither"
            )
    cu_cache: dict[str, Any] = {"host": None}

    def attn_forward(
        self: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert position_embeddings is not None
        S = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(S, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        if cu_cache["host"] is not cu_seqlens:
            lens = cu_seqlens[1:] - cu_seqlens[:-1]
            dev = cu_seqlens.to(torch.int32)
            if dev.device.type == "cpu":
                dev = dev.pin_memory().to(q.device, non_blocking=True)
            cu_cache.update(host=cu_seqlens, dev=dev, max=int(lens.max()))
        out = flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q=cu_cache["dev"],
            cu_seqlens_k=cu_cache["dev"],
            max_seqlen_q=cu_cache["max"],
            max_seqlen_k=cu_cache["max"],
            softmax_scale=self.scaling,
            causal=False,
        )
        return self.proj(out.reshape(S, -1))

    for blk in visual.blocks:
        blk.attn.forward = types.MethodType(attn_forward, blk.attn)

    # Bound via MethodType (not closures over bound methods) so a deepcopy rebinds to the copy.
    def per_grid(fn: Any) -> Any:
        def tiled(self: Any, grid_thw: torch.Tensor) -> torch.Tensor:
            rows = [tuple(r) for r in grid_thw.tolist()]
            uniq = {
                r: fn(self, torch.tensor([r], dtype=grid_thw.dtype)) for r in dict.fromkeys(rows)
            }
            return torch.cat([uniq[r] for r in rows])

        return tiled

    for name in ("rot_pos_emb", "fast_pos_embed_interpolate"):
        setattr(visual, name, types.MethodType(per_grid(getattr(type(visual), name)), visual))


def _packed_gdn_forward(
    attn: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_idx: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """HF's ``Qwen3_5GatedDeltaNet.forward`` (no-cache branch) with packing added. ``seq_idx``
    segments the causal conv and ``cu_seqlens`` (int64) segments the chunked delta rule, so
    token mixing never crosses a sample boundary. The FLA and causal-conv1d kernels are HF's.
    ``cu_seqlens_cpu`` is an optional host copy. With it, FLA indexes the chunks without a host
    sync, which CUDA-graph capture requires.
    """
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = attn.in_proj_qkv(hidden_states).transpose(1, 2)
    z = attn.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, attn.head_v_dim)
    b = attn.in_proj_b(hidden_states)
    a = attn.in_proj_a(hidden_states)

    mixed_qkv = causal_conv1d_fn(
        x=mixed_qkv,
        weight=attn.conv1d.weight.squeeze(1),
        bias=attn.conv1d.bias,
        activation=attn.activation,
        seq_idx=seq_idx,
    ).transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, attn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, attn.head_v_dim)

    beta = b.sigmoid()
    # Without the .float(), A might be -inf in low precision (as in HF).
    g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
    rep = attn.num_v_heads // attn.num_k_heads
    if rep > 1:
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    core_attn_out, _ = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )

    core_attn_out = attn.norm(
        core_attn_out.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim)
    )
    return attn.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


class _PackedQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    """HF decoder layer whose ``linear_attn`` also receives the packing kwargs.

    HF's layer passes ``**kwargs`` (including the flash-attention varlen kwargs) only to
    ``self_attn``; this subclass also passes ``gdn_cu_seqlens``/``gdn_seq_idx`` to the gated
    delta net. Installed by a ``__class__`` swap, so the weights are untouched.
    """

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        cache_position: torch.LongTensor | None = None,
        gdn_cu_seqlens: torch.Tensor | None = None,
        gdn_seq_idx: torch.Tensor | None = None,
        gdn_cu_seqlens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            if gdn_cu_seqlens is None or gdn_seq_idx is None:
                raise ValueError(
                    "packed GDN layer requires gdn_cu_seqlens/gdn_seq_idx (packed BS=1)"
                )
            assert past_key_values is None, "packed GDN forward is cache-free"
            hidden_states = _packed_gdn_forward(
                self.linear_attn, hidden_states, gdn_cu_seqlens, gdn_seq_idx, gdn_cu_seqlens_cpu
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def install_packed_decoder_layers(text_model: nn.Module) -> None:
    """Swap the ``__class__`` of every HF decoder layer to the packed variant."""
    for layer in text_model.layers:  # type: ignore[attr-defined]
        if type(layer) not in (Qwen3_5DecoderLayer, _PackedQwen3_5DecoderLayer):
            raise TypeError(f"expected stock Qwen3_5DecoderLayer, got {type(layer).__name__}")
        layer.__class__ = _PackedQwen3_5DecoderLayer
