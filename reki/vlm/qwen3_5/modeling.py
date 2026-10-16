"""Qwen3.5 with the action expert: ``Qwen3_5VLMWithExpert``.

Samples are packed into one row ([1, total_L, D], batch size 1) with ``cu_seqlens`` marking
the sample boundaries. The VLM pass is the stock HF ``Qwen3_5TextModel`` forward; packing
reaches it as

  full_attention   — flash-attention varlen kwargs (``cu_seq_lens_q/k``, ``max_length_q/k``),
                     passed through HF's ``FlashAttentionKwargs``
  linear_attention — ``_PackedQwen3_5DecoderLayer`` (``hf_patches``), which passes
                     ``gdn_cu_seqlens`` and ``gdn_seq_idx`` to the FLA
                     ``chunk_gated_delta_rule`` and ``causal_conv1d_fn`` kernels

The action expert (``Qwen3_5ActionExpert``) runs, per block, self-attention over the packed
expert rows (registers ∥ actions), cross-attention to the final-layer VLM output of the user
turn, then SwiGLU. Inference reruns the VLM pass every env step and keeps only
``last_hidden_state``: there is no VLM KV cache.
``hf_patches`` holds the HF patches; the inference expert is ``Qwen3_5ExpertInference``
(``inference``).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, NamedTuple
from typing import cast as type_cast

import torch
import torch.nn as nn
import torch.utils.checkpoint
from flash_attn import flash_attn_varlen_func
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from ..adarms import RegisterAdaRMSNorm
from ..adarms import gated_residual as _gated_residual
from ..adarms_base import PiGemmaRMSNorm as AdaRMSNorm
from ..common import cu_seqlens as _cu_seqlens
from ..common import register_shared_row_offsets
from .action_expert import Qwen3_5ActionExpert
from .hf_patches import (
    fla_inference_configs,
    install_packed_decoder_layers,
    install_vision_varlen,
    linearize_vision_patch_embed,
)
from .inference import CrossAttnCache, Qwen3_5ExpertInference


def _host_copies(ts: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    """Host copies of many small tensors in one device→host transfer (one sync)."""
    present = [t for t in ts if t is not None]
    if not present:
        return list(ts)
    flat = torch.cat([t.reshape(-1).to(present[0].dtype) for t in present]).cpu()
    parts = iter(flat.split([t.numel() for t in present]))
    return [None if t is None else next(parts).view(t.shape).to(t.dtype) for t in ts]


class _RegisterAttentionIndex(NamedTuple):
    """The register-memory self-attention of the expert (training), derived once per forward:
    the key rows gathered per query segment, and the varlen bounds of the one call."""

    kv_idx: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    max_q: int
    max_kv: int


class _CrossAttnIndex(NamedTuple):
    """The cross-attention source rows of the expert (training), derived once per forward."""

    flat_idx: torch.Tensor
    src_counts: torch.Tensor
    cu_src: torch.Tensor
    max_src: int
    src_pe: tuple[torch.Tensor, torch.Tensor]


@dataclass
class ExpertForwardState:
    """The per-forward inputs of the action expert, shared by the policy and
    ``Qwen3_5VLMWithExpert``.

    ``embed_vlm_inputs`` starts every forward (training and inference) with a fresh state:
    that is the only reset. The policy then sets the expert inputs, and ``forward`` caches
    what it derives from them, so a gradient-checkpoint recompute reuses the same values.
    """

    position_ids: torch.Tensor | None = None
    """[3, 1, total_L] MRoPE positions of the packed VLM stream (``embed_vlm_inputs``)."""
    expert_starts: list[int] | None = None
    """Per sample: the MRoPE position after its last token (``embed_vlm_inputs``)."""

    flow_draws: int = 1
    """Flow draws per owner sample (training)."""
    cs_per_sample: torch.Tensor | None = None
    """Expert rows per block: register rows + action rows."""
    expert_is_register: torch.Tensor | None = None
    """[N] True on register rows (training; None at inference)."""
    expert_frame_pos: torch.Tensor | None = None
    """Per owner sample: its frame index in the history group (training)."""
    reg_kv_flat_idx: torch.Tensor | None = None
    reg_kv_counts: torch.Tensor | None = None
    """Register-memory gather indices and per-block counts (training)."""

    register_attention: _RegisterAttentionIndex | None = None
    cross_attn: _CrossAttnIndex | None = None


class Qwen3_5VLMWithExpert(Qwen3_5ExpertInference, nn.Module):
    """The Qwen3.5 VLM (``vlm``), the action expert (``action_expert``) and the per-embodiment
    state MLPs (``state_mlps``). The VLM pass runs over packed samples (BS=1, ``cu_seqlens``);
    the expert runs over packed register and action rows and cross-attends the final-layer VLM
    output. Training: ``embed_vlm_inputs`` then ``forward``. Inference: ``forward`` with
    ``prefill_only=True`` (the VLM pass without a KV cache), then the
    ``Qwen3_5ExpertInference`` methods.
    """

    def __init__(
        self,
        *,
        vlm: PreTrainedModel,
        action_expert: Qwen3_5ActionExpert,
        train_vision_blocks: int = -1,
        train_expert_only: bool = False,
        adarms_cond_dim: int,
        action_expert_num_registers: int,
    ) -> None:
        super().__init__()
        self.train_vision_blocks = train_vision_blocks
        self.train_expert_only = train_expert_only
        self.num_registers: int = action_expert_num_registers

        self.vlm = vlm
        attn_impl = vlm.config.text_config._attn_implementation
        if attn_impl != "flash_attention_2":
            raise ValueError(
                "Qwen3_5VLMWithExpert requires the text backbone loaded with "
                f"attn_implementation='flash_attention_2' (got {attn_impl!r}); "
                "packed BS=1 relies on the FA varlen kwargs."
            )
        install_packed_decoder_layers(self.text_backbone)
        self.action_expert = action_expert
        self.state_mlps: nn.ModuleDict | None = None

        # Do not cast the whole module: Qwen's rotary inv_freq must stay float32.

        self._replace_expert_norms(adarms_cond_dim)

        self._set_requires_grad()
        self.fwd = ExpertForwardState()
        self._gradient_checkpointing: bool = False
        self.collect_gate_stats = False
        self.gate_stats: dict[str, float] = {}

    @property
    def gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        """Full-layer checkpointing: HF per-decoder-layer checkpointing on the VLM
        (use_reentrant=False), per-block torch.utils.checkpoint on the expert stack."""
        self._gradient_checkpointing = value
        backbone = self.text_backbone
        if value:
            backbone.gradient_checkpointing_enable(  # type: ignore[operator]
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            backbone.gradient_checkpointing_disable()  # type: ignore[operator]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        action_expert_hidden_size: int,
        action_expert_intermediate_size: int,
        action_expert_num_layers: int,
        action_expert_num_heads: int,
        train_vision_blocks: int = -1,
        train_expert_only: bool = False,
        adarms_cond_dim: int,
        action_expert_num_registers: int,
        load_vlm_weights: bool = True,
    ) -> "Qwen3_5VLMWithExpert":
        """The pretrained VLM ``model_id`` with the folded linear patch embed, varlen vision
        attention and a newly initialized action expert. ``load_vlm_weights=False`` builds the
        VLM from its config only (random weights, for a full checkpoint load after)."""
        # Text needs FA varlen kwargs for packed BS=1; vision loads as sdpa,
        # then install_vision_varlen replaces its attention with FA varlen.
        attn_implementation = {"text_config": "flash_attention_2", "vision_config": "sdpa"}
        if load_vlm_weights:
            vlm = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=dtype, attn_implementation=attn_implementation
            )
        else:
            # No random init (~90 s on CPU): the checkpoint load overwrites every VLM weight.
            with no_init_weights():
                vlm = AutoModelForImageTextToText.from_config(
                    AutoConfig.from_pretrained(model_id),
                    dtype=dtype,
                    attn_implementation=attn_implementation,
                )
            vlm.tie_weights()  # no_init_weights also skips the lm_head / embed_tokens tie
        vlm = type_cast(PreTrainedModel, vlm)
        linearize_vision_patch_embed(vlm)
        install_vision_varlen(vlm)
        tc = vlm.config.text_config
        rope_params = getattr(tc, "rope_parameters", None) or {}
        partial = rope_params.get("partial_rotary_factor") or getattr(
            tc, "partial_rotary_factor", None
        )
        rotary_dim = int(tc.head_dim * (partial if partial is not None else 1.0))
        expert_head_dim = action_expert_hidden_size // action_expert_num_heads
        if expert_head_dim < rotary_dim:
            raise ValueError(
                f"action expert head_dim={expert_head_dim} (hidden "
                f"{action_expert_hidden_size} / {action_expert_num_heads} heads) is "
                f"smaller than the VLM rotary dim {rotary_dim} — the shared rotary "
                "cos/sin cannot be applied."
            )
        action_expert = Qwen3_5ActionExpert(
            vlm.config.text_config,
            expert_hidden=action_expert_hidden_size,
            expert_intermediate=action_expert_intermediate_size,
            num_layers=action_expert_num_layers,
            num_heads=action_expert_num_heads,
            num_registers=action_expert_num_registers,
        ).to(device=vlm.device, dtype=dtype)
        return cls(
            vlm=vlm,
            action_expert=action_expert,
            train_vision_blocks=train_vision_blocks,
            train_expert_only=train_expert_only,
            adarms_cond_dim=adarms_cond_dim,
            action_expert_num_registers=action_expert_num_registers,
        )

    @property
    def image_encoder(self) -> nn.Module:
        return type_cast(nn.Module, self.vlm.model.visual)

    @property
    def text_backbone(self) -> nn.Module:
        return type_cast(nn.Module, self.vlm.model.language_model)

    @property
    def mm_model(self) -> nn.Module:
        return type_cast(nn.Module, self.vlm.model)

    def expert_layer(self, layer_idx: int) -> Any:
        return self.action_expert.layers[layer_idx]  # type: ignore[index]

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_backbone.embed_tokens(tokens)  # type: ignore[operator]

    def init_state_mlps(self, embodiments: dict[str, int], hidden: int) -> None:
        """Per-embodiment proprio MLP bank (state_dim → ``hidden``, ``Linear → GELU``). It gives
        one token per sample to the AdaLN cond. ``hidden`` is the expert width.

        A zero-init square projection (``state_adaln_proj``) follows the bank, so the bank has
        no output projection. Do not add a zero-init one: two stacked zero-init projections get
        no gradient, and proprio never trains.
        """
        self.state_mlps = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(d, hidden), nn.GELU())
                for name, d in embodiments.items()
            }
        )
        ref = next(self.parameters())
        self.state_mlps.to(device=ref.device, dtype=ref.dtype)

    def _replace_expert_norms(self, adarms_cond_dim: int) -> None:
        layers = type_cast(nn.ModuleList, self.action_expert.layers)
        hidden: int = layers[0].input_layernorm.weight.shape[0]  # type: ignore[attr-defined]
        eps = self.vlm.config.text_config.rms_norm_eps
        ref_param = next(self.action_expert.parameters())
        device, dtype = ref_param.device, ref_param.dtype

        def _make_adarms() -> AdaRMSNorm:
            norm = RegisterAdaRMSNorm(hidden, eps=eps, cond_dim=adarms_cond_dim).to(
                device=device, dtype=dtype
            )
            assert norm.dense is not None
            nn.init.zeros_(norm.dense.bias)
            return norm

        for layer in layers:
            layer.input_layernorm = _make_adarms()
            layer.cross_attn_layernorm = _make_adarms()
            layer.post_attention_layernorm = _make_adarms()
        self.action_expert.norm = _make_adarms()

    @torch.no_grad()
    def _collect_adarms_gate_stats(
        self, adarms_cond: torch.Tensor, cs_per_sample: torch.Tensor
    ) -> None:
        """Mean |gate| of the AdaRMS modulations per expert layer, over the τ draws of the batch.
        Runs on logging steps only.

        AdaLN-Zero starts every gate at zero. If the gates stay near zero, the expert output does
        not depend on τ (or on proprio under state_encoding="adaln"), and the training loss can
        still look normal. The cond is constant within a block, so one row per (owner, draw)
        block covers every τ. The method reads the last row of each block, which is always an
        action row (register rows come first).
        """
        cond = adarms_cond[cs_per_sample.cumsum(0) - 1]
        gates = []
        layers: Any = self.action_expert.layers
        for layer in layers:
            mods = [
                norm.dense(cond).chunk(3, dim=-1)
                for norm in (
                    layer.input_layernorm,
                    layer.cross_attn_layernorm,
                    layer.post_attention_layernorm,
                )
            ]
            gates.append(torch.stack([m[2].abs().mean() for m in mods]).mean())
        g = torch.stack(gates)
        self.gate_stats = {
            "adarms_gate_mean": round(g.mean().item(), 4),
            "adarms_gate_last_layer": round(g[-1].item(), 4),
        }

    def _set_requires_grad(self) -> None:
        if self.train_vision_blocks >= 0:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            n = self.train_vision_blocks
            if n > 0:
                blocks: nn.ModuleList = self.image_encoder.blocks  # type: ignore[assignment]
                for m in [*blocks[-n:], getattr(self.image_encoder, "merger")]:
                    for p in m.parameters():
                        p.requires_grad = True
        if self.train_expert_only:
            for p in self.vlm.parameters():
                p.requires_grad = False

    def embed_vlm_inputs(
        self,
        input_ids: torch.Tensor,
        vision_inputs: list[dict[str, Any]],
        seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed text and images and start a fresh ``ExpertForwardState``.

        Packed (training): ``input_ids`` is [total_L] and ``seq_lens`` gives the sample
        boundaries (``_embed_packed``). ``seq_lens=None``: one padded sample (inference).
        """
        lang_emb = self.embed_language_tokens(input_ids)
        device = self.vlm.device
        dtype = self.vlm.dtype

        if seq_lens is None:
            prefix = lang_emb.unsqueeze(0) if lang_emb.ndim == 2 else lang_emb
            self._embed_single_sample_state(
                input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0),
                prefix,
                vision_inputs,
                device,
                dtype,
            )
            return prefix

        assert input_ids.ndim == 1, "packed input_ids must be [total_L]"
        return self._embed_packed(
            input_ids, lang_emb.unsqueeze(0), vision_inputs, seq_lens, device, dtype
        )

    def _embed_packed(
        self,
        input_ids: torch.Tensor,
        prefix: torch.Tensor,
        vision_inputs: list[dict[str, Any]],
        seq_lens: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Packed [total_L] path. One vision-tower call covers every image of the batch. The
        tower attends per image through its cu_seqlens, so the result equals per-sample calls.
        The MRoPE positions use host copies of the small integer inputs, with no per-sample
        device sync. The features, positions and expert starts equal those of per-sample
        ``_embed_one_sample_inplace`` calls."""
        n = len(vision_inputs)
        lens = seq_lens.tolist()
        ids_cpu = input_ids.cpu()
        with_px = ["pixel_values" in v for v in vision_inputs]
        grid_cpu = _host_copies([v.get("image_grid_thw") for v in vision_inputs])
        mm_cpu = _host_copies([v.get("mm_token_type_ids") for v in vision_inputs])

        if any(with_px):
            px = [v["pixel_values"] for v, h in zip(vision_inputs, with_px) if h]
            gx = [g for g, h in zip(grid_cpu, with_px) if h]
            assert all(g is not None for g in gx), "pixel_values without image_grid_thw"
            gx = type_cast(list[torch.Tensor], gx)
            with torch.profiler.record_function("vision_encode"):
                vision_out = self.mm_model.get_image_features(  # type: ignore[attr-defined]
                    pixel_values=torch.cat(px).to(device=device, dtype=dtype),
                    image_grid_thw=torch.cat(gx),
                )
            image_features = torch.cat(vision_out.pooler_output, dim=0)
            seg_has_px = torch.tensor(with_px).repeat_interleave(torch.tensor(lens))
            img_tok = (ids_cpu == self.mm_model.config.image_token_id) & seg_has_px  # type: ignore[attr-defined]
            # Per segment, so a surplus in one segment cannot hide a deficit in another.
            feat_per_img = iter(f.shape[0] for f in vision_out.pooler_output)
            seg_tok = [int(t.sum()) for t, h in zip(img_tok.split(lens), with_px) if h]
            seg_feat = [sum(next(feat_per_img) for _ in range(g.shape[0])) for g in gx]
            bad = [(i, t, f) for i, (t, f) in enumerate(zip(seg_tok, seg_feat)) if t != f]
            assert not bad, (
                "Image features and image tokens do not match per segment "
                f"(segment-with-pixels index, tokens, features): {bad[:5]}"
            )
            mask = img_tok.to(device, non_blocking=True).view(1, -1, 1).expand_as(prefix)
            prefix = prefix.masked_scatter(mask, image_features)

        pos_parts: list[torch.Tensor] = []
        expert_starts: list[int] = []
        off = 0
        for i in range(n):
            ids_i = ids_cpu[off : off + lens[i]].unsqueeze(0)
            off += lens[i]
            mm_i = mm_cpu[i]
            if mm_i is None:
                mm_i = torch.zeros_like(ids_i, dtype=torch.int32)
            pos_i, start_i = self._mrope_positions(ids_i, mm_i, grid_cpu[i])
            pos_parts.append(pos_i)
            expert_starts.append(start_i)
        self.fwd = ExpertForwardState(
            position_ids=torch.cat(pos_parts, dim=2).to(device, non_blocking=True),
            expert_starts=expert_starts,
        )
        return prefix

    def _embed_one_sample_inplace(
        self,
        sample_ids_2d: torch.Tensor,
        sample_prefix: torch.Tensor,
        sample_vision: dict[str, Any],
        prefix_out: torch.Tensor,
        sample_slice: slice,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """Run the vision tower and scatter its features into ``prefix_out``; return the
        ``(3, 1, L)`` position_ids and ``seq_len + mrope_delta`` (Qwen3.5's next text-token
        position)."""
        grid_thw = sample_vision.get("image_grid_thw")
        mm_ids = sample_vision.get("mm_token_type_ids")
        if mm_ids is None:
            mm_ids = torch.zeros_like(sample_ids_2d, dtype=torch.int32)

        if "pixel_values" in sample_vision:
            assert grid_thw is not None
            with torch.profiler.record_function("vision_encode"):
                vision_out = self.mm_model.get_image_features(  # type: ignore[attr-defined]
                    pixel_values=sample_vision["pixel_values"].to(device=device, dtype=dtype),
                    image_grid_thw=grid_thw.to(device=device),
                )
            image_features = torch.cat(vision_out.pooler_output, dim=0)
            image_mask, _ = self.mm_model.get_placeholder_mask(  # type: ignore[attr-defined]
                input_ids=sample_ids_2d,
                inputs_embeds=sample_prefix,
                image_features=image_features,
            )
            updated = sample_prefix.masked_scatter(image_mask, image_features)
            prefix_out[:, sample_slice, :] = updated

        return self._mrope_positions(
            sample_ids_2d.to(device),
            mm_ids.to(device),
            grid_thw.to(device) if grid_thw is not None else None,
        )

    def _mrope_positions(
        self,
        sample_ids_2d: torch.Tensor,
        mm_ids: torch.Tensor,
        grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        """``(3, 1, L)`` MRoPE position_ids + natural next position of one sample, on the
        device of its inputs (host tensors in the packed training path)."""
        position_ids, mrope_deltas = self.mm_model.get_rope_index(  # type: ignore[attr-defined]
            input_ids=sample_ids_2d,
            mm_token_type_ids=mm_ids,
            image_grid_thw=grid_thw,
            attention_mask=None,
        )
        L_sample = int(sample_ids_2d.shape[1])
        assert position_ids.shape == (3, 1, L_sample), (
            f"position_ids shape {tuple(position_ids.shape)} != (3, 1, {L_sample})"
        )
        assert (position_ids >= 0).all(), "MRoPE positions contain negative values"
        if L_sample > 1:
            t_axis = position_ids[0, 0]
            assert (t_axis[1:] >= t_axis[:-1]).all(), (
                "MRoPE T-axis not monotonically non-decreasing within sample"
            )
        text_mask = mm_ids.to(position_ids.device).squeeze(0) == 0
        if text_mask.any():
            text_pos = position_ids[..., text_mask]
            assert (text_pos[0] == text_pos[1]).all() and (text_pos[0] == text_pos[2]).all(), (
                "MRoPE: text-token positions differ across (T, H, W) axes"
            )
        if grid_thw is not None and grid_thw.shape[0] > 0:
            image_mask = mm_ids.to(position_ids.device).squeeze(0) == 1
            if image_mask.any():
                merge = self.mm_model.config.vision_config.spatial_merge_size**2  # type: ignore[attr-defined]
                tokens_per_img = (grid_thw.prod(dim=-1) // merge).to(torch.long)
                img_t = position_ids[0, 0, image_mask]
                cursor = 0
                for n in tokens_per_img.tolist():
                    if n == 0:
                        continue
                    block = img_t[cursor : cursor + n]
                    assert (block == block[0]).all(), (
                        "MRoPE: T axis varies within a single image grid "
                        f"(grid {cursor}..{cursor + n}, values {block.unique().tolist()})"
                    )
                    cursor += n
        # natural_next_pos (max position + 1) is the expert MRoPE anchor at inference.
        natural_next_pos = int(sample_ids_2d.shape[1] + mrope_deltas[0].item())
        assert natural_next_pos == int(position_ids.max().item()) + 1, (
            f"natural_next_pos={natural_next_pos} != max(position_ids)+1="
            f"{int(position_ids.max().item()) + 1} (mrope_deltas inconsistent)"
        )
        return position_ids, natural_next_pos

    def _embed_single_sample_state(
        self,
        input_ids_2d: torch.Tensor,
        prefix: torch.Tensor,
        vision_inputs: list[dict[str, Any]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Embed one padded sample in place (inference)."""
        assert len(vision_inputs) == 1
        sample_slice = slice(0, input_ids_2d.shape[1])
        pos, start = self._embed_one_sample_inplace(
            input_ids_2d,
            prefix,
            vision_inputs[0],
            prefix,
            sample_slice,
            device,
            dtype,
        )
        self.fwd = ExpertForwardState(position_ids=pos, expert_starts=[start])

    def _expert_position_embeddings(
        self,
        t_starts: list[int],
        cs_per_sample: torch.Tensor,
        reference_hidden: torch.Tensor,
        row_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin ``[1, N, head_dim]`` for the packed expert rows. Virtual sample b has
        ``cs_per_sample[b]`` consecutive positions from ``t_starts[b]``. ``row_offsets`` replaces
        the offsets within a block. The draw-shared register layout uses it to restart the
        action rows of each draw at R."""
        device = reference_hidden.device
        starts = torch.tensor(t_starts, device=device, dtype=torch.long)
        cs_long = cs_per_sample.long().to(device)
        if row_offsets is None:
            N = int(cs_long.sum().item())
            cu_e = torch.zeros(cs_long.shape[0] + 1, dtype=torch.long, device=device)
            cu_e[1:] = cs_long.cumsum(0)
            row_offsets = torch.arange(N, device=device, dtype=torch.long) - cu_e[
                :-1
            ].repeat_interleave(cs_long)
        positions = starts.repeat_interleave(cs_long) + row_offsets
        mrope_positions = positions.view(1, -1).unsqueeze(0).expand(3, -1, -1)
        return self.text_backbone.rotary_emb(reference_hidden, mrope_positions)  # type: ignore[attr-defined]

    def _register_sa_position_embeddings(
        self,
        frame_pos: torch.Tensor,
        cs_per_sample: torch.Tensor,
        reference_hidden: torch.Tensor,
        row_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MRoPE cos/sin for the expert self-attention under register memory.

        The temporal axis holds the frame index. Both spatial axes hold the row slot in the
        frame: register i is slot i, action t is slot R+t. The positions stay small (frame ≤ K),
        so inference uses window-local frame indices without bf16 error at large positions.
        ``cs_per_sample`` is the R register rows plus the action rows of each virtual sample.
        ``frame_pos`` has one entry per virtual sample.
        """
        device = reference_hidden.device
        cs_long = cs_per_sample.long().to(device)
        if row_offsets is None:
            N = int(cs_long.sum().item())
            cu_e = torch.zeros(cs_long.shape[0] + 1, dtype=torch.long, device=device)
            cu_e[1:] = cs_long.cumsum(0)
            slot = torch.arange(N, device=device) - cu_e[:-1].repeat_interleave(cs_long)
        else:
            slot = row_offsets
        frame = frame_pos.long().to(device).repeat_interleave(cs_long)
        mrope_positions = torch.stack([frame, slot, slot], dim=0).unsqueeze(1)
        return self.text_backbone.rotary_emb(reference_hidden, mrope_positions)  # type: ignore[attr-defined]

    def _seq_idx(self, seq_lens: torch.Tensor, total_L: int) -> torch.Tensor:
        """[1, total_L] int32 sample index for causal_conv1d_fn."""
        return torch.repeat_interleave(
            torch.arange(seq_lens.shape[0], device=seq_lens.device, dtype=torch.int32),
            seq_lens.to(torch.int32),
        ).view(1, total_L)

    def register_memory_attention(
        self,
        e_attn: Any,
        q_e: torch.Tensor,
        k_e: torch.Tensor,
        v_e: torch.Tensor,
        head_dim: int,
        N: int,
    ) -> torch.Tensor:
        """Expert self-attention under register memory, as one varlen call over per-segment KV.

        The query segments keep their order: per owner block, one segment of R registers, then
        one segment per draw. A register segment of frame f attends the register rows of frames
        <= f in its group. It never attends noisy actions, so it does not depend on τ and
        inference can cache it. An action segment attends the same register rows plus the action
        rows of its own draw. The KV indices are built once per forward. The KV rows of each
        segment are gathered once per layer, so the softmax over the union and its gradient are
        exact.
        """
        fw = self.fwd
        if fw.register_attention is None:
            assert fw.cs_per_sample is not None and fw.reg_kv_counts is not None
            assert fw.reg_kv_flat_idx is not None
            R = self.num_registers
            dev = q_e.device
            cs_e = fw.cs_per_sample.long().to(dev)
            B = cs_e.shape[0]
            nd = fw.flow_draws
            cs_a = (cs_e - R) // nd
            cu_e = torch.zeros(B + 1, dtype=torch.long, device=dev)
            cu_e[1:] = cs_e.cumsum(0)
            mem = fw.reg_kv_counts.long().to(dev)
            cu_mem = torch.zeros(B + 1, dtype=torch.long, device=dev)
            cu_mem[1:] = mem.cumsum(0)
            seg_draw = torch.arange(nd + 1, device=dev).repeat(B)  # 0 = the register segment
            seg_own = torch.arange(B, device=dev).repeat_interleave(nd + 1)
            q_len = torch.where(seg_draw == 0, R, cs_a[seg_own])
            kv_len = mem[seg_own] + torch.where(seg_draw == 0, 0, cs_a[seg_own])
            chk = torch.stack(
                [((cs_e - R) % nd).max(), cs_a.min(), q_len.max(), kv_len.max(), kv_len.sum()]
            ).tolist()
            assert chk[0] == 0, "register memory: owner action rows not divisible by flow_draws"
            assert chk[1] > 0, "register memory: empty action segment"
            max_q_seg, max_kv, total = chk[2], chk[3], chk[4]
            seg = torch.arange(seg_own.shape[0], device=dev).repeat_interleave(
                kv_len, output_size=total
            )
            cu_kv = torch.zeros(seg_own.shape[0] + 1, dtype=torch.long, device=dev)
            cu_kv[1:] = kv_len.cumsum(0)
            j = torch.arange(total, device=dev) - cu_kv[seg]
            o, d = seg_own[seg], seg_draw[seg]
            reg_rows = fw.reg_kv_flat_idx[(cu_mem[o] + j).clamp(max=cu_mem[-1] - 1)]
            act_rows = cu_e[o] + R + (d - 1).clamp(min=0) * cs_a[o] + (j - mem[o])
            fw.register_attention = _RegisterAttentionIndex(
                kv_idx=torch.where(j < mem[o], reg_rows, act_rows),
                cu_q=_cu_seqlens(q_len),
                cu_kv=cu_kv.to(torch.int32),
                max_q=int(max_q_seg),
                max_kv=int(max_kv),
            )
        idx, cu_q, cu_kv, max_q_seg, max_kv = fw.register_attention
        y = flash_attn_varlen_func(
            q_e,
            k_e[idx],
            v_e[idx],
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q_seg,
            max_seqlen_k=max_kv,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )
        return e_attn.o_proj(y.reshape(N, -1))

    def _cross_attn_source(
        self,
        vlm_out: torch.Tensor,
        seq_lens: torch.Tensor,
        robot_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Gather the expert's cross-attention source.

        Rows of the final-layer VLM output of each owner sample. The indices and the rows'
        MRoPE cos/sin depend on neither τ nor the layer, so they are cached per forward in
        ``fwd.cross_attn``; the rows themselves are re-gathered from ``vlm_out`` on every
        call, to stay in the autograd graph.
        Returns (src [S, H_vlm], cu_src, src_counts, max_src, s_cos, s_sin).
        """
        fw = self.fwd
        device = vlm_out.device
        vlm_flat = vlm_out.reshape(-1, vlm_out.shape[-1])
        total_L = vlm_flat.shape[0]
        if fw.cross_attn is None:
            sample_idx = self._seq_idx(seq_lens.to(torch.int32), total_L).view(-1).long()
            own = sample_idx.unsqueeze(0) == robot_indices.to(device).unsqueeze(1)
            src_counts = own.sum(dim=1).to(torch.int32)
            assert int(src_counts.min().item()) > 0, "a virtual sample has an empty CA source"
            flat_idx = own.nonzero(as_tuple=False)[:, 1]
            assert fw.position_ids is not None
            src_pos = fw.position_ids[:, 0, :][:, flat_idx].unsqueeze(1)
            s_cos, s_sin = self.text_backbone.rotary_emb(vlm_out, src_pos)  # type: ignore[attr-defined]
            fw.cross_attn = _CrossAttnIndex(
                flat_idx=flat_idx,
                src_counts=src_counts,
                cu_src=_cu_seqlens(src_counts),
                max_src=int(src_counts.max().item()),
                src_pe=(s_cos.squeeze(0), s_sin.squeeze(0)),
            )
        ca = fw.cross_attn
        return (
            vlm_flat[ca.flat_idx],
            ca.cu_src,
            ca.src_counts,
            ca.max_src,
            ca.src_pe[0],
            ca.src_pe[1],
        )

    def _action_expert_stack(
        self,
        vlm_out: torch.Tensor,
        action_hidden: torch.Tensor,
        seq_lens: torch.Tensor,
        robot_indices: torch.Tensor,
        action_pe: tuple[torch.Tensor, torch.Tensor],
        sa_pe: tuple[torch.Tensor, torch.Tensor],
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        """The expert stack in training over the packed expert rows (registers ∥ actions).
        Returns the packed hidden ``[N, expert_hidden]`` before the final norm.

        Each block runs SA, CA, then SwiGLU, with pre-norm gated residuals: τ-AdaRMS on action
        rows, plain norms on register rows. SA is the register-memory attention (one varlen
        call). Every block projects its own CA K/V from the same final-layer rows
        (``_cross_attn_source``). With several draws, the k draw blocks of an owner form one CA
        query segment over one copy of its source. CA queries are independent, so this is
        exact.
        """
        assert self.fwd.cs_per_sample is not None
        cs_per_sample = self.fwd.cs_per_sample.to(action_hidden.device)
        N = action_hidden.shape[0]
        B_e_own = robot_indices.shape[0]
        n_blocks = cs_per_sample.shape[0]
        n_draws_q = n_blocks // B_e_own
        is_register = self.fwd.expert_is_register
        assert is_register is not None
        if self.collect_gate_stats:
            self._collect_adarms_gate_stats(adarms_cond, cs_per_sample)

        src, cu_src, src_counts, max_src, s_cos, s_sin = self._cross_attn_source(
            vlm_out, seq_lens, robot_indices
        )
        a_cos, a_sin = action_pe[0].squeeze(0), action_pe[1].squeeze(0)

        cu_q_e = _cu_seqlens(cs_per_sample.to(torch.int32))
        max_q = int(cs_per_sample.max().item())
        cu_q_ca, max_q_ca = cu_q_e, max_q
        if n_draws_q > 1:
            per_owner_q = cs_per_sample.to(torch.int32).view(B_e_own, n_draws_q).sum(dim=1)
            cu_q_ca = _cu_seqlens(per_owner_q)
            max_q_ca = int(per_owner_q.max().item())

        use_ckpt = self.gradient_checkpointing and self.training

        def _layer_step(action_hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
            e_layer = self.expert_layer(layer_idx)
            e_attn = e_layer.self_attn
            D = e_attn.head_dim

            e_norm, e_gate = e_layer.input_layernorm(
                action_hidden, cond=adarms_cond, is_register=is_register
            )
            assert e_gate is not None
            q_e = e_attn.q_norm(e_attn.q_proj(e_norm).view(N, e_attn.n_q_heads, D))
            k_e = e_attn.k_norm(e_attn.k_proj(e_norm).view(N, e_attn.n_kv_heads, D))
            v_e = e_attn.v_proj(e_norm).view(N, e_attn.n_kv_heads, D)
            q_e, k_e = apply_rotary_pos_emb(
                q_e, k_e, sa_pe[0].squeeze(0), sa_pe[1].squeeze(0), unsqueeze_dim=1
            )
            out_sa = self.register_memory_attention(e_attn, q_e, k_e, v_e, D, N)
            h = _gated_residual(action_hidden, out_sa, e_gate)

            ca_norm, ca_gate = e_layer.cross_attn_layernorm(
                h, cond=adarms_cond, is_register=is_register
            )
            assert ca_gate is not None
            c_attn = e_layer.cross_attn
            q_ca = c_attn.q_norm(c_attn.q_proj(ca_norm).view(N, c_attn.n_q_heads, D))
            k_dummy = torch.empty_like(q_ca[:, :1])
            q_ca, _ = apply_rotary_pos_emb(q_ca, k_dummy, a_cos, a_sin, unsqueeze_dim=1)
            S = src.shape[0]
            k_ca = c_attn.k_norm(c_attn.k_proj(src).view(S, c_attn.n_kv_heads, D))
            _, k_ca = apply_rotary_pos_emb(
                torch.empty_like(k_ca[:, :1]), k_ca, s_cos, s_sin, unsqueeze_dim=1
            )
            v_ca = c_attn.v_proj(src).view(S, c_attn.n_kv_heads, D)
            y_ca = flash_attn_varlen_func(
                q_ca,
                k_ca,
                v_ca,
                cu_seqlens_q=cu_q_ca,
                cu_seqlens_k=cu_src,
                max_seqlen_q=max_q_ca,
                max_seqlen_k=max_src,
                softmax_scale=D**-0.5,
                causal=False,
            ).view(N, -1)
            h = _gated_residual(h, c_attn.o_proj(y_ca), ca_gate)

            h_norm2, gate2 = e_layer.post_attention_layernorm(
                h, cond=adarms_cond, is_register=is_register
            )
            return _gated_residual(h, e_layer.mlp(h_norm2), gate2)

        for layer_idx in range(len(self.action_expert.layers)):  # type: ignore[arg-type]
            if use_ckpt:
                action_hidden = type_cast(
                    torch.Tensor,
                    torch.utils.checkpoint.checkpoint(
                        _layer_step, action_hidden, layer_idx, use_reentrant=False
                    ),
                )
            else:
                action_hidden = _layer_step(action_hidden, layer_idx)
        return action_hidden

    def forward(
        self,
        inputs_embeds: tuple[torch.Tensor | None, torch.Tensor | None],
        seq_lens: torch.Tensor | None = None,
        prefill_only: bool = False,
        adarms_cond: torch.Tensor | None = None,
        robot_indices: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], CrossAttnCache | None]:
        """The VLM pass over ``inputs_embeds`` (packed by ``seq_lens``), plus the expert when
        ``expert_hidden`` is given: ``((vlm_out, expert_out | None), cache)``. ``prefill_only``
        is the inference prefill: fla inference configs and a ``CrossAttnCache`` with only
        ``expert_start`` set, which the caller fills for ``gather_cross_attn_source``. The VLM
        keeps no KV cache."""
        vlm_hidden, expert_hidden = inputs_embeds

        assert vlm_hidden is not None
        text_model = self.text_backbone
        fw = self.fwd

        _, L_total, _ = vlm_hidden.shape
        if seq_lens is None:
            seq_lens = torch.tensor([L_total], device=vlm_hidden.device, dtype=torch.int32)
        cu_vlm = _cu_seqlens(seq_lens.to(torch.int32))
        max_vlm = int(seq_lens.max().item())
        seq_idx = self._seq_idx(seq_lens.to(torch.int32), L_total)

        expert_starts_all = fw.expert_starts
        n_draws = fw.flow_draws
        expert_pe: tuple[torch.Tensor, torch.Tensor] | None = None
        sa_pe: tuple[torch.Tensor, torch.Tensor] | None = None
        if expert_hidden is not None and robot_indices is not None:
            ri_action = robot_indices
            assert expert_starts_all is not None
            cs_per_sample = fw.cs_per_sample
            assert cs_per_sample is not None, "the expert needs fwd.cs_per_sample"
            row_off = (
                register_shared_row_offsets(cs_per_sample, self.num_registers, n_draws)
                if n_draws > 1
                else None
            )
            expert_pe = self._expert_position_embeddings(
                [expert_starts_all[ri] for ri in ri_action.tolist()],
                cs_per_sample,
                vlm_hidden,
                row_offsets=row_off,
            )
            assert fw.expert_frame_pos is not None, "register memory needs fwd.expert_frame_pos"
            sa_pe = self._register_sa_position_embeddings(
                fw.expert_frame_pos, cs_per_sample, vlm_hidden, row_offsets=row_off
            )

        # Always pass cu_seq_lens_q/k explicitly: without them HF's flash-attention path hits a
        # CUDA illegal memory access (transformers 5.3; MRoPE position_ids break its packing
        # detection).
        with fla_inference_configs() if prefill_only else contextlib.nullcontext():
            out = text_model(
                inputs_embeds=vlm_hidden,
                position_ids=fw.position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_vlm,
                cu_seq_lens_k=cu_vlm,
                max_length_q=max_vlm,
                max_length_k=max_vlm,
                gdn_cu_seqlens=cu_vlm.to(torch.long),
                gdn_seq_idx=seq_idx,
            )
        vlm_out = out.last_hidden_state

        cache: CrossAttnCache | None = None
        if prefill_only:
            assert expert_starts_all is not None
            cache = CrossAttnCache()
            cache.expert_start = int(expert_starts_all[0])

        if expert_hidden is not None:
            assert robot_indices is not None and expert_pe is not None and sa_pe is not None
            assert adarms_cond is not None
            action_out = self._action_expert_stack(
                vlm_out, expert_hidden, seq_lens, robot_indices, expert_pe, sa_pe, adarms_cond
            )
            return (vlm_out, self._apply_expert_final_norm(action_out, adarms_cond)), cache
        return (vlm_out, None), cache

    def _apply_expert_final_norm(
        self, expert_h: torch.Tensor, adarms_cond: torch.Tensor
    ) -> torch.Tensor:
        norm = self.action_expert.norm
        assert isinstance(norm, AdaRMSNorm)
        is_register = self.fwd.expert_is_register
        assert is_register is not None
        out, _ = norm(expert_h, cond=adarms_cond, is_register=is_register)
        return out
