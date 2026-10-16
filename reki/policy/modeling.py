"""The REKI policy: a Qwen3.5 VLM with a flow-matching action expert, its training losses
and action sampling."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar
from typing import cast as type_cast

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from reki.vlm.qwen3_5 import check_model_id

from .configuration import RekiConfig
from .constants import (
    ACTION,
    ACTION_IS_PAD,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from .flow_utils import create_sinusoidal_pos_embedding, sample_beta

if TYPE_CHECKING:
    from reki.vlm.qwen3_5.modeling import Qwen3_5VLMWithExpert

R = TypeVar("R")


def _proj_fp32(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a flow-matching ``nn.Linear`` head in fp32 (its params are cast to fp32 in
    ``RekiFlowModel.__init__``): the bf16 mantissa is too coarse for the sinusoidal time
    embedding and the regression target."""
    assert isinstance(module, nn.Linear), f"_proj_fp32 expects nn.Linear, got {type(module)}"
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), module.weight.float(), bias)


def _prepend_registers(
    action_emb: torch.Tensor,
    adarms_cond: torch.Tensor,
    reg_bank: torch.Tensor,
    cs_action: torch.Tensor,
    n_draws: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-owner expert blocks [R registers ∥ draw_0 … draw_{k-1} actions].

    The registers are the same for every flow draw of a sample (zero conditioning,
    self-attention over the registers only, cross-attention to the owner's VLM output). So the
    function builds the R rows once per owner.

    ``cs_action`` is the action row count per (owner, draw), tiled by draw (owner-major).
    ``reg_bank`` is the [R, hidden] register parameter. Returns (suffix', adarms', cs_expert,
    is_register, action_row_idx). adarms' is zero on the register rows (the norm ignores it).
    action_row_idx gathers the action rows back out.
    """
    device = action_emb.device
    n_reg, hidden = reg_bank.shape
    act_own = cs_action.long().view(-1, n_draws).sum(dim=1)
    B_own = act_own.shape[0]
    cs_e = act_own + n_reg
    cu_e = torch.zeros(B_own + 1, dtype=torch.long, device=device)
    cu_e[1:] = cs_e.cumsum(0)
    n_prime = int(cu_e[-1].item())
    register_idx = cu_e[:-1].repeat_interleave(n_reg) + torch.arange(n_reg, device=device).repeat(
        B_own
    )
    is_register = torch.zeros(n_prime, dtype=torch.bool, device=device)
    is_register[register_idx] = True
    action_row_idx = (~is_register).nonzero(as_tuple=True)[0]

    suffix = action_emb.new_zeros((n_prime, hidden))
    suffix[register_idx] = (
        reg_bank.to(suffix.dtype).unsqueeze(0).expand(B_own, n_reg, hidden).reshape(-1, hidden)
    )
    suffix[action_row_idx] = action_emb
    adarms = adarms_cond.new_zeros((n_prime, hidden))
    adarms[action_row_idx] = adarms_cond
    return suffix, adarms, cs_e.to(cs_action.dtype), is_register, action_row_idx


PolicyInputs = TypedDict(
    "PolicyInputs",
    {
        "vision_inputs": list[dict[str, Any]],
        "camera_indices": list[torch.Tensor],
        "action": list[torch.Tensor],
        "action_is_pad": list[torch.Tensor | None],
        "chunk_size": list[int],
        "frame_pos": torch.Tensor,
        "group_index": torch.Tensor,
        "k_eff": list[int],
        "observation.language.tokens": torch.Tensor,
        "seq_lens": torch.Tensor,
    },
    total=False,
)


def build_register_memory_indices(
    cs_expert: torch.Tensor,
    frame_pos_own: torch.Tensor,
    group_own: torch.Tensor,
    n_reg: int,
    n_draws: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat KV gather indices for the expert register-memory attention.

    ``cs_expert`` [B_e_draws] is the expert row count of each block (``n_reg + cs``), tiled by
    draw (owner-major). The memory of block (owner o, draw d) is the register rows of the
    blocks (o', d) where o' is in the same group and ``frame_pos(o') <= frame_pos(o)``. These
    owners are the range ``[o - frame_pos[o], o]``. The training forward builds the registers
    once per owner (``_prepend_registers``) and calls this with ``n_draws=1`` (blocks of
    ``n_reg + k·cs`` rows).

    Returns (flat_idx [Σ(frame_pos+1)·n_reg·n_draws], counts_rows [B_e_draws] int32).
    The segments of a group must be consecutive and in ascending frame order (from collate).
    Otherwise it raises ``ValueError``.
    """
    device = cs_expert.device
    fp = frame_pos_own.to(device=device, dtype=torch.long)
    gi = group_own.to(device=device, dtype=torch.long)
    B_own = fp.shape[0]
    B = cs_expert.shape[0]
    assert B == B_own * n_draws, f"cs_expert {B} != owners {B_own} × draws {n_draws}"
    cs_long = cs_expert.long()
    cu_e = torch.zeros(B + 1, dtype=torch.long, device=device)
    cu_e[1:] = cs_long.cumsum(0)
    o_idx = torch.arange(B_own, device=device)
    start_own = o_idx - fp
    if not bool((start_own >= 0).all()) or not bool((gi[start_own.clamp(min=0)] == gi).all()):
        raise ValueError(
            "register memory: group segments must be consecutive and frame-ascending "
            "among robot samples (collate invariant violated)."
        )
    counts_own = fp + 1
    counts_blk = counts_own.repeat_interleave(n_draws)
    total = int(counts_blk.sum().item())
    cu_cnt = torch.zeros(B + 1, dtype=torch.long, device=device)
    cu_cnt[1:] = counts_blk.cumsum(0)
    blk = torch.repeat_interleave(torch.arange(B, device=device), counts_blk)
    slot = torch.arange(total, device=device) - cu_cnt[blk]
    src_own = start_own[blk // n_draws] + slot
    src_blk = src_own * n_draws + (blk % n_draws)
    reg_rows = cu_e[src_blk].unsqueeze(1) + torch.arange(n_reg, device=device).unsqueeze(0)
    return reg_rows.reshape(-1), (counts_blk * n_reg).to(torch.int32)


class RekiModel(nn.Module):
    """Policy wrapper around ``RekiFlowModel``: batch → loss (training) and batch → action chunk
    (``predict_action_chunk``, inference), plus the per-episode register cache."""

    def __init__(self, config: RekiConfig, load_vlm_weights: bool = True):
        super().__init__()
        self.config = config

        self._register_cache: Any = None
        self._compiled = False

        self.model = RekiFlowModel(config, load_vlm_weights=load_vlm_weights)

        if config.use_camera_role_tokens:
            self._init_camera_role_embeddings()

    def _init_camera_role_embeddings(self) -> None:
        """Initialize the camera-role tokens (spare embedding rows) from ``<|vision_start|>``,
        the token each of them precedes."""
        from reki.vlm.qwen3_5.preprocessor import CAMERA_ROLE_TOKEN_IDS

        vlm = self.model.vlm_with_expert.vlm
        embed = type_cast(nn.Embedding, vlm.get_input_embeddings())
        with torch.no_grad():
            seed = embed.weight[vlm.config.vision_start_token_id].clone()
            for tid in CAMERA_ROLE_TOKEN_IDS.values():
                embed.weight[tid].copy_(seed)

    def reset_episode(self) -> None:
        """Clear the per-episode register memory.

        Call at ``env.reset()`` so the next ``predict_action_chunk`` starts fresh. The cache
        object and its buffers are kept (emptied in place): a compiled expert reads them at
        fixed addresses.
        """
        if self._register_cache is not None:
            self._register_cache.reset()

    def compile_inference(self) -> None:
        """Compile the expert (``RekiFlowModel.compile_expert``: one compile for every cache fill
        level, prompt length and step count) and replay the prefill as one CUDA graph per input
        layout. The first predict compiles; a new prompt layout captures a new prefill graph."""
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        self.model.compile_expert()
        self.model.enable_prefill_graphs()
        self._compiled = True

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write ``config.json`` and ``model.safetensors``. ``save_model`` stores the tied
        ``embed_tokens`` / ``lm_head`` pair once and records the tie in the file metadata.
        ``load_pretrained_weights`` and the export expect this layout."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / "config.json", "w") as f:
            json.dump(self.config.to_json_dict(), f, indent=4)
        safetensors.torch.save_model(self, str(save_directory / "model.safetensors"))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        """Predict an action chunk from a preprocessed batch.

        The batch has the layout of REKI's training preprocessor:

        - ``batch['vision_inputs']``: one dict per sample,
          ``{pixel_values, image_grid_thw, mm_token_type_ids}``.
        - ``batch[OBS_LANGUAGE_TOKENS]``: [L] or [B, L] input_ids.
        - ``batch['action_dim']`` (number of real action dims).
        - With ``state_encoding="adaln"``: ``batch['state_mlp_embodiment']`` and
          ``batch[OBS_STATE]``.

        Each call reruns the VLM prefill for the current frame; only expert register memory
        persists across calls.
        """
        if self._compiled and torch.is_inference_mode_enabled():
            # fla's intra-card prefill syncs the host under inference_mode (not capturable),
            # and the register cache must not be created as an inference tensor.
            with torch.inference_mode(False), torch.no_grad():
                return self.predict_action_chunk(batch)

        self.eval()
        vision_inputs = batch["vision_inputs"]

        tokens = batch[OBS_LANGUAGE_TOKENS]
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        expert_state: tuple[list[str | None], list[torch.Tensor | None]] | None = None
        if self.config.state_encoding == "adaln":
            embs_batch = batch.get("state_mlp_embodiment")
            states_batch = batch.get(OBS_STATE)
            assert embs_batch is not None and states_batch is not None, (
                f"state_encoding={self.config.state_encoding!r} requires state_mlp_embodiment "
                "+ OBS_STATE in the batch"
            )
            ve_emb: list[str | None] = []
            ve_state: list[torch.Tensor | None] = []
            for b in range(tokens.shape[0]):
                st = states_batch[b]
                if embs_batch[b] is None or st is None:
                    raise ValueError(
                        f"state_encoding={self.config.state_encoding!r}: inference sample {b} "
                        "has no state / embodiment (stateless datasets are unsupported)."
                    )
                if st.ndim == 2:
                    st = st[-1]
                ve_emb.append(embs_batch[b])
                ve_state.append(st.to(device=tokens.device))
            expert_state = (ve_emb, ve_state)

        if self._register_cache is None:
            from reki.vlm.common import RegisterCache

            vwe = self.model.vlm_with_expert
            attn = vwe.expert_layer(0).self_attn
            # Not an inference tensor: the cache outlives the predict and is reset in place.
            with torch.inference_mode(False):
                self._register_cache = RegisterCache(
                    len(vwe.action_expert.layers),  # type: ignore[arg-type]
                    self.config.action_expert_num_registers,
                    max(0, self.config.history_k - 1),
                    attn.n_kv_heads,
                    attn.head_dim,
                    attn.k_proj.weight.dtype,
                    attn.k_proj.weight.device,
                )
        register_cache = self._register_cache

        actions = self.model.sample_actions(
            vision_inputs=vision_inputs,
            tokens=tokens,
            expert_state=expert_state,
            register_cache=register_cache,
            action_dim=int(batch["action_dim"]),
        )
        return actions

    def prepare_action(
        self, batch: "PolicyInputs"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand K-frame samples into B_expert = sum(K_eff) virtual expert samples,
        packed along the time axis.

        Returns (actions [N, max_action_dim] packed, mask same shape, robot_indices
        [B_e] owning-sample idx, anchor_frame [B_e] chunk idx within its sample,
        cs_per_sample [B_e] per-virtual-sample chunk length). N = sum(cs_per_sample).
        """
        action = batch[ACTION]
        cs_list = batch["chunk_size"]
        k_eff_list = batch["k_eff"]
        time_pad_list = batch[ACTION_IS_PAD]

        per_chunk: list[torch.Tensor] = []
        per_chunk_pad: list[torch.Tensor | None] = []
        owner: list[int] = []
        anchor: list[int] = []
        cs_list_out: list[int] = []
        cands: list[tuple[int, int, torch.Tensor, torch.Tensor | None]] = []
        for i, a in enumerate(action):
            k_eff_i = int(k_eff_list[i])
            cs_i = int(cs_list[i])
            n_steps = a.shape[0]
            assert n_steps % cs_i == 0
            K_total = n_steps // cs_i
            a3 = a.reshape(K_total, cs_i, a.shape[-1])[-k_eff_i:]
            p3: torch.Tensor | None = None
            time_pad = time_pad_list[i]
            if time_pad is not None:
                p3 = time_pad.reshape(K_total, cs_i)[-k_eff_i:]
            for k in range(k_eff_i):
                cands.append((i, k, a3[k], p3[k] if p3 is not None else None))
        # Valid lengths of every padded chunk in one device->host transfer.
        pads = [p.bool() for _, _, _, p in cands if p is not None]
        stats = iter(
            torch.stack(
                [torch.stack([(~p).sum(), (p[1:].int() < p[:-1].int()).any().long()]) for p in pads]
            ).tolist()
            if pads
            else []
        )
        for i, k, chunk, pad_k in cands:
            if pad_k is not None:
                n_valid, unpad_after_pad = next(stats)
                assert not unpad_after_pad, "action_is_pad must be a contiguous suffix"
                chunk = chunk[:n_valid]
                pad_k = None
            if chunk.shape[0] == 0:
                continue
            per_chunk.append(chunk)
            per_chunk_pad.append(pad_k)
            owner.append(i)
            anchor.append(k)
            cs_list_out.append(chunk.shape[0])

        if not per_chunk:
            raise ValueError("No robotics samples in batch")

        target_dim = self.config.max_action_dim
        per_chunk_padded: list[torch.Tensor] = []
        per_mask: list[torch.Tensor] = []
        for c, tp in zip(per_chunk, per_chunk_pad):
            valid_dim = c.shape[-1]
            c_pad = F.pad(c, (0, target_dim - valid_dim))
            m = torch.zeros_like(c_pad, dtype=torch.bool)
            m[:, :valid_dim] = True
            if tp is not None:
                m = m & ~tp.to(device=c.device, dtype=torch.bool).unsqueeze(-1)
            per_chunk_padded.append(c_pad)
            per_mask.append(m)
        actions = torch.cat(per_chunk_padded, dim=0)
        mask = torch.cat(per_mask, dim=0)

        device = actions.device
        robot_indices = torch.tensor(owner, dtype=torch.long, device=device)
        anchor_frame = torch.tensor(anchor, dtype=torch.long, device=device)
        cs_per_sample = torch.tensor(cs_list_out, dtype=torch.long, device=device)
        return actions, mask, robot_indices, anchor_frame, cs_per_sample

    def forward(self, batch: "PolicyInputs") -> tuple[torch.Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""
        vision_inputs = batch["vision_inputs"]
        packed_tokens = batch[OBS_LANGUAGE_TOKENS]
        seq_lens = batch["seq_lens"]

        actions, pad_mask, robot_indices, anchor_frame, cs_per_sample = self.prepare_action(batch)
        robot_indices = robot_indices.to(device=packed_tokens.device)
        anchor_frame = anchor_frame.to(device=packed_tokens.device)
        cs_per_sample = cs_per_sample.to(device=packed_tokens.device)

        n_draws = self.config.flow_draws if torch.is_grad_enabled() else 1
        if n_draws > 1:
            cs_long = cs_per_sample.long()
            cu = torch.zeros(cs_long.shape[0] + 1, dtype=torch.long, device=cs_long.device)
            cu[1:] = cs_long.cumsum(0)
            cs_action = cs_long.repeat_interleave(n_draws)
            cu_a = torch.zeros(cs_action.shape[0] + 1, dtype=torch.long, device=cs_action.device)
            cu_a[1:] = cs_action.cumsum(0)
            n_tok = int(cs_action.sum())
            seg = torch.arange(cs_action.shape[0], device=cs_action.device).repeat_interleave(
                cs_action
            )
            offset = torch.arange(n_tok, device=cs_action.device) - cu_a[:-1].repeat_interleave(
                cs_action
            )
            om_idx = cu[seg // n_draws] + offset
            actions = actions[om_idx]
            pad_mask = pad_mask[om_idx]
            cs_per_sample = cs_action.to(cs_per_sample.dtype)

        # Starts a fresh ExpertForwardState; the expert inputs are set on it from here on.
        prefix_embeds = self.model.vlm_with_expert.embed_vlm_inputs(
            input_ids=packed_tokens,
            vision_inputs=vision_inputs,
            seq_lens=seq_lens,
        )
        self.model.vlm_with_expert.fwd.flow_draws = n_draws

        expert_state: tuple[list[str | None], list[torch.Tensor | None]] | None = None
        if self.config.state_encoding == "adaln":
            embs_batch = batch.get("state_mlp_embodiment")
            states_batch = batch.get(OBS_STATE)
            assert embs_batch is not None and states_batch is not None, (
                "state_encoding='adaln' requires state_mlp_embodiment + OBS_STATE in the batch"
            )
            ve_emb: list[str | None] = []
            ve_state: list[torch.Tensor | None] = []
            for owner, anchor in zip(robot_indices.tolist(), anchor_frame.tolist()):
                st = states_batch[owner]
                if st is None:
                    ve_emb.append(None)
                    ve_state.append(None)
                    continue
                if st.ndim == 2:
                    st = st[anchor]
                ve_emb.append(embs_batch[owner])
                ve_state.append(st.to(device=packed_tokens.device))
            expert_state = (ve_emb, ve_state)

        loss_dict_out = self.model.forward(
            prefix_embeds,
            seq_lens,
            actions=actions,
            robot_indices=robot_indices,
            cs_per_sample=cs_per_sample,
            expert_state=expert_state,
            group_index=batch["group_index"],
            frame_pos=batch["frame_pos"],
            action_dim_mask=pad_mask,
        )

        loss_dict: dict[str, Any] = {}
        flow_losses = loss_dict_out["flow_loss"] * pad_mask.to(loss_dict_out["flow_loss"].dtype)
        denom_per_dim = pad_mask.sum(dim=0).clamp_min(1)
        # Keep on GPU: .cpu()/.item() here forces a CUDA sync every step.
        loss_dict["loss_per_dim"] = (flow_losses.sum(dim=0) / denom_per_dim).detach()
        if self.config.flow_loss_dof_balanced:
            B_e = cs_per_sample.shape[0]
            virt_idx = torch.arange(B_e, device=flow_losses.device).repeat_interleave(cs_per_sample)
            per_sample_num = torch.zeros(B_e, device=flow_losses.device, dtype=flow_losses.dtype)
            per_sample_num.scatter_add_(0, virt_idx, flow_losses.sum(dim=-1))
            per_sample_den = torch.zeros(B_e, device=flow_losses.device, dtype=flow_losses.dtype)
            per_sample_den.scatter_add_(0, virt_idx, pad_mask.sum(dim=-1).to(flow_losses.dtype))
            per_sample = per_sample_num / per_sample_den.clamp_min(1)
            flow_loss = per_sample.mean()
        else:
            denom = pad_mask.sum().clamp_min(1)
            flow_loss = flow_losses.sum() / denom
        loss_dict["flow_loss"] = flow_loss.detach()

        total_loss = flow_loss

        # Every state MLP adds a zero term, so the DDP unused-parameter set is the same on all
        # ranks.
        state_mlps = self.model.vlm_with_expert.state_mlps
        if state_mlps is not None:
            for mlp in state_mlps.values():
                total_loss = total_loss + 0 * sum(p.sum() for p in mlp.parameters())

        loss_dict["loss"] = total_loss.detach()
        return total_loss, loss_dict


class RekiFlowModel(nn.Module):
    """The VLM with action expert, plus the fp32 action/time projections and the optional
    proprio AdaLN projection."""

    def __init__(self, config: RekiConfig, load_vlm_weights: bool = True):
        super().__init__()
        self.config = config

        self.dtype = getattr(torch, config.dtype)
        self._prefill_graphs: Any = None  # set by enable_prefill_graphs (inference only)
        self._expert_graphs: Any = None  # set by compile_expert (inference only)
        # Inference: the expert's CA source is padded to a multiple of this many rows. All
        # prompts whose kept prefill length (text + image tokens) is in the same bucket use one
        # compiled expert. Set it above the longest source to get one shape.
        self.ca_source_bucket = 256

        # Imported here: the module needs the CUDA kernels, and this file must import without.
        from reki.vlm.qwen3_5.modeling import Qwen3_5VLMWithExpert

        check_model_id(config.vlm_model_id)
        self.vlm_with_expert: Qwen3_5VLMWithExpert = Qwen3_5VLMWithExpert.from_pretrained(
            config.vlm_model_id,
            dtype=self.dtype,
            action_expert_hidden_size=config.action_expert_hidden_size,
            action_expert_intermediate_size=config.action_expert_intermediate_size,
            action_expert_num_layers=config.action_expert_num_layers,
            action_expert_num_heads=config.action_expert_num_heads,
            train_vision_blocks=config.train_vision_blocks,
            train_expert_only=config.train_expert_only,
            adarms_cond_dim=config.action_expert_hidden_size,
            action_expert_num_registers=config.action_expert_num_registers,
            load_vlm_weights=load_vlm_weights,
        )
        width = config.action_expert_hidden_size
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_out_proj = nn.Linear(width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

        self.state_adaln_proj: nn.Linear | None = None
        if config.state_encoding == "adaln" and config.state_mlp_embodiments:
            self.vlm_with_expert.init_state_mlps(
                config.state_mlp_embodiments, hidden=config.action_expert_hidden_size
            )
            proj = nn.Linear(config.action_expert_hidden_size, config.action_expert_hidden_size)
            # Zero init: no perturbation of a warm-started expert at step 0. A sample with its
            # state dropped keeps only the bias (a learned null).
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.state_adaln_proj = proj

        self.to(
            device=config.device,
            dtype=self.dtype,
        )

        # Action heads and time MLP stay fp32 (see _proj_fp32).
        self.action_in_proj.to(dtype=torch.float32)
        self.action_out_proj.to(dtype=torch.float32)
        self.time_mlp_in.to(dtype=torch.float32)
        self.time_mlp_out.to(dtype=torch.float32)
        if self.state_adaln_proj is not None:
            self.state_adaln_proj.to(dtype=torch.float32)

    def sample_noise(
        self, shape: tuple[int, ...] | torch.Size, device: torch.device
    ) -> torch.Tensor:
        """Standard-normal fp32 flow noise with the requested shape."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        """Flow-matching time τ ~ Beta(α, β)·scale + offset, drawn on the CPU
        generator then moved."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        actions: torch.Tensor,
        robot_indices: torch.Tensor,
        cs_per_sample: torch.Tensor,
        expert_state: tuple[list[str | None], list[torch.Tensor | None]] | None = None,
        group_index: torch.Tensor,
        frame_pos: torch.Tensor,
        action_dim_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Do a full training forward pass and compute the loss.

        Args:
            prefix_embeds: [1, total_L, D] packed VLM prefix embeddings.
            seq_lens: [B] per-sample token lengths.
            actions: [N, max_action_dim] actions packed along time, one block of
                cs_per_sample rows per chunk (prepare_action unrolls K-frame samples).
            action_dim_mask: [N, max_action_dim] bool, True on the real action dims.
                The padded dims get no noise, so x_t and u_t are exactly 0 there.
            robot_indices: [B_chunks] owning VLM sample of each chunk.
            cs_per_sample: [B_e] action rows per chunk, after the flow_draws tiling.
            expert_state: (embodiment names, states) per chunk owner, for
                ``state_encoding="adaln"``.
            group_index: [B] history group of each VLM sample; register memory stays in a group.
            frame_pos: [B] frame position of each VLM sample in its group; a block attends to the
                registers of its group's frames up to its own.
        """
        B_e = cs_per_sample.shape[0]
        noise = self.sample_noise(actions.shape, actions.device)
        noise = noise * action_dim_mask.to(noise.dtype)

        time = self.sample_time(B_e, actions.device)

        time_per_token = time.repeat_interleave(cs_per_sample)
        time_expanded = time_per_token[:, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        state_cond: torch.Tensor | None = None
        if self.config.state_encoding == "adaln":
            assert expert_state is not None, "state_encoding='adaln' requires expert_state"
            cond_owner = self._state_adaln_cond(*expert_state)
            assert B_e % cond_owner.shape[0] == 0, (
                f"adaln cond: {B_e} action blocks not a multiple of {cond_owner.shape[0]} owners"
            )
            n_draws = B_e // cond_owner.shape[0]
            state_cond = cond_owner.repeat_interleave(n_draws, dim=0) if n_draws > 1 else cond_owner
        suffix_embeds, adarms_cond = self.embed_actions(x_t, time, cs_per_sample, state_cond)

        reg = self.vlm_with_expert.action_expert.registers
        n_draws = cs_per_sample.shape[0] // robot_indices.shape[0]
        suffix_embeds, adarms_cond, cs_expert, expert_is_register, expert_action_row_idx = (
            _prepend_registers(
                suffix_embeds,
                adarms_cond,
                reg,
                cs_per_sample,
                n_draws,
            )
        )

        # The forward state is fresh (embed_vlm_inputs started it); set the expert inputs.
        fwd = self.vlm_with_expert.fwd
        fwd.cs_per_sample = cs_expert
        fwd.expert_is_register = expert_is_register

        gi_own = group_index.to(robot_indices.device)[robot_indices]
        fp_own = frame_pos.to(robot_indices.device)[robot_indices]
        n_draws_reg = cs_expert.shape[0] // robot_indices.shape[0]
        reg_flat_idx, reg_counts = build_register_memory_indices(
            cs_expert,
            fp_own,
            gi_own,
            self.config.action_expert_num_registers,
            n_draws_reg,
        )
        fwd.reg_kv_flat_idx = reg_flat_idx
        fwd.reg_kv_counts = reg_counts
        fwd.expert_frame_pos = fp_own

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            seq_lens=seq_lens,
            inputs_embeds=(prefix_embeds, suffix_embeds),
            adarms_cond=adarms_cond,
            robot_indices=robot_indices,
        )

        losses: dict[str, torch.Tensor] = {}

        assert suffix_out is not None
        suffix_out = suffix_out[expert_action_row_idx]

        with torch.autocast(device_type=suffix_out.device.type, enabled=False):
            v_t = _proj_fp32(self.action_out_proj, suffix_out)
        losses["flow_loss"] = F.mse_loss(u_t.float(), v_t, reduction="none")

        return losses

    def embed_vlm_inputs(
        self,
        vision_inputs: list[dict[str, Any]],
        lang_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Embed padded [B, L] inference tokens and images."""
        return self.vlm_with_expert.embed_vlm_inputs(
            input_ids=lang_tokens,
            vision_inputs=vision_inputs,
            seq_lens=None,
        )

    def _state_adaln_cond(
        self, embodiments: list[str | None], states: list[torch.Tensor | None]
    ) -> torch.Tensor:
        """Per-virtual-sample proprio → AdaLN cond addend ``[B_e, expert_hidden]`` (fp32).

        The per-embodiment MLP runs in its own (backbone) dtype, one token per sample; the
        modulation projection runs in fp32. A sample with no state (dropped by
        ``state_dropout_prob``, or a stateless dataset) pools to zeros and so contributes
        exactly ``state_adaln_proj.bias``, the learned null.
        """
        assert self.state_adaln_proj is not None
        state_mlps = self.vlm_with_expert.state_mlps
        assert state_mlps is not None, "state_encoding='adaln' requires the state MLP bank"
        bank_hidden = self.state_adaln_proj.in_features
        device = next(self.state_adaln_proj.parameters()).device
        pooled: list[torch.Tensor] = []
        for emb, st in zip(embodiments, states):
            if emb is None or st is None:
                pooled.append(torch.zeros(bank_hidden, dtype=torch.float32, device=device))
                continue
            mlp = state_mlps[emb]
            dt = next(mlp.parameters()).dtype
            pooled.append(mlp(st.to(device=device, dtype=dt).reshape(-1)).float())
        stacked = torch.stack(pooled, dim=0)
        with torch.autocast(device_type=stacked.device.type, enabled=False):
            return _proj_fp32(self.state_adaln_proj, stacked)

    def _time_sinusoid(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.config.action_expert_hidden_size
        return create_sinusoidal_pos_embedding(
            timestep.float(),
            hidden,
            min_period=self.config.time_embedding_min_period,
            max_period=self.config.time_embedding_max_period,
            device=timestep.device,  # pyright: ignore[reportArgumentType]
        ).float()

    def _time_mlp(self, time_emb: torch.Tensor) -> torch.Tensor:
        """fp32 time MLP; the caller disables autocast."""
        t = F.silu(_proj_fp32(self.time_mlp_in, time_emb))
        return F.silu(_proj_fp32(self.time_mlp_out, t))

    def embed_actions(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        cs_per_sample: torch.Tensor,
        state_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the noisy actions and the timestep (plus state) conditioning for the expert.

        Inputs are packed: ``noisy_actions [N, max_action_dim]``, ``timestep [B_e]``. With
        ``cs_per_sample``, the conditioning is repeated to ``[N, hidden]``, one row per action
        token for the AdaRMS layers.

        ``state_cond`` [B_e, hidden] (state_encoding="adaln") is added to the time
        conditioning in fp32, before the repeat.
        """
        noisy_actions = noisy_actions.float()
        time_emb = self._time_sinusoid(timestep)

        with torch.autocast(device_type=noisy_actions.device.type, enabled=False):
            action_emb = _proj_fp32(self.action_in_proj, noisy_actions)
            time_emb = self._time_mlp(time_emb)
        if state_cond is not None:
            assert state_cond.shape == time_emb.shape, (
                f"state_cond {tuple(state_cond.shape)} != time cond {tuple(time_emb.shape)}"
            )
            time_emb = time_emb + state_cond.to(time_emb.dtype)
        time_emb = time_emb.repeat_interleave(cs_per_sample, dim=0)
        return action_emb.to(self.dtype), time_emb.to(self.dtype)

    def _prefill(self, vision_inputs: list[dict[str, Any]], tokens: torch.Tensor) -> Any:
        """The VLM pass of one inference step: returns the conditioning stash
        (``CrossAttnCache``) with the final-layer output and its positions."""
        prefix_embs = self.embed_vlm_inputs(vision_inputs, tokens)
        seq_lens_inf = torch.tensor(
            [prefix_embs.shape[1]], device=prefix_embs.device, dtype=torch.int32
        )
        (vlm_out, _), cross_attn_cache = self.vlm_with_expert.forward(
            seq_lens=seq_lens_inf,
            inputs_embeds=(prefix_embs, None),
            prefill_only=True,
        )
        assert cross_attn_cache is not None and vlm_out is not None
        fwd = self.vlm_with_expert.fwd
        assert fwd.position_ids is not None
        cross_attn_cache.vlm_out = vlm_out.detach()
        cross_attn_cache.vlm_position_ids = fwd.position_ids[:, 0, :].detach().clone()
        return cross_attn_cache

    def enable_prefill_graphs(self) -> None:
        """Replay the inference prefill as one CUDA graph per input layout (bitwise equal to
        the eager prefill; see ``reki/vlm/qwen3_5/prefill_graph.py``)."""
        from reki.vlm.qwen3_5.prefill_graph import PrefillGraphs

        self._prefill_graphs = PrefillGraphs()

    def clear_cuda_graphs(self) -> None:
        """Drop the captured prefill and expert graphs. The next predict captures them again.
        The graphs use the weights and buffers at their capture-time addresses. ``_apply`` and
        ``_load_from_state_dict`` call this method. Call it yourself after you replace a
        submodule or call ``.to()`` on a submodule only."""
        for graphs in (self._prefill_graphs, self._expert_graphs):
            if graphs is not None:
                graphs.clear()

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> RekiFlowModel:
        self.clear_cuda_graphs()
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self.clear_cuda_graphs()
        super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def sample_actions(
        self,
        vision_inputs: list[dict[str, Any]],
        tokens: torch.Tensor,
        action_dim: int,
        register_cache: Any,
        expert_state: tuple[list[str | None], list[torch.Tensor | None]] | None = None,
    ) -> torch.Tensor:
        """Sample an action chunk ``[B, chunk_size, max_action_dim]`` by Euler integration.

        ``action_dim`` is the number of real action dims: the padded dims ``[action_dim:]``
        are zero in training (no noise, zero target), so the sampler keeps them at zero.

        Reruns the VLM prefill, stores its final-layer output as the expert's cross-attention
        source, and performs Euler integration while carrying register memory across calls.
        """

        num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        noise = self.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device
        )
        noise = self._zero_pad_dims(noise, action_dim)

        if self._prefill_graphs is not None:
            cross_attn_cache = self._prefill_graphs(self, vision_inputs, tokens)
        else:
            cross_attn_cache = self._prefill(vision_inputs, tokens)
        self.vlm_with_expert.gather_cross_attn_source(cross_attn_cache, self.ca_source_bucket)

        state_cond: torch.Tensor | None = None
        if self.config.state_encoding == "adaln":
            assert expert_state is not None, "state_encoding='adaln' requires expert_state"
            state_cond = self._state_adaln_cond(*expert_state)

        args = (noise, cross_attn_cache, register_cache, state_cond, action_dim, num_steps)
        if self._expert_graphs is not None:
            x_t, cur_k, cur_v = self._expert_graphs(self, *args)
        else:
            x_t, cur_k, cur_v = self._expert_actions(*args)
        register_cache.append_frame(cur_k, cur_v)
        return x_t

    def compile_expert(self) -> None:
        """Compile the inference expert. The shapes are the same for every register-cache fill
        level and prompt length.

        Compile the step modulations (dynamic in the step count), Euler step 1 and steps 2..N
        separately (Inductor, no CUDA graphs). The Euler loop stays in Python
        (``_expert_actions``), so one compile serves every step count. ``ExpertGraphs`` then
        captures the whole expert of one predict as one CUDA graph per step count."""
        from reki.vlm.qwen3_5.expert_graph import ExpertGraphs

        self._step_modulations = torch.compile(  # type: ignore[method-assign]
            self._step_modulations, dynamic=True
        )
        for name in ("_expert_first_step", "_expert_step"):
            setattr(self, name, torch.compile(getattr(self, name), dynamic=False))
        self._expert_graphs = ExpertGraphs()

    def _expert_actions(
        self,
        noise: torch.Tensor,
        cross_attn_cache: Any,
        register_cache: Any,
        state_cond: torch.Tensor | None,
        action_dim: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert constants and the Euler loop of one inference step. Leaves the register
        cache untouched and returns the integrated actions plus this frame's register k / v.
        Each callable here is one graph under ``compile_expert``; all shapes are independent
        of the cache fill level, the prompt and the step count."""
        assert noise.shape[0] == 1, "register-memory inference runs one sample"
        mods, dt = self._step_modulations(num_steps, state_cond)
        x_t, cur_k, cur_v = self._expert_first_step(
            noise, cross_attn_cache, register_cache, mods[:1], dt, action_dim
        )
        for step in range(1, num_steps):
            x_t = self._expert_step(x_t, cross_attn_cache, mods[step : step + 1], dt, action_dim)
        return x_t, cur_k, cur_v

    def _expert_first_step(
        self,
        noise: torch.Tensor,
        cross_attn_cache: Any,
        register_cache: Any,
        mod: torch.Tensor,
        dt: torch.Tensor,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the step constants (``register_prepare``) and Euler step 1. Step 1 also runs
        the register rows of the current frame. Return x_t after the step and the pre-RoPE
        register k / v of this frame. Store on ``cross_attn_cache`` the data that steps 2..N
        read (the constants and the memory K/V). ``mod`` is the ``[1, ...]`` modulation row of
        the step, ``dt`` the 0-d step size."""
        vwe = self.vlm_with_expert
        T = noise.shape[1]
        vwe.register_prepare(cross_attn_cache, register_cache, T)
        x_t = self._zero_pad_dims(noise.float(), action_dim)
        out, cur_k, cur_v, mem_kv = vwe.denoise_first_step_register(
            self._action_embs(x_t),
            cross_attn_cache,
            vwe.split_modulation(mod),
            register_cache.k,
            register_cache.v,
        )
        cross_attn_cache.register_step["mem_kv"] = mem_kv
        return x_t + dt * self._velocity(out, 1, T), cur_k, cur_v

    def _expert_step(
        self,
        x_t: torch.Tensor,
        cross_attn_cache: Any,
        mod: torch.Tensor,
        dt: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        """Euler step 2..N: actions only, against the constants and memory K/V step 1 left on
        ``cross_attn_cache``."""
        vwe = self.vlm_with_expert
        x_t = self._zero_pad_dims(x_t, action_dim)
        out = vwe.denoise_forward_register(
            self._action_embs(x_t),
            cross_attn_cache,
            vwe.split_modulation(mod),
        )
        return x_t + dt * self._velocity(out, 1, x_t.shape[1])

    def _step_modulations(
        self, num_steps: int, state_cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the AdaRMS modulations of all ``num_steps`` Euler steps (one row per step, see
        ``expert_modulations``) and the step size ``dt`` as a 0-d fp32 tensor. The time MLP
        and each norm's dense layer run once on all steps. The result equals a per-step
        computation up to GEMM rounding."""
        device = self.time_mlp_in.weight.device
        dt = -1.0 / num_steps
        times = 1.0 + torch.arange(num_steps, dtype=torch.float64, device=device) * dt
        with torch.autocast(device_type=device.type, enabled=False):
            cond = self._time_mlp(self._time_sinusoid(times))
        if state_cond is not None:
            cond = cond + state_cond.to(cond.dtype)
        mods = self.vlm_with_expert.expert_modulations(cond.to(self.dtype))
        return mods, torch.full((), dt, dtype=torch.float32, device=device)

    @staticmethod
    def _zero_pad_dims(x: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Return a copy of ``x`` with the padded action dims ``[action_dim:]`` set to 0. This is
        their training value τ·noise, because the padding noise is 0."""
        x = x.clone()
        x[..., action_dim:] = 0.0
        return x

    def _action_embs(self, x_t: torch.Tensor) -> torch.Tensor:
        """Packed expert action rows ``[B*T, hidden]`` for one Euler step (``embed_actions``'s
        action branch)."""
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            a = _proj_fp32(self.action_in_proj, x_t.reshape(-1, x_t.shape[-1]).float())
        return a.to(self.dtype)

    def _velocity(self, expert_out: torch.Tensor, B: int, T: int) -> torch.Tensor:
        expert_out = expert_out[-B * T :]
        with torch.autocast(device_type=expert_out.device.type, enabled=False):
            return _proj_fp32(self.action_out_proj, expert_out).reshape(B, T, -1)
