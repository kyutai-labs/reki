"""Packing helpers and the per-episode register cache of the action expert."""

from __future__ import annotations

import torch


def cu_seqlens(counts: torch.Tensor) -> torch.Tensor:
    """Cumulative sequence lengths for flash_attn_varlen / FLA kernels."""
    return torch.cat(
        [torch.zeros(1, device=counts.device, dtype=torch.int32), counts.cumsum(0)]
    ).to(torch.int32)


def cu_single(n: int, device: torch.device) -> torch.Tensor:
    """int32 cu_seqlens ``[0, n]`` for one varlen sample, built on device:
    ``torch.tensor([0, n], device=cuda)`` is a pageable H2D copy that blocks the host."""
    return torch.arange(0, 2 * n, n, dtype=torch.int32, device=device)


def cu_seqlens_from(lens: list[int], device: torch.device) -> torch.Tensor:
    """int32 cu_seqlens ``[0, l0, l0 + l1, ...]`` for host-known lengths, built on device."""
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    for i, n in enumerate(lens):
        cu[i + 1] = n
    return cu.cumsum(0, dtype=torch.int32)


def register_shared_row_offsets(
    cs_per_sample: torch.Tensor, n_reg: int, n_draws: int
) -> torch.Tensor:
    """Within-block position offsets for draw-shared register blocks.

    Owner block = [R registers ∥ draw_0 … draw_{k-1} actions]: registers get 0..R-1 once;
    every draw's action rows restart at R, the positions each draw would have with its own
    copy of the registers."""
    device = cs_per_sample.device
    cs_long = cs_per_sample.long()
    N = int(cs_long.sum().item())
    cu_e = torch.zeros(cs_long.shape[0] + 1, dtype=torch.long, device=device)
    cu_e[1:] = cs_long.cumsum(0)
    off = torch.arange(N, device=device) - cu_e[:-1].repeat_interleave(cs_long)
    cs_act = ((cs_long - n_reg) // n_draws).clamp_min(1).repeat_interleave(cs_long)
    return torch.where(off < n_reg, off, n_reg + (off - n_reg) % cs_act)


class RegisterCache:
    """Per-episode buffer of the expert's register keys/values, for streaming inference.

    Holds, per expert layer, the register ``k``/``v`` of the last ``capacity`` previous frames
    (R rows per frame), with ``k`` stored before RoPE, in one fixed-size buffer
    ``[num_layers, capacity * R, H, D]`` per tensor. The buffer is right-aligned: the
    ``n_frames`` valid frames are its last ``n_frames * R`` rows, oldest first; the rows before
    them are stale and never read. Self-attention positions are window-local MRoPE, so the
    cached ``k`` is re-rotated to the current window-local frame index each env step (see
    ``Qwen3_5ExpertInference.register_prepare``); ``v`` is not rotated.

    The fill level is device data: ``n_valid`` is a device copy of ``n_frames``. The buffers
    and ``n_valid`` are updated in place and keep their addresses across episodes. So one
    compiled or CUDA-graphed expert serves every fill level with no host sync.

    ``capacity = history_k - 1``: with the current frame, a step sees at most history_k frames.
    The current frame is appended after the Euler loop; a full buffer drops its oldest frame.
    """

    def __init__(
        self,
        num_layers: int,
        n_reg: int,
        capacity: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.n_reg = n_reg
        self.capacity = capacity
        shape = (num_layers, capacity * n_reg, n_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.n_valid = torch.zeros((), dtype=torch.int32, device=device)
        self.n_frames = 0
        for t in (self.k, self.v, self.n_valid):
            torch._dynamo.mark_static_address(t)

    def reset(self) -> None:
        """Clear the valid-frame count while retaining the allocated buffers."""
        self.n_frames = 0
        self.n_valid.zero_()

    def append_frame(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append the just-processed frame's pre-RoPE register k/v ``[num_layers, n_reg, H, D]``
        at the end of the buffer, shifting every row one frame towards the front (the oldest
        frame falls off once the buffer is full). Copies: a CUDA-graph output is overwritten by
        the next replay."""
        assert k_new.dtype == self.k.dtype and v_new.dtype == self.v.dtype
        if self.capacity == 0:
            return
        self.k.copy_(torch.cat([self.k[:, self.n_reg :], k_new], dim=1))
        self.v.copy_(torch.cat([self.v[:, self.n_reg :], v_new], dim=1))
        self.n_frames = min(self.n_frames + 1, self.capacity)
        self.n_valid.fill_(self.n_frames)
