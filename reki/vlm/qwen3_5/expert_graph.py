"""CUDA-graph replay of the inference expert: step modulations, Euler step 1 and steps 2..N.

``RekiFlowModel.compile_expert`` compiles the per-step callables without CUDA graphs. This
module captures the whole expert of one predict (``RekiFlowModel._expert_actions``, with the
Euler loop in Python) into one ``torch.cuda.CUDAGraph`` per step count and replays it. No shape
depends on the register-cache fill level (device data) or on the prompt (the CA source is
padded to a bucket), so one graph serves a whole evaluation. Each replay first copies the
expert inputs into static buffers. The graph reads the register cache in place, at fixed
addresses, so each cache object gets its own graph. The normal path keeps one cache per policy
and resets it in place. Capture checks that the replay is bitwise equal to the captured run.
"""

from __future__ import annotations

from typing import Any

import torch

from .cuda_graph import GraphLRU, capture_checked
from .inference import CrossAttnCache

# The CrossAttnCache fields the expert reads (set by ``gather_cross_attn_source``).
_PAST_INPUTS = ("ca_src", "ca_src_pos", "ca_cu", "ca_start")


class _StepGraph:
    def __init__(
        self,
        model: Any,
        noise: torch.Tensor,
        past: CrossAttnCache,
        register_cache: Any,
        state_cond: torch.Tensor | None,
        action_dim: int,
        num_steps: int,
    ) -> None:
        self.noise = noise.clone()
        self.state_cond = None if state_cond is None else state_cond.clone()
        self.past = CrossAttnCache()
        for name in _PAST_INPUTS:
            setattr(self.past, name, getattr(past, name).clone())
        self.args = (model, register_cache, action_dim, num_steps)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model, register_cache, action_dim, num_steps = self.args
        return model._expert_actions(
            self.noise, self.past, register_cache, self.state_cond, action_dim, num_steps
        )

    def capture(self, reference: tuple[torch.Tensor, ...]) -> None:
        self.graph, self.out = capture_checked(self._forward, reference, "expert")

    def replay(
        self, noise: torch.Tensor, past: CrossAttnCache, state_cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.graph is not None and self.out is not None
        self.noise.copy_(noise)
        if state_cond is not None:
            assert self.state_cond is not None
            self.state_cond.copy_(state_cond)
        for name in _PAST_INPUTS:
            getattr(self.past, name).copy_(getattr(past, name))
        self.graph.replay()
        x_t, cur_k, cur_v = self.out
        # The outputs are overwritten by the next replay: x_t is returned to the caller, cur_k /
        # cur_v are copied into the register cache before then.
        return x_t.clone(), cur_k, cur_v


class ExpertGraphs(GraphLRU[_StepGraph]):
    """One captured expert per input signature (step count, register cache, shapes, dtypes),
    for the ``max_graphs`` most recently used. The first predict of a signature runs the (compiled)
    expert directly and captures it; later ones replay. Capture costs that first predict
    three extra expert runs (two warm-up runs and the check replay)."""

    def __init__(self, max_graphs: int = 4) -> None:
        super().__init__(max_graphs)

    def __call__(
        self,
        model: Any,
        noise: torch.Tensor,
        past: CrossAttnCache,
        register_cache: Any,
        state_cond: torch.Tensor | None,
        action_dim: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = [noise, state_cond, *(getattr(past, name) for name in _PAST_INPUTS)]
        # The graph reads the captured register cache's buffers in place, so another cache
        # object needs its own graph. Keyed on the buffer addresses: a cache reset in place
        # keeps them, and a captured graph keeps its cache alive, so they are never reused.
        cache = tuple(
            t.data_ptr() for t in (register_cache.k, register_cache.v, register_cache.n_valid)
        )
        key = (num_steps, action_dim, cache) + tuple(
            None if t is None else (tuple(t.shape), t.dtype) for t in tensors
        )
        entry = self.get(key)
        if entry is not None:
            return entry.replay(noise, past, state_cond)
        self.make_room()
        out = model._expert_actions(noise, past, register_cache, state_cond, action_dim, num_steps)
        entry = _StepGraph(model, noise, past, register_cache, state_cond, action_dim, num_steps)
        entry.capture(out)
        self.put(key, entry)
        return out
