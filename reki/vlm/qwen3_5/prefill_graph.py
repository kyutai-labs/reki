"""CUDA-graph replay of the single-sample inference prefill: vision tower → image rows → LM.

The prefill launches about two thousand small kernels and syncs the host several times
(vision attention split sizes, FLA chunk indices, MRoPE index checks). A normal eager prefill
builds the inputs that depend only on the input layout (token ids, image grid), once per
layout. The pixel-dependent kernels are then captured into one CUDA graph and
replayed. The graph runs the same kernels in the same order as the eager path. Capture checks
that its output is bitwise equal to the eager output.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..common import cu_single
from .cuda_graph import GraphLRU, capture_checked
from .hf_patches import fla_inference_configs
from .inference import CrossAttnCache


class _LayoutGraph:
    def __init__(self, model: Any, vision: dict[str, Any], tokens: torch.Tensor, past: Any):
        vwe = model.vlm_with_expert
        self.vwe = vwe
        visual = vwe.image_encoder
        device = tokens.device
        grid = vision["image_grid_thw"].cpu()
        # The vision tower's layout-only inputs, exactly as Qwen3_5VisionModel.forward makes
        # them. cu_seqlens is a device tensor held here: the varlen vision attention caches its
        # device copy of the last cu_seqlens it saw, which a later layout would free.
        self.vis_pos = visual.fast_pos_embed_interpolate(grid)
        rot = visual.rot_pos_emb(grid)
        emb = torch.cat((rot, rot), dim=-1)
        self.vis_pe = (emb.cos(), emb.sin())
        self.vis_cu = F.pad(
            torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            ),
            (1, 0),
            value=0,
        ).to(device)
        self.tokens = tokens.clone()
        image_token = vwe.mm_model.config.image_token_id
        self.image_rows = (tokens[0] == image_token).nonzero().squeeze(1)
        assert vwe.fwd.position_ids is not None
        self.position_ids = vwe.fwd.position_ids.clone()
        self.L = tokens.shape[1]
        self.expert_start = past.expert_start
        self.vlm_position_ids = past.vlm_position_ids
        self.pixels = vision["pixel_values"].to(device=device, dtype=vwe.vlm.dtype).clone()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: torch.Tensor | None = None
        self.device = device

    def _forward(self) -> torch.Tensor:
        """The eager prefill's kernels for this layout (``embed_vlm_inputs`` + the prefill branch
        of ``Qwen3_5VLMWithExpert.forward`` + the stock text-model loop)."""
        vwe = self.vwe
        visual = vwe.image_encoder
        h = visual.patch_embed(self.pixels) + self.vis_pos
        for block in visual.blocks:
            h = block(h, cu_seqlens=self.vis_cu, position_embeddings=self.vis_pe)
        image_features = visual.merger(h)
        prefix = vwe.embed_language_tokens(self.tokens)
        prefix[0, self.image_rows] = image_features
        text = vwe.text_backbone
        L = self.L
        # Built inside the captured region, so FLA's last-call cache keys off fresh tensors
        # and its chunk indices live in the graph's memory pool.
        cu = cu_single(L, self.device)
        cu_cpu = torch.tensor([0, L], dtype=torch.long)
        cache_position = torch.arange(L, device=self.device)
        position_embeddings = text.rotary_emb(prefix, self.position_ids)
        h = prefix
        for layer in text.layers[: text.config.num_hidden_layers]:
            h = layer(
                h,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=self.position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=L,
                max_length_k=L,
                gdn_cu_seqlens=cu.long(),
                gdn_cu_seqlens_cpu=cu_cpu,
            )
        return text.norm(h)

    def capture(self, reference: torch.Tensor) -> None:
        with fla_inference_configs():  # same FLA configs as the eager prefill
            self.graph, self.out = capture_checked(self._forward, (reference,), "prefill")

    def replay(self, pixel_values: torch.Tensor) -> CrossAttnCache:
        assert self.graph is not None and self.out is not None
        self.pixels.copy_(pixel_values)
        self.graph.replay()
        past = CrossAttnCache()
        past.expert_start = self.expert_start
        past.vlm_out = self.out
        past.vlm_position_ids = self.vlm_position_ids
        return past


class PrefillGraphs(GraphLRU[_LayoutGraph]):
    """One captured prefill per input layout, for the ``max_layouts`` most recently used
    layouts. The layout includes the token ids of the task text, so the limit stops an
    evaluation over many tasks from keeping one graph and memory pool per task. The first
    predict of a layout runs eagerly. Later predicts replay the graph. Capture adds about
    three prefills to that first predict."""

    def __init__(self, max_layouts: int = 2) -> None:
        super().__init__(max_layouts)

    def __call__(
        self,
        model: Any,
        vision_inputs: list[dict[str, Any]],
        tokens: torch.Tensor,
    ) -> CrossAttnCache:
        assert len(vision_inputs) == 1 and tokens.shape[0] == 1
        vision = vision_inputs[0]
        # Small host copies at the start of the predict, while the GPU is idle.
        key = tuple(
            t.cpu().numpy().tobytes()
            for t in (tokens, vision["image_grid_thw"], vision["mm_token_type_ids"])
        ) + (str(tuple(vision["pixel_values"].shape)).encode(),)
        layout = self.get(key)
        if layout is not None:
            return layout.replay(vision["pixel_values"])
        self.make_room()
        past = model._prefill(vision_inputs, tokens)
        layout = _LayoutGraph(model, vision, tokens, past)
        layout.capture(past.vlm_out)
        self.put(key, layout)
        return past
