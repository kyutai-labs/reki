# smart_resize adapted from Transformers v5.3.0,
# src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py. Copyright 2024 The Qwen team,
# Alibaba Group and the HuggingFace Inc. team. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). See NOTICE.
"""Tensor-native replacement for the image path of HF's Qwen3.5 processor
(``Qwen3VLProcessor``), without HF's per-call validation and ``BatchFeature`` overhead. The
tokenizer and chat template stay HF's. Inputs are uint8 ``(3, H, W)`` tensors
(``lerobot_patches`` keeps decoded frames uint8) or PIL RGB images; float tensors in [0, 1] are
converted to uint8. Resize and normalization are HF's (``smart_resize``, bicubic+antialias,
mean/std). ``from_processor`` emits single-frame patches (``temporal_patch_size=1``): HF's
``t=0`` slice to fp tolerance, half its width, for the folded ``LinearPatchEmbed``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen's aspect-ratio-preserving resize target (factor-aligned, in pixel budget)."""
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def _to_uint8_chw(img: Any) -> torch.Tensor:
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return tvF.pil_to_tensor(img)
    assert isinstance(img, torch.Tensor), f"unsupported image type: {type(img)}"
    if img.dtype == torch.uint8:
        return img
    return (img.to(torch.float32).clamp(0.0, 1.0) * 255.0).to(torch.uint8)


class FastQwenImageProcessor:
    """Tensor-native Qwen3.5 image processing.

    Direct construction defaults to HF's two-frame temporal layout; ``from_processor`` uses
    single-frame patches for the folded ``LinearPatchEmbed``.
    """

    def __init__(
        self,
        min_pixels: int,
        max_pixels: int,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        mean: tuple[float, float, float] = OPENAI_CLIP_MEAN,
        std: tuple[float, float, float] = OPENAI_CLIP_STD,
    ) -> None:
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self._mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    @classmethod
    def from_processor(cls, processor: Any) -> FastQwenImageProcessor:
        """The single-frame processor corresponding to an HF Qwen3.5 processor."""
        ip = processor.image_processor
        return cls(
            min_pixels=ip.size["shortest_edge"],
            max_pixels=ip.size["longest_edge"],
            patch_size=ip.patch_size,
            temporal_patch_size=1,
            merge_size=ip.merge_size,
            mean=tuple(ip.image_mean),
            std=tuple(ip.image_std),
        )

    def __call__(self, images: list[Any]) -> dict[str, torch.Tensor]:
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        factor = p * m

        flat_patches: list[torch.Tensor] = []
        grids: list[list[int]] = []
        for img in images:
            img = _to_uint8_chw(img)
            _, h, w = img.shape
            rh, rw = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)
            x = tvF.resize(
                img,
                [rh, rw],
                interpolation=tvF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            x = (x.to(torch.float32) * (1.0 / 255.0) - self._mean) / self._std
            C = x.shape[0]
            gh, gw = rh // p, rw // p
            x = x.unsqueeze(0).expand(t, C, rh, rw).contiguous()
            x = x.view(1, 1, t, C, gh // m, m, p, gw // m, m, p)
            x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
            x = x.reshape(gh * gw, C * t * p * p)
            flat_patches.append(x)
            grids.append([1, gh, gw])

        pixel_values = torch.cat(flat_patches, dim=0)
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
