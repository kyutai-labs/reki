"""Token item of one robot frame: the VLM prompt (images + text)."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

SPATIAL_MERGE_TOKENS = 4
"""Vision patches per vision token: Qwen3.5 merges 2x2 patches."""


def count_vision_tokens(vision_inputs: dict[str, Any]) -> int:
    """Vision tokens of one sample, for the image-token budget of a batch: the
    ``image_grid_thw`` patches over the spatial merge."""
    thw = vision_inputs.get("image_grid_thw")
    if thw is not None:
        return int(thw.prod(dim=-1).sum().item() // SPATIAL_MERGE_TOKENS)
    return 0


class FrameTokenizer:
    """Builds the token item of one robot frame with the VLM preprocessor."""

    def __init__(self, preprocessor: Any, *, use_camera_role_tokens: bool) -> None:
        self._preprocessor = preprocessor
        self._use_camera_role_tokens = use_camera_role_tokens

    def tokenize(
        self,
        item: dict[str, Any],
        images: list[torch.Tensor],
        camera_indices: list[int],
        prompt: str,
    ) -> None:
        """Set the language tokens, ``vision_inputs`` and ``n_image_tokens`` of ``item``."""
        preprocessor = self._preprocessor
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        if self._use_camera_role_tokens:
            from reki.vlm.qwen3_5.preprocessor import role_token_id_for_camera_index

            cam_role_ids = [role_token_id_for_camera_index(c) for c in camera_indices]
            sample = preprocessor.preprocess(
                messages=conversation, camera_role_token_ids=cam_role_ids
            )
        else:
            sample = preprocessor.preprocess(messages=conversation)

        lang_tokens = sample["input_ids"]
        vision_inputs = sample["vision_inputs"]
        item[OBS_LANGUAGE_TOKENS] = lang_tokens

        item["vision_inputs"] = vision_inputs
        item["n_image_tokens"] = count_vision_tokens(vision_inputs)
