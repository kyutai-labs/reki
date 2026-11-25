"""Token item of one robot frame: the VLM prompt (images + text), the optional subtask turn,
labels and region tags."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

from reki.policy.constants import IGNORE_INDEX, RegionTag

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
    """Builds the token item of one robot frame with the VLM preprocessor.

    The subtask is its own turn (subtask header, text, end of turn), in the assistant turn or,
    with ``subtask_in_user_stream``, in the user turn.
    """

    def __init__(
        self,
        preprocessor: Any,
        *,
        subtask_in_user_stream: bool,
        use_camera_role_tokens: bool,
    ) -> None:
        self._preprocessor = preprocessor
        self._subtask_in_user_stream = subtask_in_user_stream
        self._use_camera_role_tokens = use_camera_role_tokens
        self._asst_end_ids = list(preprocessor.asst_end_tokens)
        self._subtask_open_ids = list(preprocessor.subtask_header_ids)

    def tokenize(
        self,
        item: dict[str, Any],
        images: list[torch.Tensor],
        camera_indices: list[int],
        prompt: str,
        subtask_text: str | None,
    ) -> None:
        """Set the language tokens, ``region_tags``, ``labels`` (with a subtask),
        ``vision_inputs`` and ``n_image_tokens`` of ``item``."""
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
        prompt_tags = torch.full((lang_tokens.shape[0],), RegionTag.USER, dtype=torch.int8)

        if subtask_text is not None:
            suffix_ids: list[int] = []
            suffix_tags: list[int] = []
            for ids, tag in self._subtask_turn(subtask_text):
                suffix_ids.extend(ids)
                suffix_tags.extend([tag] * len(ids))
            item[OBS_LANGUAGE_TOKENS] = torch.cat(
                [lang_tokens, torch.tensor(suffix_ids, dtype=torch.long)]
            )
            region_tags = torch.cat([prompt_tags, torch.tensor(suffix_tags, dtype=torch.int8)])
            labels = torch.full_like(item[OBS_LANGUAGE_TOKENS], IGNORE_INDEX)
            sup = (
                (region_tags != RegionTag.USER)
                & (region_tags != RegionTag.ASST_HEADER)
                & (region_tags != RegionTag.USER_SUBTASK_HEADER)
            )
            labels[sup] = item[OBS_LANGUAGE_TOKENS][sup]
            item["labels"] = labels
            item["region_tags"] = region_tags
            mm_ids = vision_inputs["mm_token_type_ids"]
            pad = mm_ids.new_zeros((*mm_ids.shape[:-1], len(suffix_ids)))
            vision_inputs["mm_token_type_ids"] = torch.cat([mm_ids, pad], dim=-1)
        else:
            item[OBS_LANGUAGE_TOKENS] = lang_tokens
            item["region_tags"] = prompt_tags

        item["vision_inputs"] = vision_inputs
        item["n_image_tokens"] = count_vision_tokens(vision_inputs)

    def _subtask_turn(self, subtask_text: str) -> list[tuple[list[int], RegionTag]]:
        """(token ids, region tag) of each part of the subtask turn."""
        tokenizer = self._preprocessor.tokenizer
        text_ids = tokenizer(subtask_text, add_special_tokens=False)["input_ids"]
        if self._subtask_in_user_stream:
            header, text = RegionTag.USER_SUBTASK_HEADER, RegionTag.USER_SUBTASK
        else:
            header, text = RegionTag.ASST_HEADER, RegionTag.SUBTASK
        return [(self._subtask_open_ids, header), (text_ids, text), (self._asst_end_ids, text)]
