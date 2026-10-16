"""Qwen3_5Preprocessor: the tokenizer and chat template of the Qwen3.5 processor, with images
through ``FastQwenImageProcessor`` (single-frame patches for the folded vision patch embed).

The chat template puts one image placeholder (`<|image_pad|>` = token 248056) per image in
the token stream. Each placeholder is expanded to the token count of its vision-tower grid.
The image positions are the placeholder positions in the tokens.

``vision_inputs`` payload::
    {
        "pixel_values":   Tensor,              # flattened patches for vision tower
        "image_grid_thw": Tensor [N, 3],       # per-image (t, h, w) grid
        "mm_token_type_ids": Tensor [1, L],    # 0=text, 1=image, 2=video (used by get_rope_index)
    }
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor

from reki.data.robotics.cameras import camera_role

from . import check_model_id
from .image_processor import FastQwenImageProcessor

CAMERA_ROLE_TOKEN_IDS: dict[str, int] = {
    "overhead": 248080,
    "side": 248081,
    "wrist_left": 248082,
    "wrist_right": 248083,
}
"""The camera-role token of each camera role. These are spare rows of the Qwen3.5 embedding
table (248320 rows) past the tokenizer vocabulary (248077 tokens), so no text maps to them and
the tokenizer has no name for them. The preprocessor inserts one before each image's
``<|vision_start|>``; ``RekiModel`` initializes them from the ``<|vision_start|>`` embedding."""


def role_token_id_for_camera_index(idx: int) -> int | None:
    """Map a canonical camera index (``cameras.CAMERA_KEYS``) to its camera-role token ID. A
    camera without a role (depth) gets no marker."""
    role = camera_role(idx)
    return CAMERA_ROLE_TOKEN_IDS[role] if role else None


def _content_key(content: Any) -> Any:
    if isinstance(content, str):
        return content
    return [
        (item.get("type"), item.get("text")) if isinstance(item, dict) else item for item in content
    ]


class Qwen3_5Preprocessor:
    """Chat messages with images → ``input_ids`` and the ``vision_inputs`` of
    ``Qwen3_5VLMWithExpert.embed_vlm_inputs`` (see the module docstring)."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.tokenizer = processor.tokenizer
        self._fast_img = FastQwenImageProcessor.from_processor(processor)

        tok = self.tokenizer
        if min(CAMERA_ROLE_TOKEN_IDS.values()) < len(tok):
            raise ValueError(
                f"camera-role token IDs {sorted(CAMERA_ROLE_TOKEN_IDS.values())} collide with "
                f"the tokenizer vocabulary ({len(tok)} tokens)"
            )
        self.image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        self.video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
        self.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self._image_tok_str = "<|image_pad|>"
        # Opt-in (inference) memo of token ids. The ids are the same for every step of an
        # episode.
        self.token_cache: dict[tuple[Any, ...], torch.Tensor] | None = None

    @classmethod
    def from_pretrained(
        cls, vlm_model_id: str, image_size: int | None = None
    ) -> Qwen3_5Preprocessor:
        """The preprocessor of ``vlm_model_id``. The Qwen3.5 image processor budgets pixel
        area and ignores ``image_size``, so ``image_size`` sets ``max_pixels =
        image_size**2``: larger images are scaled down to that area, smaller ones keep their
        resolution."""
        check_model_id(vlm_model_id)
        kwargs: dict[str, Any] = {}
        if image_size is not None:
            kwargs["max_pixels"] = image_size * image_size
        return cls(AutoProcessor.from_pretrained(vlm_model_id, **kwargs))

    def preprocess(
        self,
        messages: list[dict[str, Any]],
        camera_role_token_ids: list[int | None] | None = None,
    ) -> dict[str, Any]:
        """Token IDs and packed vision inputs for chat messages and images."""
        images = [
            item["image"]
            for msg in messages
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        ]

        vision_inputs: dict[str, Any] = {}
        image_grid_thw: torch.Tensor | None = None
        if images:
            img_out = self._fast_img(images)
            vision_inputs["pixel_values"] = img_out["pixel_values"]
            vision_inputs["image_grid_thw"] = img_out["image_grid_thw"]
            image_grid_thw = img_out["image_grid_thw"]

        if self.token_cache is None:
            input_ids_2d = self._token_ids(messages, image_grid_thw, camera_role_token_ids)
        else:
            # The ids depend on the text, the image grids and the role tokens, not on pixels.
            key = (
                repr([(m["role"], _content_key(m["content"])) for m in messages]),
                None if image_grid_thw is None else image_grid_thw.numpy().tobytes(),
                None if camera_role_token_ids is None else tuple(camera_role_token_ids),
            )
            if key not in self.token_cache:
                if len(self.token_cache) >= 64:
                    self.token_cache.clear()
                self.token_cache[key] = self._token_ids(
                    messages, image_grid_thw, camera_role_token_ids
                )
            input_ids_2d = self.token_cache[key].clone()

        mm_token_type_ids = torch.zeros_like(input_ids_2d, dtype=torch.int32)
        mm_token_type_ids[input_ids_2d == self.image_pad_id] = 1
        mm_token_type_ids[input_ids_2d == self.video_pad_id] = 2

        vision_inputs["mm_token_type_ids"] = mm_token_type_ids

        return {
            "input_ids": input_ids_2d.squeeze(0),
            "vision_inputs": vision_inputs,
        }

    def _token_ids(
        self,
        messages: list[dict[str, Any]],
        image_grid_thw: torch.Tensor | None,
        camera_role_token_ids: list[int | None] | None,
    ) -> torch.Tensor:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = text.replace("<think>\n\n</think>\n\n", "")

        if image_grid_thw is not None:
            merge_length = self._fast_img.merge_size**2
            parts = text.split(self._image_tok_str)
            assert len(parts) - 1 == image_grid_thw.shape[0], (
                f"chat template has {len(parts) - 1} image placeholders but "
                f"got {image_grid_thw.shape[0]} images"
            )
            pieces = [parts[0]]
            for i, tail in enumerate(parts[1:]):
                n = int(image_grid_thw[i].prod().item()) // merge_length
                pieces.append(self._image_tok_str * n)
                pieces.append(tail)
            text = "".join(pieces)

        enc = self.tokenizer([text], return_tensors="pt")
        input_ids_2d = enc["input_ids"]

        if camera_role_token_ids is not None and image_grid_thw is not None:
            assert len(camera_role_token_ids) == image_grid_thw.shape[0], (
                f"camera_role_token_ids has {len(camera_role_token_ids)} entries "
                f"but {image_grid_thw.shape[0]} images"
            )
            input_ids = input_ids_2d.squeeze(0)
            vs_positions = (input_ids == self.vision_start_id).nonzero(as_tuple=True)[0].tolist()
            assert len(vs_positions) == image_grid_thw.shape[0], (
                f"expected {image_grid_thw.shape[0]} <|vision_start|> markers, "
                f"got {len(vs_positions)}"
            )
            for img_i in range(len(camera_role_token_ids) - 1, -1, -1):
                tid = camera_role_token_ids[img_i]
                if tid is None:
                    continue
                pos = vs_positions[img_i]
                input_ids = torch.cat(
                    [
                        input_ids[:pos],
                        torch.tensor([tid], dtype=input_ids.dtype),
                        input_ids[pos:],
                    ]
                )
            input_ids_2d = input_ids.unsqueeze(0)

        return input_ids_2d
