"""FrameTokenizer's subtask turn pads the preprocessor's ``mm_token_type_ids`` with text."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

from reki.data.robotics.tokenization import FrameTokenizer

pytestmark = pytest.mark.cpu

IMAGE_PAD, VIDEO_PAD = 900, 901


class _Tokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # The subtask text holds a pad id: a suffix token is never a vision token.
        return {"input_ids": [ord(c) for c in text] + [IMAGE_PAD]}


class _Preprocessor:
    """The Qwen3.5 preprocessor's output contract: ``mm_token_type_ids`` [1, L] int32."""

    image_pad_id, video_pad_id = IMAGE_PAD, VIDEO_PAD
    asst_end_tokens = [7]
    subtask_header_ids = [5, 6]
    assistant_header_ids = [5, 8]
    tokenizer = _Tokenizer()

    def preprocess(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        n_images = sum(c.get("type") == "image" for c in messages[0]["content"])
        ids = [1, 2]
        for _ in range(n_images):
            ids += [3] + [IMAGE_PAD] * 4 + [4]
        ids += [VIDEO_PAD] * 2 + [10, 11, 12]
        input_ids = torch.tensor([ids])
        mm = torch.zeros_like(input_ids, dtype=torch.int32)
        mm[input_ids == IMAGE_PAD] = 1
        mm[input_ids == VIDEO_PAD] = 2
        return {"input_ids": input_ids.squeeze(0), "vision_inputs": {"mm_token_type_ids": mm}}


def _rebuild(ids: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Reference ``mm_token_type_ids`` rebuilt from the token ids, over the prompt prefix only."""
    mm = torch.zeros_like(ids, dtype=torch.int32)
    prefix = ids[:prefix_len]
    mm[:prefix_len][prefix == IMAGE_PAD] = 1
    mm[:prefix_len][prefix == VIDEO_PAD] = 2
    return mm.unsqueeze(0)


@pytest.mark.parametrize("subtask_in_user_stream", [False, True])
@pytest.mark.parametrize("subtask", [None, "grasp"])
def test_mm_token_type_ids_match_rebuild(subtask_in_user_stream: bool, subtask: str | None) -> None:
    prep = _Preprocessor()
    tok = FrameTokenizer(
        prep, subtask_in_user_stream=subtask_in_user_stream, use_camera_role_tokens=False
    )
    images = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]
    item: dict[str, Any] = {}
    tok.tokenize(item, images, [0, 1], "pick", subtask)

    ids = item[OBS_LANGUAGE_TOKENS]
    prefix_len = prep.preprocess([{"content": [{"type": "image"}] * 2}])["input_ids"].shape[0]
    mm = item["vision_inputs"]["mm_token_type_ids"]
    assert (ids.shape[0] > prefix_len) == (subtask is not None)
    assert mm.dtype == torch.int32
    assert torch.equal(mm, _rebuild(ids, prefix_len))
