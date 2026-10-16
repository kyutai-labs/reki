"""Our Qwen3.5 preprocessor against the HF processor.

Our image path emits single-frame ``t=1`` patches (width ``C*p*p``) for the temporally folded
patch embed, where HF duplicates the frame across ``t=2`` (width ``C*2*p*p``). The contract:
identical tokens/grids, and our patch == HF's ``t=0`` slice. Float ``[0, 1]`` input (LIBERO's
parquet images) must give the same result as its uint8 original.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoProcessor

from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

pytestmark = pytest.mark.hf


@pytest.fixture(scope="module")
def processor() -> Any:
    return AutoProcessor.from_pretrained(
        "Qwen/Qwen3.5-2B", min_pixels=448 * 448, max_pixels=448 * 448
    )


def _messages(images: list, text: str = "pick up the red cube") -> list[dict]:
    content: list[dict] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def test_preprocessor_matches_hf_t0_slice(processor: Any) -> None:
    torch.manual_seed(0)
    imgs = [
        torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        for h, w in ((256, 256), (180, 320), (480, 640))
    ]
    msgs = _messages(imgs)
    ours = Qwen3_5Preprocessor(processor).preprocess(msgs)

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    hf = processor(images=imgs, text=[text], return_tensors="pt", return_mm_token_type_ids=True)
    assert torch.equal(ours["input_ids"], hf["input_ids"].squeeze(0))
    vi = ours["vision_inputs"]
    assert torch.equal(vi["mm_token_type_ids"], hf["mm_token_type_ids"])
    assert torch.equal(vi["image_grid_thw"], hf["image_grid_thw"])

    ip = processor.image_processor
    C, t, p = 3, ip.temporal_patch_size, ip.patch_size
    n = hf["pixel_values"].shape[0]
    hf_t = hf["pixel_values"].view(n, C, t, p, p)
    assert torch.equal(hf_t[:, :, 0], hf_t[:, :, 1]), "HF duplicates the frame across t"
    torch.testing.assert_close(
        vi["pixel_values"], hf_t[:, :, 0].reshape(n, C * p * p), atol=1e-5, rtol=1e-5
    )


def test_float01_input_matches_uint8(processor: Any) -> None:
    torch.manual_seed(0)
    img_u8 = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    pre = Qwen3_5Preprocessor(processor)
    out_u8 = pre.preprocess(_messages([img_u8]))["vision_inputs"]["pixel_values"]
    out_f = pre.preprocess(_messages([img_u8.float() / 255.0]))["vision_inputs"]["pixel_values"]
    torch.testing.assert_close(out_f, out_u8, atol=1e-5, rtol=0)
