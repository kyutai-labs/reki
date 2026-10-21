"""Parity: the per-channel select of ``install_mrope_select`` (the Triton 3.7 workaround) equals
HF's interleaved MRoPE on the Qwen3.5-2B rotary geometry."""

from __future__ import annotations

import copy

import pytest
import torch
from conftest import tiny_qwen_config

pytestmark = pytest.mark.hf


def test_mrope_select_matches_hf_interleave() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    from reki.vlm.qwen3_5.hf_patches import install_mrope_select

    rotary = Qwen3_5TextRotaryEmbedding(tiny_qwen_config().text_config)
    select = copy.deepcopy(rotary)
    install_mrope_select(select)

    torch.manual_seed(0)
    freqs = torch.randn(3, 2, 5, rotary.inv_freq.shape[0])
    ref = rotary.apply_interleaved_mrope(freqs.clone(), rotary.mrope_section)
    assert torch.equal(select.apply_interleaved_mrope(freqs, select.mrope_section), ref)
