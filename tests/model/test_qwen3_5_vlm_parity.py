"""Our packed Qwen3.5 forward on a tiny random model: packing never crosses ``cu_seqlens``, and
one sample matches the stock HF logits (text only)."""

from __future__ import annotations

import pytest
import torch
from conftest import QWEN, TINY_EXPERT, TinyQwenAuto

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("tiny_qwen")]


def _load_ours(device: str = "cuda"):
    from reki.vlm.qwen3_5.modeling import Qwen3_5VLMWithExpert

    return (
        Qwen3_5VLMWithExpert.from_pretrained(
            QWEN,
            dtype=torch.bfloat16,
            **TINY_EXPERT,
            adarms_cond_dim=TINY_EXPERT["action_expert_hidden_size"],
            action_expert_num_registers=8,
        )
        .to(device)  # type: ignore[arg-type]
        .eval()
    )


def _text_vision_inputs(ids_2d: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"mm_token_type_ids": torch.zeros_like(ids_2d, dtype=torch.int32)}


def test_packed_forward_matches_per_segment() -> None:
    """The full packed text forward (2 samples, one stream) reproduces each sample
    forwarded alone — varlen kwargs + GDN shim + per-sample MRoPE compose."""
    torch.manual_seed(0)
    device = "cuda"
    ours = _load_ours(device)
    vocab = ours.vlm.config.text_config.vocab_size

    L0, L1 = 45, 78
    ids0 = torch.randint(0, min(vocab, 100_000), (L0,), device=device)
    ids1 = torch.randint(0, min(vocab, 100_000), (L1,), device=device)
    ids1_alt = torch.randint(0, min(vocab, 100_000), (L1,), device=device)

    def run(ids_list: list[torch.Tensor]) -> torch.Tensor:
        packed = torch.cat(ids_list)
        seq_lens = torch.tensor([t.numel() for t in ids_list], device=device, dtype=torch.int32)
        vis = [_text_vision_inputs(t.unsqueeze(0)) for t in ids_list]
        with torch.no_grad():
            prefix = ours.embed_vlm_inputs(input_ids=packed, vision_inputs=vis, seq_lens=seq_lens)
            (out, _), _ = ours.forward(inputs_embeds=(prefix, None), seq_lens=seq_lens)
        assert out is not None
        return out.squeeze(0).float()

    both = run([ids0, ids1])
    both_alt = run([ids0, ids1_alt])
    solo0 = run([ids0])
    solo1 = run([ids1])

    leak = (both[:L0] - both_alt[:L0]).abs().max().item()
    assert leak == 0.0, f"seg0 depends on seg1 content: max abs diff {leak}"

    scale = max(solo0.abs().max().item(), solo1.abs().max().item())
    d0 = (both[:L0] - solo0).abs().max().item() / scale
    d1 = (both[L0:] - solo1).abs().max().item() / scale
    assert d0 < 5e-2 and d1 < 5e-2, f"packed forward != per-segment: rel {d0=} {d1=}"


def test_qwen3_5_vlm_only_matches_native() -> None:
    """Our packed single-sample forward matches the native HF forward's logits
    (text-only prompt; shared weights isolate the plumbing under test)."""
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Briefly describe flow matching."}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)

    native = TinyQwenAuto.from_pretrained(QWEN, dtype=dtype, attn_implementation="sdpa")
    native = native.to(device).eval()  # type: ignore[assignment]
    with torch.no_grad():
        native_logits = native(input_ids=input_ids, use_cache=False).logits.float()
    lm_head = native.lm_head
    del native

    ours = _load_ours(device)
    ids_packed = input_ids.view(-1)
    seq_lens = torch.tensor([ids_packed.numel()], device=device, dtype=torch.int32)
    with torch.no_grad():
        prefix = ours.embed_vlm_inputs(
            input_ids=ids_packed,
            vision_inputs=[_text_vision_inputs(input_ids)],
            seq_lens=seq_lens,
        )
        (hidden, _), _ = ours.forward(inputs_embeds=(prefix, None), seq_lens=seq_lens)
        assert hidden is not None
        our_logits = lm_head(hidden).float()

    rel = (our_logits - native_logits).abs().max().item() / native_logits.abs().max().item()
    assert rel < 2e-2, f"our forward != native: relative logit diff {rel:.4g}"
