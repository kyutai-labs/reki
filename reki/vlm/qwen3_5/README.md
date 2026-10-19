# Qwen3.5 with the action expert

This package implements the VLM and action expert described in the [model
architecture](../../../docs/architecture.md).

## Files

| File | Contents |
| --- | --- |
| `modeling.py` | `Qwen3_5VLMWithExpert`: construction, `from_pretrained`, embedding and MRoPE positions, and the training forward of the VLM and expert. |
| `inference.py` | `Qwen3_5ExpertInference`, the inference methods of `Qwen3_5VLMWithExpert`, and `CrossAttnCache`. |
| `hf_patches.py` | Patches to the HF model and FLA kernels, installed by `from_pretrained` and `__init__`. |
| `action_expert.py` | The DiT action expert with self-attention, VLM cross-attention and a SwiGLU MLP. |
| `prefill_graph.py` | CUDA-graph replay of the VLM prefill. |
| `expert_graph.py` | CUDA-graph replay of the inference expert. |
| `cuda_graph.py` | Shared checked-capture and graph-LRU helpers. |
| `preprocessor.py` | Tokenization, chat templating and image preparation for Qwen3.5. |
| `image_processor.py` | Tensor-native Qwen3.5 image processing (single-frame patches via `from_processor`). |

`action_expert.py`, `cuda_graph.py`, `preprocessor.py` and `image_processor.py` import on a CPU-only
host; the other modules import the CUDA kernels (flash-attn and FLA).

## Training

`RekiFlowModel` in `reki/policy/modeling.py` builds the model with
`Qwen3_5VLMWithExpert.from_pretrained`. Each step calls `embed_vlm_inputs`, sets the expert inputs
on `fwd` (`ExpertForwardState`), then calls `forward(...)`.

## Inference

`RekiFlowModel.sample_actions` runs the prefill (`embed_vlm_inputs` and
`forward(prefill_only=True)`, with no VLM KV cache, or a `PrefillGraphs` replay), then
`gather_cross_attn_source`. `_expert_actions` computes
`expert_modulations`, runs `register_prepare` and `denoise_first_step_register` for Euler step 1,
then `denoise_forward_register` for each further step. `compile_expert` compiles the step callables,
and `ExpertGraphs` replays one predict as one CUDA graph.

## Pinned versions

`pyproject.toml` pins the versions of Transformers, FLA, torch, triton and flash-attn.
`hf_patches` re-implements HF forwards and checks the patched HF layout:
`install_vision_varlen` and `install_packed_decoder_layers` raise on a mismatch, so a Transformers
upgrade requires these patches to be checked again. `fla_inference_configs` pins FLA kernel configurations by name.

Parts of `hf_patches.py`, `inference.py` and `image_processor.py` (`smart_resize`) are adapted
from Transformers; see [NOTICE](../../../NOTICE).
