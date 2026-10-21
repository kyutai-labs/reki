"""Qwen3.5 with the action expert (file map: README.md). ``action_expert``, ``cuda_graph``,
``preprocessor`` and ``image_processor`` import on a CPU-only host; the other modules import the
CUDA kernels (torch varlen attention, FLA)."""


def check_model_id(vlm_model_id: str) -> None:
    """Raise unless ``vlm_model_id`` names a Qwen3.5 model, the only VLM REKI supports."""
    lower = vlm_model_id.lower()
    if "qwen3.5" not in lower and "qwen3_5" not in lower:
        raise ValueError(f"REKI needs a Qwen3.5 VLM, got vlm_model_id={vlm_model_id!r}")
