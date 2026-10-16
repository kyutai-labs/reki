"""Collate function for robot batches."""

from collections.abc import Callable

import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from reki.policy.constants import ACTION_IS_PAD


def _trim_units_by(
    units: list[list[dict]], budget: int | None, cost_fn: Callable[[dict], int]
) -> list[list[dict]]:
    """Drop trailing units (one register group or one single item) so that the summed cost
    stays within ``budget``. The first unit is always kept, also when it is over budget."""
    if budget is None:
        return units
    total = 0
    for i, unit in enumerate(units):
        total += sum(cost_fn(m) for m in unit)
        if total > budget:
            return units[: max(i, 1)]
    return units


def _text_cost(item: dict) -> int:
    return item[OBS_LANGUAGE_TOKENS].shape[-1]


def combined_collate(
    batch: list[dict],
    max_image_tokens: int | None = None,
    max_text_tokens: int | None = None,
) -> dict[str, torch.Tensor | list]:
    """Collate for robot batches.

    Outputs packed tokens ``[total_L]`` + ``seq_lens [B]`` instead of
    padded ``[B, max_L]``.

    Register memory: a dataset item may be a group wrapper
    ``{"group_items": [frame_0, ..., anchor]}``; a plain item is a group of one. Groups are
    exploded into consecutive segments in ascending frame order (the memory-attention index
    of the action expert relies on this order), trims drop whole groups, and
    ``group_index``/``frame_pos`` [B] identify each segment's logical sample.
    """
    units = [item["group_items"] if "group_items" in item else [item] for item in batch]
    units = _trim_units_by(units, max_image_tokens, lambda m: int(m.get("n_image_tokens", 0)))
    units = _trim_units_by(units, max_text_tokens, _text_cost)
    batch = [m for unit in units for m in unit]
    group_index = [gi for gi, unit in enumerate(units) for _ in unit]
    frame_pos = [f for unit in units for f in range(len(unit))]

    all_tokens = torch.cat([item[OBS_LANGUAGE_TOKENS].squeeze(0) for item in batch])
    seq_lens = torch.tensor(
        [item[OBS_LANGUAGE_TOKENS].shape[-1] for item in batch], dtype=torch.int32
    )

    result: dict = {
        OBS_LANGUAGE_TOKENS: all_tokens,
        "seq_lens": seq_lens,
        "vision_inputs": [item["vision_inputs"] for item in batch],
        "camera_indices": [item["camera_indices"] for item in batch],
        "k_eff": [item.get("k_eff", 1) for item in batch],
        "group_index": torch.tensor(group_index, dtype=torch.long),
        "frame_pos": torch.tensor(frame_pos, dtype=torch.long),
    }
    for key in [OBS_STATE, ACTION, ACTION_IS_PAD]:
        result[key] = [item.get(key) for item in batch]

    result["chunk_size"] = [item.get("chunk_size") for item in batch]

    if any("state_mlp_embodiment" in item for item in batch):
        result["state_mlp_embodiment"] = [item.get("state_mlp_embodiment") for item in batch]

    return result
