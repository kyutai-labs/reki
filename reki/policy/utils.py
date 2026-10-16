"""Small helpers shared by training and evaluation."""

from __future__ import annotations

import torch


def to_device(batch: dict, device: torch.device) -> dict:
    """Move tensors in a collated batch to ``device`` (recursively through lists/dicts)."""

    def _move(v: object) -> object:
        if torch.is_tensor(v):
            return v.to(device, non_blocking=True)
        if isinstance(v, list):
            return [_move(el) for el in v]
        if isinstance(v, dict):
            return {k: _move(el) for k, el in v.items()}
        return v

    return {k: _move(v) for k, v in batch.items()}
