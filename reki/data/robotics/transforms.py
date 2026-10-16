"""Tensor transforms of one sample's state and action: slicing, gripper sign and rotation
representation."""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np
import torch

ROT_BLOCKS: dict[str, list[tuple[int, int]]] = {
    "bimanual_joint": [(3, 6), (10, 13)],
    "bimanual_dexterous": [(3, 6), (14, 17)],
}
"""Rotation blocks ``[lo, hi)`` of the bimanual end-effector embodiments."""
DEFAULT_ROT_BLOCKS: list[tuple[int, int]] = [(3, 6)]
"""The rotation block of every other (single-arm) embodiment."""


def rot_blocks(embodiment: str | None) -> list[tuple[int, int]]:
    """The rotation-coordinate blocks for an embodiment."""
    return ROT_BLOCKS.get(embodiment or "", DEFAULT_ROT_BLOCKS)


def gather_with_pad(x: torch.Tensor, sl: list[int | None]) -> torch.Tensor:
    """Gather along last dim; ``None`` entries become zero-padded slots."""
    out = x.new_zeros(*x.shape[:-1], len(sl))
    for d, i in enumerate(sl):
        if i is not None:
            out[..., d] = x[..., i]
    return out


def negate_gripper_dims(x: torch.Tensor | np.ndarray, flip_idxs: list[int] | None):
    """Negate the gripper dim(s) of a normalized vector in place, so that high means open.

    The training data (``CombinedLeRobotDataset``) and the eval decode
    (``reki.inference.features``) both use this function. Works on torch tensors and numpy
    arrays. ``flip_idxs`` may be None or empty (no-op)."""
    if flip_idxs:
        for i in flip_idxs:
            if x.shape[-1] > i:
                # torch|numpy union setitem
                x[..., i] = -x[..., i]  # type: ignore[arg-type]
    return x


def axisangle_to_euler(
    x: torch.Tensor, blocks: list[tuple[int, int]] = DEFAULT_ROT_BLOCKS
) -> torch.Tensor:
    """Convert each ``[lo:hi]`` block of ``x`` from an axis-angle rotvec to euler ``xyz`` (radians).

    The single_arm_ee_euler and bimanual end-effector layouts then share one rotation format.
    The robosuite eval applies the inverse (``reki.inference.features.euler_to_axisangle_np``).
    The pair round-trips exactly. ``blocks`` defaults to the single-arm ``[(3, 6)]``; bimanual
    embodiments pass their ``ROT_BLOCKS`` entry.
    """
    from scipy.spatial.transform import Rotation as R

    out = x.clone()
    for lo, hi in blocks:
        rot = x[..., lo:hi]
        eul = R.from_rotvec(rot.reshape(-1, 3).cpu().numpy()).as_euler("xyz")
        out[..., lo:hi] = torch.as_tensor(eul, dtype=x.dtype, device=x.device).reshape(rot.shape)
    return out


ArrayT = TypeVar("ArrayT", torch.Tensor, np.ndarray)


def recenter_euler_roll(x: ArrayT, blocks: list[tuple[int, int]] = DEFAULT_ROT_BLOCKS) -> ArrayT:
    """Shift the euler roll (first dim) of each ``[lo:hi]`` block by π, wrapped to [-π, π], in
    place: the euler xyz of ``R·Rx(π)``. A downward gripper has its roll at ±π, where a tiny
    rotation flips its sign; shifted, it is near 0. Commutes with the mirror's roll negation.
    Works for torch tensors and numpy arrays."""
    for lo, _ in blocks:
        # torch|numpy union setitem
        x[..., lo] = x[..., lo] % (2 * math.pi) - math.pi  # type: ignore[arg-type]
    return x
