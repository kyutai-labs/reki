"""Tensor transforms of one sample's state and action: slicing, gripper sign, rotation
representation, and the egocentric wrist action."""

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

HUMAN_HAND_WRIST_STARTS = (0, 16)
"""Start of each hand's 16-dim block in the egocentric action: wrist translation (3),
wrist axis-angle (3), eigengrasp coefficients (10)."""


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


def interpolate_wrist_chunk(
    action: torch.Tensor,
    pace: float,
    wrist_starts: tuple[int, ...] = HUMAN_HAND_WRIST_STARTS,
) -> torch.Tensor:
    """Motion-pace-stretch the abs-world wrist chunk (``pace < 1``).

    Waypoint ``i`` is the trajectory evaluated at fractional demo-frame ``i*pace``:
    lerp on translation + eigengrasps, slerp on the wrist rotation — no staircase (the
    fetched chunk is native-rate; ``(N-1)*pace < N`` so every query is in range). A hand
    whose neighboring source row is invalid (stored all-zero block) falls back to the
    nearest row, preserving the stored-zeros convention. ``pace == 1.0`` is the identity.
    """
    n = action.shape[0]
    if pace == 1.0 or n == 1:
        return action
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    f = np.arange(n) * pace
    i0 = torch.from_numpy(np.floor(f).astype(np.int64))
    i1 = torch.minimum(i0 + 1, torch.tensor(n - 1))
    a = torch.as_tensor(f - i0.numpy(), dtype=action.dtype).unsqueeze(1)
    nearest = torch.from_numpy(np.minimum(np.round(f).astype(np.int64), n - 1))
    out = action.clone()
    for s in wrist_starts:
        hand = action[:, s : s + 16]
        lin = torch.cat([hand[:, 0:3], hand[:, 6:16]], dim=1)
        lin_f = (1 - a) * lin[i0] + a * lin[i1]
        rots = R.from_rotvec(hand[:, 3:6].detach().cpu().numpy())
        rot_f = Slerp(np.arange(n), rots)(f).as_rotvec()
        out[:, s : s + 3] = lin_f[:, 0:3]
        out[:, s + 3 : s + 6] = torch.as_tensor(rot_f, dtype=action.dtype)
        out[:, s + 6 : s + 16] = lin_f[:, 3:13]
        invalid = ~hand.any(dim=1)
        snap = invalid[i0] | invalid[i1]
        if snap.any():
            out[snap, s : s + 16] = hand[nearest[snap]]
    return out


def wrist_to_camera_frame(
    action: torch.Tensor,
    camera_pose: torch.Tensor,
    wrist_starts: tuple[int, ...] = HUMAN_HAND_WRIST_STARTS,
) -> torch.Tensor:
    """Express each stored absolute world wrist pose in the observation camera frame (the
    egocentric representation of VITRA): ``waypoint = T_cam^-1 · T_wrist`` — translation ``R_cam^T (p_wrist - p_cam)``, rotation
    ``R_cam^T R_wrist`` (axis-angle). Eigengrasp / other dims pass through.

    ``action`` is ``[N, D]`` (abs-world wrist at ``s:s+3`` transl, ``s+3:s+6`` axis-angle per hand);
    ``camera_pose`` is ``[6]`` = the observation-frame camera ``[transl(3), axis-angle(3)]`` (world).
    An untracked hand row (its 16-dim block all zero) stays zero.
    """
    from scipy.spatial.transform import Rotation as R

    cam = camera_pose.detach().cpu().numpy()
    cam_t = cam[:3]
    cam_Rm = R.from_rotvec(cam[3:6]).as_matrix()
    out = action.clone()
    for s in wrist_starts:
        wt = action[:, s : s + 3].detach().cpu().numpy()
        w_Rm = R.from_rotvec(action[:, s + 3 : s + 6].detach().cpu().numpy()).as_matrix()
        rel_t = (wt - cam_t) @ cam_Rm
        rel_Rm = np.einsum("ij,njk->nik", cam_Rm.T, w_Rm)
        out[:, s : s + 3] = torch.as_tensor(rel_t, dtype=action.dtype)
        out[:, s + 3 : s + 6] = torch.as_tensor(
            R.from_matrix(rel_Rm).as_rotvec(), dtype=action.dtype
        )
        out[~action[:, s : s + 16].any(dim=1), s : s + 16] = 0
    return out


def untracked_frames(
    action: np.ndarray, wrist_starts: tuple[int, ...] = HUMAN_HAND_WRIST_STARTS
) -> np.ndarray:
    """``[N]`` bool: the rows of an ``[N, D]`` egocentric action where every hand is untracked
    (each 16-dim block all zero), so the frame has no action target."""
    tracked = np.stack([action[:, s : s + 16].any(axis=1) for s in wrist_starts])
    return np.logical_not(tracked.any(axis=0))


def tracked_hand_dims(
    action: torch.Tensor, wrist_starts: tuple[int, ...] = HUMAN_HAND_WRIST_STARTS
) -> torch.Tensor:
    """``[N, D]`` bool: False on the 16 dims of each hand row that is untracked (stored all
    zero), True elsewhere."""
    valid = torch.ones_like(action, dtype=torch.bool)
    for s in wrist_starts:
        valid[:, s : s + 16] = action[:, s : s + 16].any(dim=1, keepdim=True)
    return valid


def wrist_action_in_camera_frame(
    action: torch.Tensor, state: torch.Tensor, pace: float
) -> torch.Tensor:
    """The egocentric ("relative end-effector") action chunk(s) of one sample: each chunk
    paced (``interpolate_wrist_chunk``) and expressed in the frame of
    its observation camera, whose pose is the first 6 state dims of that frame. A history of
    ``k`` frames (``state`` ``[k, D]``) holds ``k`` stacked chunks."""
    if state.ndim == 2 and state.shape[0] > 1:
        k_frames = state.shape[0]
        chunks = action.reshape(k_frames, -1, action.shape[-1])
        composed = []
        for f in range(k_frames):
            chunk = interpolate_wrist_chunk(chunks[f], pace)
            composed.append(wrist_to_camera_frame(chunk, state[f][:6]))
        return torch.cat(composed, dim=0)
    return wrist_to_camera_frame(interpolate_wrist_chunk(action, pace), state.reshape(-1)[:6])
