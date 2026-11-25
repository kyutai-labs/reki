"""Cameras of a sample: the canonical cameras and their roles, which raw cameras a leaf
loads (``select_top_cameras``), and which cameras one sample feeds the model
(``take_sample_cameras``, ``reduce_cameras``, ``wrist_static_flags``)."""

from __future__ import annotations

import random
from typing import Any, Literal

import torch

from reki.data.robotics.registry import SINGLE_ARM
from reki.policy.constants import ACTION, ACTION_IS_PAD

CAMERA_RANKING: list[str] = [
    "camera_top",
    "camera_wrist_left",
    "camera_wrist_right",
    "camera_side_1",
    "camera_side_2",
    "camera_side_3",
    "camera_depth",
]

CAMERA_KEYS: list[str] = [f"observation.images.{c}" for c in CAMERA_RANKING]
"""Canonical camera keys; the index of a key is its camera index."""
CAMERA_KEY_TO_INDEX: dict[str, int] = {k: i for i, k in enumerate(CAMERA_KEYS)}
TOP, WRIST_LEFT, WRIST_RIGHT = 0, 1, 2
SIDE_INDICES = (3, 4, 5)

CANONICAL_INDEX_TO_ROLE: list[str | None] = [
    "overhead",
    "wrist_left",
    "wrist_right",
    "side",
    "side",
    "side",
    None,
]


def camera_role(idx: int) -> str | None:
    """Canonical camera index → role name, or None for a camera with no role (depth)."""
    if not 0 <= idx < len(CANONICAL_INDEX_TO_ROLE):
        raise ValueError(f"camera index {idx} is outside CAMERA_RANKING")
    return CANONICAL_INDEX_TO_ROLE[idx]


def select_top_cameras(
    cam_map: dict[str, str],
    max_cameras: int | None,
    available: set[str] | None = None,
) -> list[str]:
    """Raw keys of ``cam_map`` (a ``DatasetSpec.cameras``), sorted by the ranking of their
    canonical camera and capped at ``max_cameras``.

    ``available`` keeps only the raw keys of one leaf. So one map can list the camera keys of
    every leaf of a collection. When several raw keys map to one canonical camera, only the
    first in ``cam_map`` order is kept."""
    keys = [k for k in cam_map if available is None or k in available]
    ranked = sorted(keys, key=lambda k: CAMERA_RANKING.index(cam_map[k].rsplit(".", 1)[-1]))
    seen: set[str] = set()
    out: list[str] = []
    for k in ranked:
        canon = cam_map[k]
        if canon in seen:
            continue
        seen.add(canon)
        out.append(k)
    return out if max_cameras is None else out[:max_cameras]


def close_camera_pairs(
    camera_keys: list[str],
    cam_map: dict[str, str],
    paired_cameras: tuple[tuple[str, str], ...],
    available: set[str],
) -> list[str]:
    """Add the raw partner of each selected camera of a mirror pair.

    A mirror swaps the two cameras of a pair before capping, so both need the same history
    fetch: a partner left out of ``camera_keys`` would come back as one frame and give the
    swapped slot no history."""
    canonical_to_raw: dict[str, str] = {}
    for raw in select_top_cameras(cam_map, None, available=available):
        canonical_to_raw[cam_map[raw]] = raw
    closed = list(camera_keys)
    for left, right in paired_cameras:
        for selected, partner in ((left, right), (right, left)):
            raw_selected = canonical_to_raw.get(selected)
            raw_partner = canonical_to_raw.get(partner)
            if raw_selected in closed and raw_partner is not None and raw_partner not in closed:
                closed.append(raw_partner)
    return closed


def take_sample_cameras(
    item: dict[str, Any],
    *,
    max_cameras: int | None,
    dropout_prob: float,
    pin_overhead: bool,
    shuffle: bool,
    k_avail: int,
) -> tuple[list[list[torch.Tensor]], list[int], list[str], int]:
    """Pop the cameras one sample feeds the model out of ``item`` (canonical keys).

    The first ``max_cameras`` present cameras in ranking order, each dropped with
    ``dropout_prob`` (never the overhead one with ``pin_overhead``; never all of them), in
    shuffled order with ``shuffle``. An all-zero (missing) camera is skipped. A history camera
    ``[k, C, H, W]`` keeps its last ``k_avail`` frames.

    Returns ``(frames per camera, camera indices, camera keys, k_eff)``, ``k_eff`` the most
    history frames of any camera (1 without history).
    """
    k_eff = 1
    per_cam_frames: list[list[torch.Tensor]] = []
    camera_indices: list[int] = []
    kept_cam_keys: list[str] = []
    present = [k for k in CAMERA_KEYS if k in item]
    cam_iter = present[:max_cameras] if max_cameras is not None else list(present)
    if dropout_prob > 0.0 and len(cam_iter) > 1:
        top = CAMERA_KEYS[TOP] if pin_overhead else None
        survivors = [k for k in cam_iter if k == top or random.random() >= dropout_prob]
        cam_iter = survivors if survivors else [cam_iter[0]]
    if shuffle:
        random.shuffle(cam_iter)
    for cam_key in cam_iter:
        if cam_key not in item:
            continue
        cam_tensor = item.pop(cam_key)
        item.pop(f"{cam_key}_is_pad", None)
        if cam_tensor.numel() == 0 or not bool(cam_tensor.any()):
            continue
        if cam_tensor.ndim == 3:
            frames = [cam_tensor]
        else:
            k_real = max(1, min(cam_tensor.shape[0], k_avail))
            cam_tensor = cam_tensor[-k_real:]
            frames = [cam_tensor[t] for t in range(cam_tensor.shape[0])]
            k_eff = max(k_eff, cam_tensor.shape[0])
        per_cam_frames.append(frames)
        camera_indices.append(CAMERA_KEY_TO_INDEX[cam_key])
        kept_cam_keys.append(cam_key)
    return per_cam_frames, camera_indices, kept_cam_keys, k_eff


def reduce_cameras(
    camera_indices: list[int],
    per_cam_frames: list,
    kept_cam_keys: list,
    drop_side: bool,
    drop_wrist: bool,
    left_static: bool,
    right_static: bool,
) -> None:
    """Mutate the parallel camera lists in place: reduce >=2 side cams to one (random keep),
    and drop the wrist of each static arm. Always leaves >=1 non-overhead camera (never
    collapses to overhead-only)."""

    def _drop(pos: int) -> None:
        del per_cam_frames[pos]
        del camera_indices[pos]
        del kept_cam_keys[pos]

    if drop_side:
        side = [i for i, ci in enumerate(camera_indices) if ci in SIDE_INDICES]
        if len(side) > 1:
            keep = random.choice(side)
            for pos in sorted((i for i in side if i != keep), reverse=True):
                _drop(pos)
    if drop_wrist and WRIST_LEFT in camera_indices and WRIST_RIGHT in camera_indices:
        for ci in (WRIST_LEFT, WRIST_RIGHT):
            static = left_static if ci == WRIST_LEFT else right_static
            if static and sum(c != TOP for c in camera_indices) > 1:
                _drop(camera_indices.index(ci))


def wrist_static_flags(
    item: dict[str, Any],
    cs_ds: int,
    stats: dict[str, dict[str, torch.Tensor]],
    *,
    arm_action_dims: tuple[list[int], list[int]] | Literal["single_arm"] | None,
    control_mode: str,
    gripper_dims: list[int] | None,
    threshold: float,
    dataset_name: str,
) -> tuple[bool, bool]:
    """(left_static, right_static) for a bimanual sample from the RAW (pre-normalization)
    future action chunk. Static is judged in mean_std/std units: absolute-control pose dims
    use range (max-min), delta-control pose dims use magnitude (max|.|); gripper dims always
    use range (gripper is absolute even under delta-EE). Returns (False, False) when it can't
    judge (no left/right ``arm_action_dims``, too-short chunk). Fail-loud on missing
    stats."""
    if ACTION not in item or arm_action_dims is None or arm_action_dims == SINGLE_ARM:
        return False, False
    action_stats = stats[ACTION]
    if "mean" not in action_stats or "std" not in action_stats:
        raise KeyError(
            f"drop_static_wrist_camera needs mean/std action stats for dataset "
            f"{dataset_name!r}; got {sorted(action_stats)}."
        )
    chunk = item[ACTION][-cs_ds:].float()
    pad = item.get(ACTION_IS_PAD)
    if pad is not None:
        pad = torch.as_tensor(pad)[-cs_ds:].bool()
        if pad.any():
            chunk = chunk[~pad]
    if chunk.shape[0] < 2:
        return False, False
    normed = (chunk - action_stats["mean"]) / action_stats["std"].clamp_min(1e-6)
    grip = set(gripper_dims or [])
    is_abs = "absolute" in control_mode
    rng = (normed.max(0).values - normed.min(0).values).abs()
    mag = normed.abs().max(0).values

    def arm_static(dims: list[int]) -> bool:
        pose = [d for d in dims if d not in grip]
        gd = [d for d in dims if d in grip]
        pose_m = float((rng if is_abs else mag)[pose].max()) if pose else 0.0
        grip_m = float(rng[gd].max()) if gd else 0.0
        return pose_m < threshold and grip_m < threshold

    left, right = arm_action_dims
    return arm_static(left), arm_static(right)
