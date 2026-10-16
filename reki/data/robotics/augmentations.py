"""Image augmentations for robot frames.

Applied per frame on uint8 [C, H, W] tensors before VLM preprocessing, so the fast Qwen
image processor still receives uint8 input.

Color jitter: 0.2 brightness, contrast and saturation, no hue. All cameras of one frame get the
same jitter.
"""

from __future__ import annotations

import random

import torch
from torchvision.transforms.v2 import functional as tvF

AUG_COLOR_JITTER_DEFAULT = 0.2
AUG_CROP_FRACTION_DEFAULT = 0.95

CROP_CAMERA_KEYS: frozenset[str] = frozenset(
    {
        "observation.images.camera_top",
        "observation.images.camera_side_1",
        "observation.images.camera_side_2",
        "observation.images.camera_side_3",
    }
)


def sample_color_jitter_scalars(
    magnitude: float = AUG_COLOR_JITTER_DEFAULT,
) -> tuple[float, float, float]:
    """Sample (brightness, contrast, saturation) factors, each ~U[1-m, 1+m]."""
    m = magnitude
    b = random.uniform(max(0.0, 1.0 - m), 1.0 + m)
    c = random.uniform(max(0.0, 1.0 - m), 1.0 + m)
    s = random.uniform(max(0.0, 1.0 - m), 1.0 + m)
    return b, c, s


def apply_color_jitter_uint8(img: torch.Tensor, b: float, c: float, s: float) -> torch.Tensor:
    """Apply ColorJitter functionally to a uint8 [C, H, W] tensor; returns uint8."""
    out = tvF.adjust_brightness(img, b)
    out = tvF.adjust_contrast(out, c)
    out = tvF.adjust_saturation(out, s)
    return out


def sample_and_apply_random_crop(
    img: torch.Tensor, fraction: float = AUG_CROP_FRACTION_DEFAULT
) -> torch.Tensor:
    """Random crop preserving aspect ratio; uint8 [C, H, W] in, uint8 out.

    No resize-back — the VLM image processor resizes the crop to its pixel budget.
    """
    _, h, w = img.shape
    crop_h = int(h * fraction)
    crop_w = int(w * fraction)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    return img[:, top : top + crop_h, left : left + crop_w]


def sample_crop_box(h: int, w: int, min_fraction: float) -> tuple[int, int, int, int]:
    """Sample (top, left, crop_h, crop_w) with a random scale ``fraction ~ U(min_fraction, 1)``.

    The range includes 1.0, the scale at inference (eval has no center crop). The fraction is
    linear, so ``min_fraction=0.9`` gives an area of 0.81-1.0.
    """
    fraction = random.uniform(min_fraction, 1.0)
    crop_h = int(h * fraction)
    crop_w = int(w * fraction)
    return random.randint(0, h - crop_h), random.randint(0, w - crop_w), crop_h, crop_w


def apply_crop(img: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Crop ``img`` to ``box``; uint8 [C, H, W] in, uint8 out."""
    top, left, crop_h, crop_w = box
    return img[:, top : top + crop_h, left : left + crop_w]


def horizontal_flip(img: torch.Tensor) -> torch.Tensor:
    """Mirror an image left-right; [C, H, W] or [K, C, H, W] in, same shape out.

    Flips the width axis only, so every history frame flips the same way. It is part of the
    sagittal mirror (with the state/action transform and the wrist camera swap) and runs only
    on a mirror draw.
    """
    if img.ndim not in (3, 4):
        raise ValueError(f"expected [C, H, W] or [K, C, H, W], got shape {tuple(img.shape)}")
    return img.flip(-1)


def augment_frames(
    per_cam_frames: list[list[torch.Tensor]],
    cam_keys: list[str],
    k_eff: int,
    color_jitter: float | None,
    random_crop: float | None,
    random_scale: bool,
) -> list[torch.Tensor]:
    """The images of one sample, frame-major (all cameras of frame 0, then frame 1, …),
    augmented: a random crop of the ``CROP_CAMERA_KEYS`` cameras (one box per camera for the
    whole history with ``random_scale``, else one per image) and one color jitter per frame,
    shared by its cameras. A camera with fewer frames repeats its last one."""
    cam_boxes: list[tuple[int, int, int, int] | None] = []
    for cam_idx, frames in enumerate(per_cam_frames):
        if cam_keys[cam_idx] not in CROP_CAMERA_KEYS:
            cam_boxes.append(None)
            continue
        _, h_cam, w_cam = frames[0].shape
        cam_boxes.append(
            sample_crop_box(h_cam, w_cam, random_crop)
            if random_scale and random_crop is not None
            else None
        )
    images: list[torch.Tensor] = []
    for t in range(k_eff):
        jitter = sample_color_jitter_scalars(color_jitter) if color_jitter is not None else None
        for cam_idx, frames in enumerate(per_cam_frames):
            img = frames[t] if t < len(frames) else frames[-1]
            box = cam_boxes[cam_idx]
            if box is not None:
                img = apply_crop(img, box)
            elif random_crop is not None and cam_keys[cam_idx] in CROP_CAMERA_KEYS:
                img = sample_and_apply_random_crop(img, random_crop)
            if jitter is not None:
                img = apply_color_jitter_uint8(img, *jitter)
            images.append(img)
    return images
