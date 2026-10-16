"""Left-right (sagittal) mirror of state/action vectors, as an extra embodiment.

A reflection of the whole scene across a vertical plane gives a mirror-image robot, R*. The
mirror is exact: the h-flipped frames are R*'s camera views, and the permuted, signed vector is
R*'s configuration. Mirrored samples use their own state-MLP key and robot type.

Each mirrorable dataset declares a :class:`MirrorSpec`: a per-feature index permutation and a
±1 sign, ``out[i] = sign[i] * x[perm[i]]``. This covers a bimanual arm swap (permutation across
arm blocks) and a single-arm axis flip (sign only).

The task text is not mirrored. A mirrored sample gets the mirror marker and keeps its original
instruction.

The sign rules are in ``docs/data.md`` (Mirroring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
import torch

ALOHA_DIM: Final = 14
ALOHA_ARM_BLOCK: Final = ALOHA_DIM // 2

ALOHA_MIRROR_SIGN_ARX5: Final = (-1, 1, 1, 1, -1, -1, 1)

_EE_POSE_SIGN: Final = (1, -1, 1, -1, 1, -1)
EE_EULER_STATE_PERM: Final = (0, 1, 2, 3, 4, 5, 7, 6)
EE_EULER_STATE_SIGN: Final = (*_EE_POSE_SIGN, -1, -1)
EE_EULER_ACTION_PERM: Final = (0, 1, 2, 3, 4, 5, 6)
EE_EULER_ACTION_SIGN: Final = (*_EE_POSE_SIGN, 1)

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)

_UNSIGNED_STAT_FIELDS: Final = frozenset({"std"})
_SCALAR_STAT_FIELDS: Final = frozenset({"count"})
_ORDERED_STAT_PAIRS: Final = (
    ("min", "max"),
    ("q01", "q99"),
    ("q10", "q90"),
    ("qk01", "qk99"),
    ("qk10", "qk90"),
)


def _validate(perm: tuple[int, ...], sign: tuple[int, ...], what: str) -> None:
    if len(perm) != len(sign):
        raise ValueError(f"{what}: perm has {len(perm)} entries but sign has {len(sign)}")
    if sorted(perm) != list(range(len(perm))):
        raise ValueError(f"{what}: perm must be a permutation of 0..{len(perm) - 1}, got {perm}")
    bad = [s for s in sign if s not in (-1, 1)]
    if bad:
        raise ValueError(f"{what}: sign entries must be ±1, got {bad} in {sign}")


def mirror_vector(x: ArrayT, *, perm: tuple[int, ...], sign: tuple[int, ...]) -> ArrayT:
    """Return ``out[..., i] = sign[i] * x[..., perm[i]]``.

    Leading dims (history frames, chunk steps) are unchanged. The output has the type, dtype and
    device of ``x``. When ``perm`` is an involution, two calls return ``x``. This does not show
    that the sign is kinematically correct; the mirror tests check that.
    """
    _validate(perm, sign, "mirror_vector")
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x).__name__}")
    if x.shape[-1] != len(sign):
        raise ValueError(f"expected last dim {len(sign)}, got shape {tuple(x.shape)}")
    identity = perm == tuple(range(len(perm)))

    if isinstance(x, torch.Tensor):
        if not x.is_floating_point():
            raise TypeError(f"mirror needs a float tensor (negation wraps on ints), got {x.dtype}")
        gathered = x if identity else x[..., list(perm)]
        return gathered * torch.as_tensor(sign, dtype=x.dtype, device=x.device)

    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"mirror needs a float array (negation wraps on ints), got {x.dtype}")
    gathered = x if identity else x[..., list(perm)]
    return gathered * np.asarray(sign, dtype=x.dtype)


def mirror_feature_stats(
    stats: dict[str, list[float]], *, perm: tuple[int, ...], sign: tuple[int, ...]
) -> dict[str, list]:
    """Map one feature's normalization stats onto the mirrored distribution, in closed form.

    ``count`` is unchanged, ``mean`` and ``q50`` mirror, and ``std`` only permutes. Each
    low/high pair mirrors both members and then sorts them: the new minimum of a negated slot
    is the negated maximum (likewise q01 from -q99).
    """
    _validate(perm, sign, "mirror_feature_stats")
    dim = len(sign)
    pairs = []
    for low, high in _ORDERED_STAT_PAIRS:
        have = (low in stats, high in stats)
        if all(have):
            pairs.append((low, high))
        elif any(have):
            raise ValueError(f"{low!r}/{high!r} must appear together; got {sorted(stats)}")
    if ("min", "max") not in pairs:
        raise ValueError(f"stats must carry min/max; got {sorted(stats)}")

    out: dict[str, list] = {f: stats[f] for f in _SCALAR_STAT_FIELDS & stats.keys()}
    paired = {name for pair in pairs for name in pair}
    unsigned = tuple(1 for _ in sign)
    for field, value in stats.items():
        if field in _SCALAR_STAT_FIELDS or field in paired:
            continue
        v = np.asarray(value, dtype=np.float64)
        if v.shape != (dim,):
            raise ValueError(f"stats field {field!r} must be {dim}-dim, got {v.shape}")
        field_sign = unsigned if field in _UNSIGNED_STAT_FIELDS else sign
        out[field] = mirror_vector(v, perm=perm, sign=field_sign).tolist()

    for low, high in pairs:
        lo_m = mirror_vector(np.asarray(stats[low], dtype=np.float64), perm=perm, sign=sign)
        hi_m = mirror_vector(np.asarray(stats[high], dtype=np.float64), perm=perm, sign=sign)
        out[low] = np.minimum(lo_m, hi_m).tolist()
        out[high] = np.maximum(lo_m, hi_m).tolist()
    return out


@dataclass(frozen=True)
class MirrorSpec:
    """How one dataset's raw state/action vectors reflect across the sagittal plane.

    A spec is per dataset. Datasets of one embodiment can share a layout but store their axes in
    different world frames, so measure the lateral axis per dataset. ``paired_cameras`` lists
    the canonical camera pairs that swap under the mirror (R*'s left wrist camera is the mirror
    of the real right one). It is empty for a single-wrist embodiment, whose camera only
    h-flips.
    """

    name: str
    state_perm: tuple[int, ...]
    state_sign: tuple[int, ...]
    action_perm: tuple[int, ...]
    action_sign: tuple[int, ...]
    paired_cameras: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate(self.state_perm, self.state_sign, f"{self.name} state")
        _validate(self.action_perm, self.action_sign, f"{self.name} action")

    def dim(self, feature: str) -> int:
        """The raw vector width for ``feature``."""
        return len(self.state_sign if feature == "state" else self.action_sign)

    def mirror(self, x: ArrayT, feature: str) -> ArrayT:
        """Mirror a raw (pre-slice) ``state`` or ``action`` tensor."""
        perm, sign = self._for(feature)
        return mirror_vector(x, perm=perm, sign=sign)

    def mirror_stats(self, stats: dict[str, list[float]], feature: str) -> dict[str, list]:
        """The mirrored normalization statistics for one feature."""
        perm, sign = self._for(feature)
        return mirror_feature_stats(stats, perm=perm, sign=sign)

    def _for(self, feature: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if feature == "state":
            return self.state_perm, self.state_sign
        if feature == "action":
            return self.action_perm, self.action_sign
        raise ValueError(f"feature must be 'state' or 'action', got {feature!r}")


def _aloha_spec(name: str, arm_sign: tuple[int, ...]) -> MirrorSpec:
    """Bimanual ALOHA: swap the two arm blocks, apply ``arm_sign`` within each."""
    if len(arm_sign) != ALOHA_ARM_BLOCK:
        raise ValueError(f"{name}: arm sign needs {ALOHA_ARM_BLOCK} entries, got {len(arm_sign)}")
    perm = (*range(ALOHA_ARM_BLOCK, ALOHA_DIM), *range(ALOHA_ARM_BLOCK))
    sign = arm_sign * 2
    return MirrorSpec(
        name=name,
        state_perm=perm,
        state_sign=sign,
        action_perm=perm,
        action_sign=sign,
        paired_cameras=(
            ("observation.images.camera_wrist_left", "observation.images.camera_wrist_right"),
        ),
    )


# ALOHA [L_arm6, L_grip, R_arm6, R_grip]: swap the arm blocks; the joint axes [z, y, y, y, z, x]
# give the parity sign (an FK search on robotwin2_clean agrees). The real mounts are not exact
# mirrors (reach error 0.020 m): never execute mirrored actions on the real robot.
ALOHA_ARX5_SPEC: Final = _aloha_spec("aloha_arx5", ALOHA_MIRROR_SIGN_ARX5)
# LIBERO raw state [pos3, rot3, q0, q1], action [Δpos3, Δrot3, grip]; lateral axis = index 1
# (scripts/data/derive_libero_mirror_axis.py), plane p1 = 0 (the Panda base).
LIBERO_SPEC: Final = MirrorSpec(
    name="libero_franka_y",
    state_perm=EE_EULER_STATE_PERM,
    state_sign=EE_EULER_STATE_SIGN,
    action_perm=EE_EULER_ACTION_PERM,
    action_sign=EE_EULER_ACTION_SIGN,
)

# panda_omron state [base_pos3, base_quat4, ee_pos3, ee_quat4, fingers2], action
# [vx, vy, wz, torso, mode, Δpos3, Δrot3, grip]; lateral axis = base y (base pose in the world
# frame, EE pose in the base frame: same signs). agentview_left/right (camera_top /
# camera_side_1) are a mirror pair.
ROBOCASA365_SPEC: Final = MirrorSpec(
    name="panda_omron_base_y",
    state_perm=(*range(14), 15, 14),
    state_sign=(1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1),
    action_perm=tuple(range(12)),
    action_sign=(1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1),
    paired_cameras=(("observation.images.camera_top", "observation.images.camera_side_1"),),
)

MIRROR_SPECS: Final[dict[str, MirrorSpec]] = {
    spec.name: spec for spec in (ALOHA_ARX5_SPEC, LIBERO_SPEC, ROBOCASA365_SPEC)
}


def mirror_aloha(x: ArrayT, *, sign: tuple[int, ...] = ALOHA_MIRROR_SIGN_ARX5) -> ArrayT:
    """Mirror a ``[..., 14]`` ALOHA state or action across the sagittal plane.

    ``sign`` selects the per-slot sign.
    """
    return _aloha_spec("aloha", sign).mirror(x, "state")
