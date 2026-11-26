"""Compose-time interpolation for per-source action-pace.

`interpolate_wrist_chunk` evaluates the abs-world wrist chunk at fractional demo-frames
i*pace — lerp on translation/eigengrasps, slerp on rotation, nearest-row snap next to
invalid (stored-zeros) rows — with pace=1.0 the exact identity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from reki.data.robotics.transforms import interpolate_wrist_chunk

pytestmark = pytest.mark.cpu


def _chunk(n: int = 30) -> torch.Tensor:
    """Synthetic [n, 32] chunk: linear transl, constant-ω rotation, linear eigengrasps."""
    t = np.arange(n, dtype=np.float64)
    act = np.zeros((n, 32))
    for s, v in ((0, 0.01), (16, -0.02)):
        act[:, s : s + 3] = t[:, None] * v * np.array([1.0, 2.0, 3.0])
        act[:, s + 3 : s + 6] = t[:, None] * 0.05 * np.array([0.0, 0.0, 1.0])
        act[:, s + 6 : s + 16] = t[:, None] * 0.1 + s + 1.0
    return torch.as_tensor(act, dtype=torch.float32)


def test_fractional_lerp_exact_on_linear():
    act, pace = _chunk(), 0.6
    out = interpolate_wrist_chunk(act, pace)
    f = np.arange(30) * pace
    for s, v in ((0, 0.01), (16, -0.02)):
        want_t = f[:, None] * v * np.array([1.0, 2.0, 3.0])
        want_e = np.repeat(f[:, None] * 0.1 + s + 1.0, 10, axis=1)
        np.testing.assert_allclose(out[:, s : s + 3].numpy(), want_t, atol=1e-6)
        np.testing.assert_allclose(out[:, s + 6 : s + 16].numpy(), want_e, atol=1e-5)


def test_general_rotation_matches_scipy_slerp():
    rng = np.random.default_rng(0)
    act = _chunk()
    rots = R.random(30, rng=rng)
    act[:, 3:6] = torch.as_tensor(rots.as_rotvec(), dtype=torch.float32)
    out = interpolate_wrist_chunk(act, 0.7)
    f = np.arange(30) * 0.7
    from scipy.spatial.transform import Slerp

    want = Slerp(np.arange(30), R.from_rotvec(act[:, 3:6].numpy()))(f).as_rotvec()
    np.testing.assert_allclose(out[:, 3:6].numpy(), want, atol=1e-5)


def test_action_pace_key_check():
    from reki.data.robotics.factory import validate_action_pace

    with pytest.raises(ValueError, match="'egodexx', not a registered dataset"):
        validate_action_pace({"egodexx": 0.6}, ["egodex"])
    validate_action_pace({"egodex": 0.6, "egoverse": 0.25}, ["libero"])
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        validate_action_pace({"egodex": 1.5}, ["egodex"])
