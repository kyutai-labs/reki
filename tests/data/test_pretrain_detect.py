"""Curation detectors of the pretraining pipeline (``pretrain.common.detect``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pretrain.common import detect, verify
from pretrain.common.v30 import Episode, write_leaf

pytestmark = pytest.mark.cpu


def _moving(n: int, d: int, seed: int) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (np.sin(2 * np.pi * t * (1 + np.arange(d))) + seed).astype(np.float32)


def test_frozen_drops_dead_episodes_only(tmp_path: Path) -> None:
    n = 30
    live = _moving(n, 7, 0)
    live[:, 6] = 0.0  # a pad dim, constant in every episode: excluded
    frozen = np.full((n, 7), 0.3, dtype=np.float32)
    gripper_only = np.repeat(live[:1], n, axis=0)
    gripper_only[:, 5] = live[:, 5]  # 1 of 6 live dims moves: 5/6 > 0.70 frozen
    one_still = live.copy()
    one_still[:, 0] = 0.5  # 1 of 6 live dims frozen: kept
    states = [live, frozen, gripper_only, one_still]
    episodes = [Episode(i, s, s[:, :2], "t") for i, s in enumerate(states)]
    write_leaf(episodes, tmp_path / "leaf", fps=10, robot_type="test")
    assert detect.frozen(tmp_path / "leaf", "test") == [1, 2]


def test_short_video_drops_episodes_with_fewer_frames_than_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    episodes = []
    for i, (rows, frames) in enumerate([(8, 8), (12, 9), (6, 7), (10, 10)]):
        ep = Episode(i, _moving(rows, 3, i), _moving(rows, 2, i), "t")
        video = rng.integers(0, 255, size=(frames, 32, 32, 3), dtype=np.uint8)
        ep.frames["observation.images.top"] = video
        episodes.append(ep)
    leaf = tmp_path / "leaf"
    write_leaf(episodes, leaf, fps=10, robot_type="test")
    assert detect.short_video(leaf, "test") == [1]
    with pytest.raises(ValueError, match="episode 1 has 9 frames, length 12"):
        verify.check_videos(leaf)
    (leaf / "meta" / "dropped_episodes.json").write_text("[1]")
    verify.check_videos(leaf)
    (leaf / "meta" / "dropped_episodes.json").write_text("[0]")
    with pytest.raises(ValueError, match="episode 1 has"):
        verify.check_videos(leaf)


def test_frozen_keeps_a_dim_that_moves_in_some_episodes() -> None:
    std = np.zeros((100, 2))
    std[:, 0] = 1.0
    std[:3, 1] = 1.0  # frozen in 97% of episodes but live in 3: not by-design constant
    assert detect.frozen_episodes(std, list(range(100))) == []
    std[:3, 1] = 0.0  # never moves: constant, excluded
    std[50, 0] = 0.0
    assert detect.frozen_episodes(std, list(range(100))) == [50]


def _smooth_episode(n: int, seed: int) -> dict[str, np.ndarray]:
    t = np.linspace(0, 1, n)
    state = np.stack([np.sin(2 * np.pi * t + seed), 0.5 * t], axis=1)
    return {"state": state, "action": state[:, :1].copy()}


def test_sudden_change_flags_jumps_over_the_floor() -> None:
    episodes = [(i, _smooth_episode(60, i)) for i in range(8)]
    episodes[3][1]["state"][30, 0] += 0.5
    episodes[5][1]["state"][20, 1] += 0.03  # under the 0.05 floor
    assert detect.sudden_change_episodes(episodes, {}) == [3]
    assert detect.sudden_change_episodes(episodes, {"state": {0}}) == []


def test_trained_signal_slices_and_converts(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    state = rng.normal(size=(5, 7)).astype(np.float32)
    action = rng.normal(size=(5, 7)).astype(np.float32) * 0.3
    signal, skip = detect.trained_signal("interndata_franka", state, action, {"state": {6}})
    assert signal["state"].shape == (5, 8)
    assert (signal["state"][:, 6] == 0).all()
    assert skip["state"] == {7}
    from scipy.spatial.transform import Rotation as R

    np.testing.assert_allclose(
        signal["action"][:, 3:6], R.from_rotvec(action[:, 3:6]).as_euler("xyz"), rtol=1e-6
    )
