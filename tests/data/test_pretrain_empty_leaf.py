"""A leaf whose episodes are all dropped (cross-leaf duplicates) samples nothing and breaks
nothing in the training sampler."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpu


def test_all_dropped_leaf_keeps_no_frame() -> None:
    from reki.data.robotics.combined_dataset import kept_frames

    bounds = ([0, 4], [4, 9], [0, 1])
    assert len(kept_frames(bounds, {0, 1}, None, None, False)) == 0


def test_empty_leaf_gets_no_ratio_and_no_sample() -> None:
    from reki.data.robotics.sampling import RatioSampler, robot_sampling_ratios
    from reki.policy.train_config import TrainConfig

    cfg = TrainConfig(dataset="robocoin_a2d_41", max_epochs_per_dataset=0)
    ratios = robot_sampling_ratios(cfg, ["robocoin_a2d_41"] * 3, [100, 0, 50])
    assert ratios[1] == 0.0 and sum(ratios) == pytest.approx(1.0)
    empty = np.array([], dtype=np.int64)
    sampler = RatioSampler([100, 9, 50], ratios, 2000, [None, empty, None])
    assert not any(100 <= i < 109 for i in sampler)


def test_rounding_fallback_skips_an_empty_last_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    from reki.data.robotics.sampling import RatioSampler

    monkeypatch.setattr(random, "random", lambda: 1.0 - 1e-9)
    empty = np.array([], dtype=np.int64)
    sampler = RatioSampler([10, 4], [1.0 - 1e-6, 0.0], 5, [None, empty])
    assert all(0 <= i < 10 for i in sampler)
