"""``RatioSampler``: dataset picked by ratio, then a local index, offset into the combined dataset."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from reki.data.robotics.factory import RatioSampler

pytestmark = pytest.mark.cpu

SIZES = [3, 10, 7]
RATIOS = [0.5, 0.2, 0.3]
BOUNDS = [(0, 3), (3, 13), (13, 20)]


def _draw(sampler: RatioSampler, seed: int) -> list[int]:
    random.seed(seed)
    return list(sampler)


def test_deterministic_with_seed() -> None:
    sampler = RatioSampler(SIZES, RATIOS, num_samples=200)
    assert _draw(sampler, 0) == _draw(sampler, 0)
    assert _draw(sampler, 0) != _draw(sampler, 1)
    assert len(_draw(sampler, 0)) == len(sampler) == 200


def test_ratios_and_index_ranges() -> None:
    n = 20_000
    counts = Counter(_draw(RatioSampler(SIZES, RATIOS, num_samples=n), 0))
    assert set(counts) == set(range(sum(SIZES))), "every global index is reachable, none beyond"
    for (lo, hi), ratio in zip(BOUNDS, RATIOS):
        share = sum(counts[i] for i in range(lo, hi)) / n
        assert abs(share - ratio) < 0.02, f"[{lo}, {hi}) drawn {share:.3f}, want {ratio}"


def test_kept_local_restricts_indices() -> None:
    kept = [None, np.array([2, 9]), np.array([0])]
    draws = set(_draw(RatioSampler(SIZES, RATIOS, num_samples=2000, kept_local=kept), 0))
    assert draws == {0, 1, 2, 3 + 2, 3 + 9, 13}


@pytest.mark.parametrize("ga", [1, 3])
def test_epoch_cap_counts_grad_accumulation(monkeypatch: pytest.MonkeyPatch, ga: int) -> None:
    from reki.data.robotics.sampling import robot_sampling_ratios
    from reki.policy.train_config import TrainConfig

    names, sizes = ["small", "big"], [1_000, 1_000_000]
    cfg = TrainConfig(steps=10_000, batch_size=30, max_epochs_per_dataset=100)
    monkeypatch.setenv("WORLD_SIZE", "24")
    reference = robot_sampling_ratios(cfg, names, sizes)
    cfg.gradient_accumulation_steps = ga
    monkeypatch.setenv("WORLD_SIZE", str(24 // ga))
    ratios = robot_sampling_ratios(cfg, names, sizes)
    assert ratios == pytest.approx(reference), "same samples per run, same mix"
    assert reference[0] < 1_000**0.5 / (1_000**0.5 + 1_000_000**0.5), "the cap binds on 'small'"
