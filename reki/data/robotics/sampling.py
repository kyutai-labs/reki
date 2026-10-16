"""Sampling of the training mix: per-dataset weights (power law over frames, epoch cap), the
resulting ratios, and ``RatioSampler``."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Sampler

from reki.data.robotics.registry import DATASETS

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig


def planned_samples(cfg: TrainConfig) -> int:
    """Samples drawn over the run: ``steps`` counts optimizer steps, each of
    ``batch_size * WORLD_SIZE * gradient_accumulation_steps`` samples."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return cfg.steps * cfg.batch_size * world_size * cfg.gradient_accumulation_steps


def robot_sampling_ratios(cfg: TrainConfig, names: list[str], sizes: list[int]) -> list[float]:
    """Sampling ratio of each robot leaf (summing to 1) for ``cfg``'s weighting.

    ``sizes`` are the kept frames of each leaf. ``power_law`` weights by
    ``compute_dataset_weights`` and caps each dataset at ``max_epochs_per_dataset``; ``manual``
    takes ``weight_overrides`` as the weights. The epoch budget is ``planned_samples(cfg)``.
    """
    if cfg.dataset_weighting == "power_law":
        weights = compute_dataset_weights(
            names,
            sizes,
            frames_alpha=cfg.frames_alpha,
            overrides=cfg.weight_overrides,
        )
    else:
        weights = cfg.weight_overrides
    if cfg.max_epochs_per_dataset and cfg.dataset_weighting == "power_law":
        weights = cap_weights_by_epochs(
            names, sizes, weights, cfg.max_epochs_per_dataset, planned_samples(cfg)
        )
    return weighted_ratios(names, sizes, weights)


def print_dataset_mix(
    names: list[str], sizes: list[int], ratios: list[float], total_samples: int
) -> None:
    """Print size, sampling ratio and epochs per dataset family (``DatasetSpec.family``)."""
    per_epochs = [(total_samples * r) / s if s > 0 else 0.0 for s, r in zip(sizes, ratios)]
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        family = DATASETS[name].family if name in DATASETS else name
        groups.setdefault(family, []).append(i)
    print("  ---- per-group ----")
    for family, idxs in groups.items():
        g_size = sum(sizes[i] for i in idxs)
        g_ratio = sum(ratios[i] for i in idxs)
        g_eps = [per_epochs[i] for i in idxs]
        eps_str = f"{min(g_eps):.2f}-{max(g_eps):.2f}" if len(idxs) > 1 else f"{g_eps[0]:.2f}"
        print(f"  {family:<12s}  size={g_size:11,d}  ratio={g_ratio:.4f}  epochs={eps_str}")


class RatioSampler(Sampler[int]):
    """Two-level sampler over the leaves of a CombinedLeRobotDataset: pick a leaf by ratio, then a
    random index within it.

    Avoids the 2^24 category limit of ``torch.multinomial`` used by ``WeightedRandomSampler``.
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        ratios: list[float],
        num_samples: int,
        kept_local: list[np.ndarray | None] | None = None,
    ) -> None:
        assert len(dataset_sizes) == len(ratios)
        assert abs(sum(ratios) - 1.0) < 1e-5
        self._sizes = dataset_sizes
        self._ratios = ratios
        self._num_samples = num_samples
        self._kept_local = kept_local
        if kept_local is not None:
            assert len(kept_local) == len(dataset_sizes)
        self._offsets = [0]
        for s in dataset_sizes:
            self._offsets.append(self._offsets[-1] + s)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):  # type: ignore[override]
        for _ in range(self._num_samples):
            r = random.random()
            cum = 0.0
            ds_idx = len(self._sizes) - 1
            for i, ratio in enumerate(self._ratios):
                cum += ratio
                if r < cum:
                    ds_idx = i
                    break
            kept = self._kept_local[ds_idx] if self._kept_local is not None else None
            if kept is not None:
                local_idx = int(kept[random.randrange(len(kept))])
            else:
                local_idx = random.randint(0, self._sizes[ds_idx] - 1)
            yield self._offsets[ds_idx] + local_idx


def cap_weights_by_epochs(
    names: list[str],
    sizes: list[int],
    weights: dict[str, float],
    max_epochs: float,
    total_samples: int,
    safety_frac: float = 0.10,
) -> dict[str, float]:
    """Scale weights so that no dataset exceeds ``max_epochs`` passes in training.

    Runs three passes, because capping one dataset changes Σ(N·w). Raises when the capped
    datasets hold more than ``safety_frac`` of all frames.
    """
    weights = dict(weights)
    before = dict(weights)
    for _ in range(3):
        total_w = sum(s * weights[n] for n, s in zip(names, sizes))
        for n, s in zip(names, sizes):
            epochs = total_samples * weights[n] / total_w
            if epochs > max_epochs:
                weights[n] *= max_epochs / epochs
    affected = {n: s for n, s in zip(names, sizes) if weights[n] < before[n] - 1e-9}
    affected_frames = sum(affected.values())
    total_frames = sum(sizes)
    if affected_frames / total_frames > safety_frac:
        raise ValueError(
            f"max_epochs={max_epochs} caps {len(affected)} dataset(s) totaling "
            f"{affected_frames:,} frames ({affected_frames / total_frames:.1%} of "
            f"mixture, safety threshold {safety_frac:.0%}). Either raise the cap, "
            f"rethink the weighting, or drop affected datasets. Capped: "
            f"{sorted(affected, key=lambda n: affected[n], reverse=True)}"
        )
    return weights


def weighted_ratios(
    names: list[str],
    sizes: list[int],
    weights: dict[str, float],
) -> list[float]:
    """Normalized sampling ratios: ``P(dataset i) ∝ size_i * weight_i``."""
    effective = [s * weights.get(n, 1.0) for n, s in zip(names, sizes)]
    total = sum(effective)
    return [e / total for e in effective]


def compute_dataset_weights(
    names: list[str],
    sizes: list[int],
    frames_alpha: float = 0.5,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """w_d ∝ frames^(α-1) × override, keyed by unique name: a collection's leaves (its name
    repeated in ``names``, with ``sizes`` in the same order) are summed and weighted as one."""
    overrides = overrides or {}

    frames_by_name: dict[str, int] = {}
    for name, n_frames in zip(names, sizes):
        if n_frames > 0:
            frames_by_name[name] = frames_by_name.get(name, 0) + n_frames

    weights = {
        name: (n_frames ** (frames_alpha - 1.0)) * overrides.get(name, 1.0)
        for name, n_frames in frames_by_name.items()
    }
    mean_w = sum(weights.values()) / len(weights)
    return {n: w / mean_w for n, w in weights.items()}
