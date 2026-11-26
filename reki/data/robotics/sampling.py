"""Sampling of the training mix: per-dataset weights (power law over frames and diversity,
epoch cap), the resulting ratios, and ``RatioSampler``."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Sampler

from reki.data.robotics.registry import DATASETS

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

WEIGHT_GROUPS: dict[str, str] = {n: s.weight_group for n, s in DATASETS.items() if s.weight_group}


def planned_samples(cfg: TrainConfig) -> int:
    """Samples drawn over the run: ``steps`` counts optimizer steps, each of
    ``batch_size * WORLD_SIZE * gradient_accumulation_steps`` samples."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return cfg.steps * cfg.batch_size * world_size * cfg.gradient_accumulation_steps


def robot_sampling_ratios(cfg: TrainConfig, names: list[str], sizes: list[int]) -> list[float]:
    """Sampling ratio of each robot leaf (summing to 1) for ``cfg``'s weighting.

    ``sizes`` are the kept frames of each leaf. ``power_law`` weights by
    ``compute_dataset_weights`` and caps each dataset at ``max_epochs_per_dataset``; ``manual``
    takes ``weight_overrides`` as the weights. The epoch budget is ``planned_samples(cfg)``,
    less the ``vl_ratio`` share of VL samples with ``vl_datasets``. Raises on a
    ``weight_overrides`` key that is not a registered dataset; one outside ``names`` is ignored.
    """
    unknown = sorted(cfg.weight_overrides.keys() - DATASETS.keys())
    if unknown:
        raise ValueError(f"weight_overrides has entries for {unknown}, not registered datasets.")
    if cfg.dataset_weighting == "power_law":
        weights = compute_dataset_weights(
            names,
            sizes,
            frames_alpha=cfg.frames_alpha,
            overrides=cfg.weight_overrides,
            language_diversity_path=cfg.language_diversity_path,
            language_alpha=cfg.language_alpha,
            vision_diversity_path=cfg.vision_diversity_path,
            vision_alpha=cfg.vision_alpha,
            weight_groups=WEIGHT_GROUPS,
        )
    else:
        weights = cfg.weight_overrides
    if cfg.max_epochs_per_dataset and cfg.dataset_weighting == "power_law":
        robot_share = 1.0 - cfg.vl_ratio if cfg.vl_datasets else 1.0
        weights = cap_weights_by_epochs(
            names,
            sizes,
            weights,
            cfg.max_epochs_per_dataset,
            int(planned_samples(cfg) * robot_share),
        )
    return weighted_ratios(names, sizes, weights)


def print_dataset_mix(
    names: list[str], sizes: list[int], ratios: list[float], total_samples: int
) -> None:
    """Print size, sampling ratio and epochs per dataset family (``DatasetSpec.family``; the
    VL block is ``VL``)."""
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
        # drawn when float rounding leaves r above the cumulative ratio
        self._last_sampled = max(i for i, r in enumerate(ratios) if r > 0)
        self._offsets = [0]
        for s in dataset_sizes:
            self._offsets.append(self._offsets[-1] + s)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):  # type: ignore[override]
        for _ in range(self._num_samples):
            r = random.random()
            cum = 0.0
            ds_idx = self._last_sampled
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
    language_diversity_path: str | None = None,
    language_alpha: float = 0.5,
    vision_diversity_path: str | None = None,
    vision_alpha: float = 0.25,
    weight_groups: dict[str, str] | None = None,
) -> dict[str, float]:
    """w_d ∝ frames^(α-1) × (L/med_L)^α_L × (V/med_V)^α_V × override, keyed by unique name: a
    collection's leaves (its name repeated in ``names``, with ``sizes`` in the same order) are
    summed and weighted as one.

    ``weight_groups`` (name → group id, default: the name) pools same-source datasets into one
    unit: frames and diversity are summed per group, and every member gets the group's
    per-frame weight.

    ``language_diversity_path`` / ``vision_diversity_path``: JSON maps of dataset name → the
    effective rank of a text encoder over the task strings / an image encoder over the
    primary-camera frames. They multiply the weight by ``(L / median L)^language_alpha`` /
    ``(V / median V)^vision_alpha``. Every dataset needs a language and a vision score.
    """
    overrides = overrides or {}
    weight_groups = weight_groups or {}

    frames_by_name: dict[str, int] = {}
    for name, n_frames in zip(names, sizes):
        if n_frames > 0:
            frames_by_name[name] = frames_by_name.get(name, 0) + n_frames
    unique_names = list(frames_by_name)

    group_of = {n: weight_groups.get(n, n) for n in unique_names}
    frames_by_group: dict[str, int] = {}
    for name, n_frames in frames_by_name.items():
        g = group_of[name]
        frames_by_group[g] = frames_by_group.get(g, 0) + n_frames

    def _group_score(div: dict[str, float]) -> dict[str, float]:
        """Frames-weighted mean over members, per group."""
        num: dict[str, float] = {}
        den: dict[str, int] = {}
        for name, n_frames in frames_by_name.items():
            g = group_of[name]
            num[g] = num.get(g, 0.0) + n_frames * div[name]
            den[g] = den.get(g, 0) + n_frames
        return {g: num[g] / den[g] for g in num}

    lang_div: dict[str, float] | None = None
    if language_diversity_path:
        lang_div = json.loads(Path(language_diversity_path).read_text())
        assert lang_div is not None
        missing_l = [n for n in unique_names if n not in lang_div]
        if missing_l:
            raise ValueError(
                f"Missing language diversity scores for {len(missing_l)} dataset(s): "
                f"{missing_l} in {language_diversity_path}. See docs/training.md#training-setup."
            )

    vis_div: dict[str, float] | None = None
    if vision_diversity_path:
        vis_div = json.loads(Path(vision_diversity_path).read_text())
        assert vis_div is not None
        missing_v = [n for n in unique_names if n not in vis_div]
        if missing_v:
            raise ValueError(
                f"Missing vision diversity scores for {len(missing_v)} dataset(s): "
                f"{missing_v} in {vision_diversity_path}. See docs/training.md#training-setup."
            )

    lang_g = _group_score(lang_div) if lang_div is not None else None
    vis_g = _group_score(vis_div) if vis_div is not None else None

    median_lang = 1.0
    if lang_g is not None:
        lang_known = sorted(lang_g.values())
        median_lang = lang_known[len(lang_known) // 2]
    median_vis = 1.0
    if vis_g is not None:
        vis_known = sorted(vis_g.values())
        median_vis = vis_known[len(vis_known) // 2]

    weights: dict[str, float] = {}
    for name in frames_by_name:
        g = group_of[name]
        f_group = frames_by_group[g]
        override = overrides.get(name, 1.0)
        w = (f_group ** (frames_alpha - 1.0)) * override
        if lang_g is not None:
            w *= (lang_g[g] / median_lang) ** language_alpha
        if vis_g is not None:
            w *= (vis_g[g] / median_vis) ** vision_alpha
        weights[name] = w

    mean_w = sum(weights.values()) / len(weights)
    return {n: w / mean_w for n, w in weights.items()}
