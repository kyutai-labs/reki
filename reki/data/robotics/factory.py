"""Build the training dataset of a ``TrainConfig``."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.datasets.io_utils import load_info
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import Sampler

from reki.data.robotics.cameras import close_camera_pairs, select_top_cameras
from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
from reki.data.robotics.lerobot_io import (
    load_lerobot_dir,
    training_leaves,
)
from reki.data.robotics.lerobot_patches import thread_safe_tqdm_lock
from reki.data.robotics.mirror import MIRROR_SPECS
from reki.data.robotics.registry import (
    DATASETS,
    expand_spec,
    mirror_state_embodiments,
    state_embodiments,
)
from reki.data.robotics.sampling import (
    RatioSampler,
    planned_samples,
    print_dataset_mix,
    robot_sampling_ratios,
)
from reki.utils.dist_utils import is_rank_zero

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

logger = logging.getLogger(__name__)


def build_preprocessor(cfg: TrainConfig) -> Any:
    """The VLM preprocessor, or the external adapter's dataset preprocessor (raw items)."""
    if cfg.external_policy is not None:
        from reki.policy.external import get_external_policy

        return get_external_policy(cfg.external_policy).dataset_preprocessor(cfg)
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    return Qwen3_5Preprocessor.from_pretrained(cfg.vlm_model_id, image_size=cfg.image_size)


def chunk_size_for(cfg: TrainConfig, fps: float) -> int:
    """Action chunk length at ``fps``: ``chunk_duration_s`` of frames, else ``chunk_size``."""
    if cfg.chunk_duration_s is not None:
        return max(1, round(cfg.chunk_duration_s * fps))
    return cfg.chunk_size


def delta_timestamps(
    cfg: TrainConfig, fps: float, chunk_size: int, camera_keys: list[str]
) -> dict[str, list[float]]:
    """LeRobot ``delta_timestamps`` of one leaf: the action chunk of each of the ``history_k``
    frames (``history_stride_s`` apart). With history, also those frames of every camera and,
    with ``state_encoding="adaln"``, of the state."""
    k = cfg.history_k
    past_offsets = [-(k - 1 - i) * cfg.history_stride_s for i in range(k)]
    out = {ACTION: [past + i / fps for past in past_offsets for i in range(chunk_size)]}
    if k > 1:
        for cam_key in camera_keys:
            out[cam_key] = past_offsets
        if cfg.state_encoding == "adaln":
            out[OBS_STATE] = past_offsets
    return out


def load_leaf(name: str, path: Path, cfg: TrainConfig) -> tuple[Any, int]:
    """(LeRobot dataset, action chunk length) of one training leaf."""
    info = load_info(path)
    fps = info["fps"]
    features = info["features"]
    cs = chunk_size_for(cfg, fps)
    spec = DATASETS[name]
    camera_keys = select_top_cameras(spec.cameras, cfg.max_cameras, available=set(features))
    if cfg.mirror_prob > 0.0 and spec.mirror:
        camera_keys = close_camera_pairs(
            camera_keys,
            spec.cameras,
            MIRROR_SPECS[spec.mirror].paired_cameras,
            set(features),
        )
    ds = load_lerobot_dir(
        name,
        path,
        tolerance_s=0.01,
        delta_timestamps=delta_timestamps(cfg, fps, cs, camera_keys),
        video_backend=cfg.video_backend,
    )
    logger.info(
        "Loaded %s (%s) fps=%s chunk=%d len=%d K=%d cams=%d/%d stride=%.2fs",
        name,
        path,
        fps,
        cs,
        len(ds),
        cfg.history_k,
        len(camera_keys),
        sum(ft["dtype"] in ("video", "image") for ft in features.values()),
        cfg.history_stride_s,
    )
    return ds, cs


def make_robotics_dataset(
    cfg: TrainConfig, names: list[str], preprocessor: Any | None = None
) -> tuple[CombinedLeRobotDataset, list[str], list[int]]:
    """The robot dataset over every leaf of ``names`` (registered dataset names).

    Returns ``(dataset, leaf dataset names, leaf sizes)``, in the order of the leaves inside
    the dataset (the per-leaf weighted sampling needs them). ``preprocessor`` defaults to the
    one of ``cfg``.
    """
    leaves = training_leaves(names)
    with thread_safe_tqdm_lock(), ThreadPoolExecutor() as executor:
        loaded = list(executor.map(lambda leaf: load_leaf(leaf[0], leaf[1], cfg), leaves))
    if not loaded:
        raise RuntimeError("No datasets found on disk.")
    datasets = [ds for ds, _ in loaded]
    leaf_names = [name for name, _ in leaves]
    dataset = CombinedLeRobotDataset(
        datasets,
        leaf_names,
        cfg,
        chunk_sizes=[cs for _, cs in loaded],
        preprocessor=build_preprocessor(cfg) if preprocessor is None else preprocessor,
    )
    return dataset, leaf_names, [len(ds) for ds in datasets]


def _check_state_mlp_embodiments(cfg: TrainConfig, names: list[str]) -> None:
    """Check that ``cfg.state_mlp_embodiments`` covers every embodiment the run trains, including
    mirrors, then trim it to them."""
    used = state_embodiments(names) | mirror_state_embodiments(names, cfg.mirror_prob)
    missing = used - cfg.state_mlp_embodiments.keys()
    if missing:
        raise ValueError(
            f"state_encoding={cfg.state_encoding!r} but state_mlp_embodiments is missing "
            f"entries for embodiments used by selected datasets: {sorted(missing)}. "
            f"Add them to TrainConfig.state_mlp_embodiments with their state dim."
        )
    cfg.state_mlp_embodiments = {k: v for k, v in cfg.state_mlp_embodiments.items() if k in used}


def make_combined_dataset(cfg: TrainConfig) -> tuple[CombinedLeRobotDataset, Sampler[int]]:
    """The training dataset of ``cfg`` and its weighted sampler.

    With a state MLP, trims ``cfg.state_mlp_embodiments`` to the used ones.
    """
    names = expand_spec(cfg.dataset)
    preprocessor = build_preprocessor(cfg)
    if cfg.state_encoding == "adaln":
        _check_state_mlp_embodiments(cfg, names)

    robot_ds, leaf_names, sizes = make_robotics_dataset(cfg, names, preprocessor)
    logger.info("Robotics dataset: %d frames", len(robot_ds))
    kept_local = robot_ds.kept_local_frames()
    kept_sizes = robot_ds.kept_frame_counts()
    if kept_sizes != sizes:
        n_drop = sum(f - k for f, k in zip(sizes, kept_sizes))
        logger.info(
            "Curation: %d frames filtered (kept %d/%d)", n_drop, sum(kept_sizes), sum(sizes)
        )
    ratios = robot_sampling_ratios(cfg, leaf_names, kept_sizes)

    sampler = RatioSampler(sizes, ratios, num_samples=sum(kept_sizes), kept_local=kept_local)
    if is_rank_zero():
        print_dataset_mix(leaf_names, sizes, ratios, planned_samples(cfg))
    return robot_ds, sampler
