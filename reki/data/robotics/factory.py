"""Build the training and validation datasets of a ``TrainConfig``."""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.datasets.io_utils import load_info
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import ConcatDataset, Sampler

from reki.data.robotics.cameras import close_camera_pairs, select_top_cameras
from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
from reki.data.robotics.lerobot_io import (
    dataset_root,
    find_lerobot_dir,
    load_lerobot_dir,
    training_leaves,
)
from reki.data.robotics.lerobot_patches import thread_safe_tqdm_lock
from reki.data.robotics.mirror import MIRROR_SPECS
from reki.data.robotics.registry import (
    DATASETS,
    RELATIVE_EE,
    VAL_DATASETS,
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
from reki.data.robotics.vl import make_vl_dataset
from reki.utils.dist_utils import is_rank_zero

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

logger = logging.getLogger(__name__)

VIDEO_TOLERANCE_S = 0.01
"""Largest gap (s) between a queried video timestamp and the decoded frame's pts."""

VALIDATION_SETTINGS: dict[str, Any] = dict(
    predict_subtask=False,
    subtask_in_user_stream=False,
    subtask_lookup_filename="subtask_lookup.jsonl",
    sample_outside_intervals=False,
    state_dropout_prob=0.0,
    speed_dropout_prob=0.0,
    embodiment_dropout_prob=0.0,
    fps_dropout_prob=0.0,
    camera_dropout_prob=0.0,
    mirror_prob=0.0,
    pin_overhead_camera=True,
    shuffle_cameras=False,
    drop_extra_side_cameras=False,
    drop_static_wrist_camera=False,
    aug_color_jitter=None,
    aug_random_crop=None,
    aug_random_scale=False,
)
"""Sample options of the validation set, in place of the training config's: no subtask
prediction, no dropout, augmentation, mirroring or camera reduction."""


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
        tolerance_s=VIDEO_TOLERANCE_S,
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


def validate_action_pace(action_pace: dict[str, float], names: list[str]) -> None:
    """Fail loud on pace factors that would silently no-op or resample unsafe reps. A key
    that is not a registered dataset raises; a registered one outside ``names`` is ignored.

    Only ego datasets ("relative end-effector") are supported: their stored ACTION is
    absolute-world wrist, which is resample-safe; delta/joint robot actions are not
    (per-step deltas change meaning under resampling). The pace itself is applied at
    compose time (``interpolate_wrist_chunk``), not at fetch — action delta_timestamps
    stay native-rate.
    """
    for name, pace in action_pace.items():
        if name not in DATASETS:
            raise ValueError(f"action_pace has an entry for {name!r}, not a registered dataset.")
        if name not in names:
            continue
        mode = DATASETS[name].control_mode
        if mode != RELATIVE_EE:
            raise ValueError(
                f"action_pace[{name!r}]: control mode {mode!r} does not support pace "
                f"subsampling (only ego 'relative end-effector' abs-world wrist actions "
                f"are resample-safe)."
            )
        if not 0.0 < pace <= 1.0:
            raise ValueError(f"action_pace[{name!r}]={pace} must be in (0, 1].")


def validate_ego_history(cfg: TrainConfig, names: list[str]) -> None:
    """Fail loud on ego datasets ("relative end-effector") with ``history_k>1`` and no state
    history: each history chunk is expressed in the camera frame of its own frame, whose pose
    is in that frame's state, and only ``state_encoding="adaln"`` fetches the state of every
    frame."""
    if cfg.history_k == 1 or cfg.state_encoding == "adaln":
        return
    ego = [n for n in names if DATASETS[n].control_mode == RELATIVE_EE]
    if ego:
        raise ValueError(
            f"{ego}: history_k={cfg.history_k} with state_encoding={cfg.state_encoding!r} has "
            "no per-frame camera pose for the history chunks; use state_encoding='adaln' or "
            "history_k=1."
        )


def make_robotics_dataset(
    cfg: TrainConfig, names: list[str], preprocessor: Any | None = None
) -> tuple[CombinedLeRobotDataset, list[str], list[int]]:
    """The robot dataset over every leaf of ``names`` (registered dataset names).

    Returns ``(dataset, leaf dataset names, leaf sizes)``, in the order of the leaves inside
    the dataset (the per-leaf weighted sampling needs them). ``preprocessor`` defaults to the
    one of ``cfg``.
    """
    if cfg.action_pace:
        validate_action_pace(cfg.action_pace, names)
    validate_ego_history(cfg, names)
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
    """Check that ``cfg.state_mlp_embodiments`` covers every embodiment the run trains (with
    the mirror and the val set), then trim it to them."""
    used = state_embodiments(names) | mirror_state_embodiments(names, cfg.mirror_prob)
    if cfg.val_dataset in VAL_DATASETS:
        val_emb = DATASETS[cfg.val_dataset].embodiment
        if val_emb and val_emb not in used:
            if not cfg.pretrained_path:
                raise ValueError(
                    f"Val dataset {cfg.val_dataset!r} needs MLP {val_emb!r}, "
                    f"which is neither trained nor loaded from pretrained_path."
                )
            used.add(val_emb)
    missing = used - cfg.state_mlp_embodiments.keys()
    if missing:
        raise ValueError(
            f"state_encoding={cfg.state_encoding!r} but state_mlp_embodiments is missing "
            f"entries for embodiments used by selected datasets: {sorted(missing)}. "
            f"Add them to TrainConfig.state_mlp_embodiments with their state dim."
        )
    cfg.state_mlp_embodiments = {k: v for k, v in cfg.state_mlp_embodiments.items() if k in used}


def make_combined_dataset(
    cfg: TrainConfig,
) -> tuple[CombinedLeRobotDataset | ConcatDataset, Sampler[int]]:
    """The training dataset of ``cfg`` (robot data, plus VL data with ``vl_datasets``) and its
    weighted sampler.

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

    dataset: CombinedLeRobotDataset | ConcatDataset = robot_ds
    all_names, all_sizes, all_kept, all_kept_local = leaf_names, sizes, kept_sizes, kept_local
    if cfg.vl_datasets:
        vl_ds = make_vl_dataset(
            vl_datasets=cfg.vl_datasets,
            preprocessor=preprocessor,
            max_num_qa_pairs=cfg.vl_max_num_qa_pairs,
        )
        logger.info("VL dataset: %d samples", len(vl_ds))
        dataset = ConcatDataset([robot_ds, vl_ds])
        all_names = leaf_names + ["VL"]
        all_sizes = sizes + [len(vl_ds)]
        all_kept = kept_sizes + [len(vl_ds)]
        all_kept_local = kept_local + [None]
        ratios = [ratio * (1.0 - cfg.vl_ratio) for ratio in ratios] + [cfg.vl_ratio]
        logger.info(
            "Combined dataset: %d total (%d robotics + %d VL, ratio=%.2f)",
            len(dataset),
            len(robot_ds),
            len(vl_ds),
            cfg.vl_ratio,
        )
    sampler = RatioSampler(all_sizes, ratios, num_samples=sum(all_kept), kept_local=all_kept_local)
    if is_rank_zero():
        print_dataset_mix(all_names, all_kept, ratios, planned_samples(cfg))
    return dataset, sampler


def make_val_dataset(cfg: TrainConfig, dataset_name: str) -> CombinedLeRobotDataset:
    """The validation set ``dataset_name`` (a ``VAL_DATASETS`` entry): the kept frames of its
    held-out episodes, or of every episode, read and preprocessed as in training with
    ``VALIDATION_SETTINGS``, so val loss is comparable to train loss.
    ``dropped_episodes.json`` is not applied.
    """
    if dataset_name not in VAL_DATASETS:
        raise ValueError(f"Unknown val dataset {dataset_name!r}. Available: {sorted(VAL_DATASETS)}")
    if cfg.action_pace and dataset_name in cfg.action_pace:
        validate_action_pace({dataset_name: cfg.action_pace[dataset_name]}, [dataset_name])
    root = find_lerobot_dir(dataset_name, root=dataset_root(dataset_name))
    info = load_info(root)
    video_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]
    cameras = DATASETS[dataset_name].cameras
    if cameras and video_keys and not set(video_keys) & set(cameras):
        raise ValueError(
            f"DATASETS[{dataset_name!r}].cameras matches none of {sorted(video_keys)}; "
            "stripping would leave the val set image-free"
        )
    base_ds, chunk_size = load_leaf(dataset_name, root, cfg)

    val_cfg = copy.copy(cfg)
    vars(val_cfg).update(VALIDATION_SETTINGS)
    val = CombinedLeRobotDataset(
        [base_ds],
        [dataset_name],
        val_cfg,
        chunk_sizes=[chunk_size],
        preprocessor=build_preprocessor(cfg),
        read_dropped_sidecar=False,
    )
    episodes = VAL_DATASETS[dataset_name]
    kept = val.kept_local_frames()[0]
    if episodes is not None:
        wanted = set(episodes)
        eps = base_ds.meta.episodes
        val.val_frame_indices = [
            i
            for e, f, t in zip(
                eps["episode_index"], eps["dataset_from_index"], eps["dataset_to_index"]
            )
            if int(e) in wanted
            for i in range(int(f), int(t))
            if val.is_kept(i)
        ]
    elif kept is not None:
        val.val_frame_indices = kept.tolist()
    return val
