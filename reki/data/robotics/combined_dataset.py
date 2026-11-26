"""CombinedLeRobotDataset: one training dataset over many LeRobot leaves.

Each leaf keeps its own layout on disk. A sample is brought to the shared conventions here:
camera keys renamed to the canonical cameras (``DatasetSpec.cameras``), state/action sliced to
the model-visible dims, rotations unified, normalized with the leaf's own stats, gripper sign
unified. Images of different shapes are allowed; ``collate.combined_collate`` keeps them as
lists.
"""

from __future__ import annotations

import bisect
import itertools
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from av.error import FFmpegError
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import Dataset

from reki.data.robotics.augmentations import augment_frames, horizontal_flip
from reki.data.robotics.cameras import (
    CAMERA_KEYS,
    WRIST_LEFT,
    WRIST_RIGHT,
    reduce_cameras,
    take_sample_cameras,
    wrist_static_flags,
)
from reki.data.robotics.mirror import MIRROR_SPECS, MirrorSpec
from reki.data.robotics.normalization import (
    PRETRAIN_STATS_HINT,
    apply_ego_wrist_stats,
    band_clipping_violation,
    check_norm_bands,
    mirror_stats_block,
    normalize_sample,
    prepare_leaf_stats,
    qk_identical_across,
    slice_leaf_stats,
    with_qk_as_q,
)
from reki.data.robotics.prompt import (
    MIRROR_MARKER,
    clean_task_text,
    normalize_task,
    structured_prompt,
)
from reki.data.robotics.registry import (
    DATASETS,
    RELATIVE_EE,
    SINGLE_ARM,
    mirrorable_datasets,
    mirrored_name,
    recenters_state_roll,
)
from reki.data.robotics.subtasks import (
    DEFAULT_LOOKUP,
    interval_success,
    interval_text,
    load_subtask_lookup,
)
from reki.data.robotics.tokenization import FrameTokenizer
from reki.data.robotics.transforms import (
    axisangle_to_euler,
    gather_with_pad,
    negate_gripper_dims,
    recenter_euler_roll,
    rot_blocks,
    tracked_hand_dims,
    untracked_frames,
    wrist_action_in_camera_frame,
)
from reki.policy.constants import ACTION_DIM_VALID, ACTION_IS_PAD
from reki.utils.dist_utils import is_rank_zero

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

logger = logging.getLogger(__name__)

_SAMPLE_KEYS = frozenset(
    {
        ACTION,
        ACTION_IS_PAD,
        OBS_STATE,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        *CAMERA_KEYS,
    }
)
"""The keys of a LeRobot item that a sample keeps (after the camera rename)."""


class _SkipSampleError(Exception):
    """The sampled frame has no camera and must be skipped."""


DECODE_ERRORS: tuple[type[Exception], ...] = (OSError, FFmpegError, FrameTimestampError)
"""Read and decode failures of one sample, retried on another frame: I/O errors,
torchcodec failures (``VideoDecodeError``, an OSError), PyAV errors and LeRobot's frame
timestamp mismatch. Any other exception is a bug or a config error and propagates."""


def item_int(item: dict[str, Any], key: str) -> int:
    """The integer value of an item field, including scalar tensors."""
    value = item[key]
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


class CombinedLeRobotDataset(Dataset):
    """Concatenation of LeRobot leaves; ``names[i]`` is the registered dataset of leaf ``i``.

    The sample options are read from ``cfg`` (a ``TrainConfig``). ``chunk_sizes[i]`` is the
    action chunk length of leaf ``i``. Without ``preprocessor`` an item holds the raw
    ``images`` instead of tokens. ``read_dropped_sidecar=False`` keeps the episodes of
    ``meta/dropped_episodes.json``.
    """

    _MAX_RETRIES = 20

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        names: list[str],
        cfg: TrainConfig,
        *,
        chunk_sizes: list[int],
        preprocessor: Any | None = None,
        read_dropped_sidecar: bool = True,
    ) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        if len(names) != len(datasets) or len(chunk_sizes) != len(datasets):
            raise ValueError(
                f"{len(datasets)} datasets but {len(names)} names, {len(chunk_sizes)} chunk sizes"
            )
        from reki.policy.external.base import ExternalDatasetPreprocessor

        self._external = isinstance(preprocessor, ExternalDatasetPreprocessor)
        self._external_raw: tuple[str, ...] = (
            tuple(preprocessor.raw_features)
            if isinstance(preprocessor, ExternalDatasetPreprocessor)
            else ()
        )
        self._external_cam_keys: list[str] | None = None
        if preprocessor is not None and not self._external and cfg.action_expert_num_registers <= 0:
            raise ValueError(
                f"action_expert_num_registers must be > 0 (got {cfg.action_expert_num_registers!r})"
            )
        self.cfg = cfg
        self.datasets = list(datasets)
        self.dataset_names = list(names)
        self.chunk_sizes = list(chunk_sizes)
        self.val_frame_indices: list[int] | None = None
        self._preprocessor = preprocessor
        self._tokenizer = (
            None
            if preprocessor is None or self._external
            else FrameTokenizer(
                preprocessor,
                subtask_in_user_stream=cfg.subtask_in_user_stream,
                use_camera_role_tokens=cfg.use_camera_role_tokens,
            )
        )

        specs = [DATASETS[n] for n in self.dataset_names]
        if self._external and cfg.mirror_prob > 0.0:
            raise ValueError(
                "external adapters get raw, unmirrored state/action and a single global "
                "normalizer built from unmirrored rows; mirror_prob>0 would negate raw dims and "
                f"swap arm blocks under it (got mirror_prob={cfg.mirror_prob})"
            )
        if self._external and any(s.gripper_flip_action or s.gripper_flip_state for s in specs):
            raise ValueError(
                "external adapters receive raw (unnormalized) state/action, but the gripper "
                "flips are defined in normalized space — negating a raw value flips it about 0 "
                "instead of about the band centre. Datasets needing a flip can't take this "
                f"path yet: {self.dataset_names!r}"
            )
        for name, spec, ds in zip(self.dataset_names, specs, datasets):
            if spec.arm_action_dims is None:
                wrists = {CAMERA_KEYS[WRIST_LEFT], CAMERA_KEYS[WRIST_RIGHT]}
                if cfg.drop_static_wrist_camera and wrists <= set(spec.cameras.values()):
                    raise ValueError(
                        f"{name}: drop_static_wrist_camera needs arm_action_dims for a dataset "
                        "with both wrist cameras; set the (left, right) arm action dims or "
                        "SINGLE_ARM in its DatasetSpec (docs/data.md, onboarding)"
                    )
                continue
            if spec.arm_action_dims == SINGLE_ARM:
                continue
            action_dim = (
                len(spec.action_slice)
                if spec.action_slice is not None
                else ds.meta.features[ACTION]["shape"][-1]
            )
            if not all(spec.arm_action_dims) or max(map(max, spec.arm_action_dims)) >= action_dim:
                raise ValueError(
                    f"{name}: arm_action_dims {spec.arm_action_dims} must be non-empty and index "
                    f"the {action_dim}-dim sliced action"
                )
        unify = cfg.unify_ee_euler_rotation
        self._specs = specs
        self.camera_maps = [s.cameras for s in specs]
        self._control_modes = [s.control_mode for s in specs]
        self._action_paces = [(cfg.action_pace or {}).get(n, 1.0) for n in self.dataset_names]
        self._state_embodiments = (
            [s.embodiment for s in specs] if cfg.state_encoding == "adaln" else None
        )
        self._action_rot_reps = [s.action_rot_rep if unify else None for s in specs]
        self._state_rot_reps = [s.state_rot_rep if unify else None for s in specs]
        self._recenter_roll = [recenters_state_roll(n, cfg) for n in self.dataset_names]
        self._mirror_specs: list[MirrorSpec | None] = [
            MIRROR_SPECS[s.mirror] if cfg.mirror_prob > 0.0 and s.mirror else None for s in specs
        ]
        self._robot_types = [s.robot_type for s in specs]
        self._fps = [ds.meta.fps for ds in self.datasets]
        self._stride_frames = [
            max(1, round(cfg.history_stride_s * ds.meta.fps)) for ds in self.datasets
        ]
        self._cum_sizes = list(itertools.accumulate(len(ds) for ds in self.datasets))

        self._build_stats()
        self._tasks = [_task_lookup(ds, normalize=not self._external) for ds in self.datasets]
        self._load_subtasks(cfg.subtask_lookup_filename, read_dropped_sidecar)
        self._kept: list[np.ndarray | None] = []
        for ds_idx, ds in enumerate(self.datasets):
            kept = kept_frames(
                _episode_bounds(ds),
                self._dropped_eps[ds_idx],
                self._keep_ranges[ds_idx],
                self._subtask_lookup[ds_idx],
                cfg.sample_outside_intervals,
                _untracked(ds) if self._control_modes[ds_idx] == RELATIVE_EE else None,
                self.chunk_sizes[ds_idx],
            )
            self._kept.append(None if len(kept) == len(ds) else kept)
        self._cum_kept = list(itertools.accumulate(self.kept_frame_counts()))
        self._speed_bins = [
            _speed_bins(ds, cfg.speed_bin_width) if cfg.use_speed_metadata else None
            for ds in self.datasets
        ]

    # ------------------------------------------------------------------ construction

    def _build_stats(self) -> None:
        """Per-leaf normalization stats (``self.stats``) and their mirror-world counterpart
        (``self.mirror_stats``), and the checks on them."""
        qcols = (ACTION, OBS_STATE)
        band_violations: list[str] = []
        band_checked: set[str] = set()
        self.stats: list[dict[str, dict[str, torch.Tensor]]] = []
        self.mirror_stats: list[dict[str, dict[str, torch.Tensor]] | None] = []
        for i, ds in enumerate(self.datasets):
            name, spec = self.dataset_names[i], self._specs[i]
            ds_stats = prepare_leaf_stats(
                ds.meta.stats,
                name=name,
                root=Path(ds.root),
                action_rot_rep=self._action_rot_reps[i],
                state_rot_rep=self._state_rot_reps[i],
                recenter_roll=self._recenter_roll[i],
            )
            if self._control_modes[i] == RELATIVE_EE:
                apply_ego_wrist_stats(
                    ds_stats,
                    Path(ds.root),
                    name=name,
                    chunk_duration_s=self.cfg.chunk_duration_s,
                    chunk_size=self.cfg.chunk_size,
                    window=self.chunk_sizes[i],
                    fps=ds.meta.fps,
                    pace=self._action_paces[i],
                )
            for c in qcols:
                if c in ds_stats and not {"qk01", "qk99"} <= ds_stats[c].keys():
                    raise KeyError(
                        f"{ds.root} [{c}]: no qk01/qk99 band. Run, {PRETRAIN_STATS_HINT}"
                        "scripts/data/augment_quantiles.py (see docs/data.md, Normalization)."
                    )
            if "absolute" in self._control_modes[i] and name not in band_checked:
                band_checked.add(name)
                reps = {ACTION: spec.action_rot_rep, OBS_STATE: spec.state_rot_rep}
                band_violations.extend(
                    band_clipping_violation(name, Path(ds.root), ds_stats, qcols, reps)
                )
            # Mirror in the raw layout, then slice: mirroring after the slice moves libero's
            # gripper swap onto a pad dim.
            mirror_spec = self._mirror_specs[i]
            mirror_stats = (
                None
                if mirror_spec is None
                else {k: mirror_stats_block(k, v, mirror_spec) for k, v in ds_stats.items()}
            )
            self.stats.append(slice_leaf_stats(ds_stats, spec.state_slice, spec.action_slice))
            self.mirror_stats.append(
                None
                if mirror_stats is None
                else slice_leaf_stats(mirror_stats, spec.state_slice, spec.action_slice)
            )

        for i, spec in enumerate(self._mirror_specs):
            if spec is None:
                continue
            cameras = set(self.camera_maps[i].values())
            for left, right in spec.paired_cameras:
                has_l, has_r = left in cameras, right in cameras
                if has_l != has_r:
                    raise ValueError(
                        f"mirror_prob>0 on {self.dataset_names[i]!r} keeps only one of the paired "
                        f"cameras ({left}={has_l}, {right}={has_r}); the mirror swaps the pair, "
                        f"so DatasetSpec.cameras must map both cameras of the pair or neither."
                    )
        if self.cfg.mirror_prob > 0.0 and not any(s is not None for s in self._mirror_specs):
            raise ValueError(
                f"mirror_prob={self.cfg.mirror_prob} but no selected dataset is mirrorable "
                f"(supported: {mirrorable_datasets()}); the run would train as if off."
            )
        if self.cfg.state_encoding == "adaln":
            missing = [
                self.datasets[i].repo_id for i, s in enumerate(self.stats) if OBS_STATE not in s
            ]
            if missing:
                raise RuntimeError(
                    f"state_encoding={self.cfg.state_encoding!r} requires OBS_STATE stats, but "
                    f"{len(missing)} dataset(s) have none. Recompute their stats or exclude them:\n"
                    + "\n".join(f"  {r}" for r in missing[:20])
                    + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
                )
        check_norm_bands(
            band_violations, self.dataset_names, self._control_modes, self.stats, qcols
        )

    def _load_subtasks(self, filename: str, read_dropped_sidecar: bool) -> None:
        """Subtask labels, task overrides, keep ranges and dropped episodes of every leaf."""
        cfg = self.cfg
        self._keep_ranges: list[dict[int, list[list[int]]] | None] = []
        self._subtask_lookup: list[dict[int, list[tuple[int, int, str]]] | None] = []
        self._task_override: list[dict[int, str]] = []
        self._dropped_eps: list[set[int]] = []
        self._mistake_intervals: list[dict[int, set[int]] | None] = []
        used_files: list[str | None] = []
        for ds in self.datasets:
            lookup = load_subtask_lookup(Path(ds.root), filename)
            if lookup.keep_ranges is not None:
                logger.info(
                    "Loaded keep_ranges from %s for %s (%d episodes)",
                    lookup.filename,
                    ds.repo_id,
                    len(lookup.keep_ranges),
                )
            self._keep_ranges.append(lookup.keep_ranges)
            self._mistake_intervals.append(lookup.mistake_intervals)
            self._subtask_lookup.append(lookup.intervals)
            self._task_override.append(lookup.tasks or {})
            self._dropped_eps.append(_dropped_episodes(ds, read_dropped_sidecar))
            used_files.append(lookup.filename)

        loaded = [i for i, f in enumerate(used_files) if f is not None]
        if is_rank_zero() and loaded:
            print(lookup_header(filename, len(loaded), len(used_files)))
            per_dataset: dict[str, list[int]] = {}
            for i in loaded:
                stored = set(self._tasks[i].values())
                counts = per_dataset.setdefault(self.dataset_names[i], [0, 0, 0, 0])
                counts[0] += 1
                counts[1] += len(self._subtask_lookup[i] or {})
                counts[2] += sum(1 for t in self._task_override[i].values() if t not in stored)
                counts[3] += len(self._dropped_eps[i])
            for name, (n_leaves, n_eps, n_corrections, n_dropped) in per_dataset.items():
                leaves = f"{n_leaves} leaf" if n_leaves == 1 else f"{n_leaves} leaves"
                extra = f", {n_dropped:,} dropped" if n_dropped else ""
                print(
                    f"  {name:<32} {leaves:<11} "
                    f"({n_eps:,} eps, {n_corrections:,} task corrections{extra})"
                )
        if cfg.predict_subtask:
            check_subtask_coverage(self.dataset_names, used_files)
        if cfg.predict_subtask and filename not in used_files:
            raise FileNotFoundError(
                f"predict_subtask is on, but no dataset in the mix has "
                f"meta/{filename}. Set subtask_lookup_filename to a file your datasets have "
                "(e.g. subtask_lookup.jsonl), or predict_subtask=false. See docs/data.md."
            )

    # ------------------------------------------------------------------ curation

    def kept_local_frames(self) -> list[np.ndarray | None]:
        """Per leaf: the sorted local indices of its kept frames (``kept_frames``), None when
        every frame is kept. ``RatioSampler`` draws only these frames, so the combined index
        space stays unchanged."""
        return self._kept

    def kept_frame_counts(self) -> list[int]:
        """Per leaf: the number of kept frames. The sampler ratios and ``num_samples`` use
        these counts."""
        return [len(ds) if k is None else len(k) for ds, k in zip(self.datasets, self._kept)]

    def is_kept(self, idx: int) -> bool:
        """Whether combined index ``idx`` is a kept frame."""
        ds_idx = bisect.bisect_right(self._cum_sizes, idx)
        kept = self._kept[ds_idx]
        if kept is None:
            return True
        local = idx - (self._cum_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        pos = int(np.searchsorted(kept, local))
        return pos < len(kept) and int(kept[pos]) == local

    def _random_kept_index(self) -> int:
        """A combined index drawn uniformly over the kept frames of all leaves."""
        r = random.randrange(self._cum_kept[-1])
        ds_idx = bisect.bisect_right(self._cum_kept, r)
        local = r - (self._cum_kept[ds_idx - 1] if ds_idx > 0 else 0)
        kept = self._kept[ds_idx]
        offset = self._cum_sizes[ds_idx - 1] if ds_idx > 0 else 0
        return offset + (local if kept is None else int(kept[local]))

    # ------------------------------------------------------------------ items

    def __len__(self) -> int:
        return self._cum_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        assert self.is_kept(idx % len(self)), f"index {idx} is not a kept frame"
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._getitem_inner(idx)
            except _SkipSampleError:
                idx = self._random_kept_index()
            except DECODE_ERRORS as e:
                ds_name, local = self._locate(idx)
                logger.warning(
                    "Decode error idx=%d (dataset=%s, local=%d), retry %d/%d: %s",
                    idx,
                    ds_name,
                    local,
                    attempt + 1,
                    self._MAX_RETRIES,
                    e,
                    exc_info=True,
                )
                idx = self._random_kept_index()
            except Exception as e:
                ds_name, local = self._locate(idx)
                raise RuntimeError(
                    f"Sample idx={idx} (dataset={ds_name}, local={local}) failed: "
                    f"{type(e).__name__}: {e}"
                ) from e
        raise RuntimeError(f"All {self._MAX_RETRIES} retries hit decode errors")

    def _locate(self, idx: int) -> tuple[str, int]:
        """The leaf repo id and local index of combined index ``idx``."""
        ds_idx = bisect.bisect_right(self._cum_sizes, idx)
        ds_name = self.datasets[ds_idx].repo_id if ds_idx < len(self.datasets) else "?"
        offset = self._cum_sizes[ds_idx - 1] if ds_idx > 0 else 0
        return ds_name, idx - offset

    def _getitem_inner(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        ds_idx = bisect.bisect_right(self._cum_sizes, idx)
        if ds_idx >= len(self.datasets):
            raise IndexError(
                f"Index {idx} out of range for CombinedLeRobotDataset with {len(self)} items"
            )
        local = idx - (self._cum_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        raw: dict[str, Any] = self.datasets[ds_idx][local]

        rename = self.camera_maps[ds_idx]
        renamed = {rename.get(k, k): v for k, v in raw.items()}
        item = {k: v for k, v in renamed.items() if k in _SAMPLE_KEYS}

        mirror_spec = self._mirror_specs[ds_idx]
        mirrored = False
        if mirror_spec is not None and random.random() < self.cfg.mirror_prob:
            self._mirror_sample(item, mirror_spec)
            mirrored = True

        self._state_action_to_model_layout(item, ds_idx)
        ds_stats = self.stats[ds_idx]
        if mirrored:
            mirror_stats = self.mirror_stats[ds_idx]
            if mirror_stats is None:
                raise ValueError(
                    f"mirrored draw for dataset {ds_idx} has no mirrored stats "
                    "(construction should have built them for every mirrorable dataset)"
                )
            ds_stats = mirror_stats

        cs_ds = self.chunk_sizes[ds_idx]
        item["chunk_size"] = cs_ds
        spec = self._specs[ds_idx]
        left_static = right_static = False
        if self.cfg.drop_static_wrist_camera:
            left_static, right_static = wrist_static_flags(
                item,
                cs_ds,
                ds_stats,
                arm_action_dims=spec.arm_action_dims,
                control_mode=self._control_modes[ds_idx],
                gripper_dims=spec.gripper_flip_action,
                threshold=self.cfg.wrist_static_threshold,
                dataset_name=self.dataset_names[ds_idx],
            )

        normalize_sample(item, ds_stats, raw=self._external_raw)
        if ACTION in item:
            negate_gripper_dims(item[ACTION], spec.gripper_flip_action)
        if OBS_STATE in item:
            negate_gripper_dims(item[OBS_STATE], spec.gripper_flip_state)

        # History frames before the episode's first subtask are not available.
        ep = item_int(item, "episode_index")
        lookup = self._subtask_lookup[ds_idx]
        first_subtask_start = lookup[ep][0][0] if lookup and lookup.get(ep) else 0
        bounded_fi = max(0, item_int(item, "frame_index") - first_subtask_start)
        k_avail = bounded_fi // self._stride_frames[ds_idx] + 1

        cfg = self.cfg
        per_cam_frames, camera_indices, cam_keys, k_eff = take_sample_cameras(
            item,
            max_cameras=cfg.max_cameras,
            dropout_prob=cfg.camera_dropout_prob,
            pin_overhead=cfg.pin_overhead_camera,
            shuffle=cfg.shuffle_cameras,
            k_avail=k_avail,
        )
        if not per_cam_frames:
            raise _SkipSampleError()
        if cfg.drop_extra_side_cameras or cfg.drop_static_wrist_camera:
            reduce_cameras(
                camera_indices,
                per_cam_frames,
                cam_keys,
                cfg.drop_extra_side_cameras,
                cfg.drop_static_wrist_camera,
                left_static,
                right_static,
            )
        if k_eff > 1:
            per_cam_frames = [frames[-k_eff:] for frames in per_cam_frames]
        images = augment_frames(
            per_cam_frames,
            cam_keys,
            k_eff,
            cfg.aug_color_jitter,
            cfg.aug_random_crop,
            cfg.aug_random_scale,
        )

        state = item.get(OBS_STATE)
        if isinstance(state, torch.Tensor) and state.ndim == 2:
            item.pop(f"{OBS_STATE}_is_pad", None)
            state = state[-max(1, min(state.shape[0], k_avail)) :]
            item[OBS_STATE] = state if k_eff > 1 and cfg.state_encoding == "adaln" else state[-1]

        if self._external:
            return self._build_external_item(item, ds_idx, images, cam_keys)
        if self._preprocessor is not None:
            return self._build_register_group(item, ds_idx, images, camera_indices, k_eff)
        item["images"] = images
        item["camera_indices"] = torch.tensor(camera_indices, dtype=torch.long)
        item["k_eff"] = k_eff
        return item

    def _state_action_to_model_layout(self, item: dict[str, Any], ds_idx: int) -> None:
        """Slice state/action to the model-visible dims, convert an axis-angle rotation to
        euler, and express an egocentric wrist action in the camera frame."""
        spec = self._specs[ds_idx]
        if spec.state_slice is not None:
            item[OBS_STATE] = gather_with_pad(item[OBS_STATE], spec.state_slice)
        if spec.action_slice is not None and ACTION in item:
            item[ACTION] = item[ACTION][..., spec.action_slice]

        action_aa = self._action_rot_reps[ds_idx] is not None
        state_aa = self._state_rot_reps[ds_idx] is not None
        if action_aa or state_aa:
            blocks = rot_blocks(spec.embodiment)
            if action_aa and ACTION in item:
                item[ACTION] = axisangle_to_euler(item[ACTION], blocks)
            if state_aa and OBS_STATE in item:
                item[OBS_STATE] = axisangle_to_euler(item[OBS_STATE], blocks)
                if self._recenter_roll[ds_idx]:
                    recenter_euler_roll(item[OBS_STATE], blocks)

        if (
            self._control_modes[ds_idx] == RELATIVE_EE
            and ACTION in item
            and OBS_STATE in item
            and item[ACTION].dim() == 2
        ):
            item[ACTION] = wrist_action_in_camera_frame(
                item[ACTION], item[OBS_STATE], self._action_paces[ds_idx]
            )
            valid = tracked_hand_dims(item[ACTION])
            if not valid.all():
                item[ACTION_DIM_VALID] = valid

    def _mirror_sample(self, item: dict[str, Any], spec: MirrorSpec) -> None:
        """Mirror one sample in place: state, action chunk, and every camera.

        Call it on canonical keys right after the rename, before slicing, normalization and
        camera capping. The perm and sign are defined on the raw (pre-slice) layout. A camera
        pair swapped after capping could keep a "left wrist" with the unswapped left view. Image
        augmentations run later, on the mirrored sample.

        Every camera h-flips: the reflection moves each camera to the pose of R*'s matching
        camera. The two cameras of a pair also swap (R*'s left wrist camera is the mirror of the
        real right one). A single-wrist embodiment has no pair and only flips.
        """
        for key, feature in ((OBS_STATE, "state"), (ACTION, "action")):
            if key in item:
                item[key] = spec.mirror(item[key], feature)
        for left, right in spec.paired_cameras:
            present = (left in item, right in item)
            if any(present) and not all(present):
                raise ValueError(
                    f"spec {spec.name!r} pairs {left} with {right} but only one is present "
                    f"({present}); a lone member would silently keep the wrong side."
                )
            if all(present) and item[left].shape != item[right].shape:
                raise ValueError(
                    f"spec {spec.name!r} pairs {left} with {right} but they were fetched with "
                    f"different shapes: {tuple(item[left].shape)} != {tuple(item[right].shape)}"
                )
        for key in [k for k in item if k.startswith("observation.images.")]:
            item[key] = horizontal_flip(item[key])
        for left, right in spec.paired_cameras:
            if left in item and right in item:
                item[left], item[right] = item[right], item[left]
        item["is_mirrored"] = True

    # ------------------------------------------------------------------ tokens

    def _robot_type(self, ds_idx: int, mirrored: bool) -> str:
        """Robot type of one sample. A mirrored sample uses ``mirrored_name`` of the real type."""
        real = self._robot_types[ds_idx]
        return mirrored_name(real) if mirrored else real

    def _format_prompt(
        self,
        ds_idx: int,
        task_text: str,
        speed_bin: int | None = None,
        success: bool | None = None,
        mirrored: bool = False,
    ) -> str:
        """The task prompt: verbosity 1 = structured multi-field, 0 = task
        only. Embodiment and fps fields drop out with their dropout probabilities."""
        cfg = self.cfg
        if mirrored:
            task_text = f"{MIRROR_MARKER}{task_text}"
        if cfg.prompt_verbosity != 1:
            return clean_task_text(task_text)
        drop_embodiment = (
            cfg.embodiment_dropout_prob > 0.0 and random.random() < cfg.embodiment_dropout_prob
        )
        drop_fps = cfg.fps_dropout_prob > 0.0 and random.random() < cfg.fps_dropout_prob
        return structured_prompt(
            task_text,
            self._robot_type(ds_idx, mirrored),
            self._fps[ds_idx],
            self._control_modes[ds_idx],
            speed_bin,
            success if cfg.condition_on_mistake else None,
            drop_embodiment=drop_embodiment,
            drop_fps=drop_fps,
        )

    def _build_lm_item(
        self,
        item: dict[str, Any],
        ds_idx: int,
        images: list[torch.Tensor],
        camera_indices: list[int],
        drop_state: bool,
    ) -> None:
        """Build the token item of one frame in place: prompt (with the speed and success
        fields), state embodiment (None when ``drop_state``), subtask turn, labels, region
        tags."""
        cfg = self.cfg
        ep, fi = item_int(item, "episode_index"), item_int(item, "frame_index")
        override = self._task_override[ds_idx].get(ep)
        task_text = override or self._tasks[ds_idx].get(int(item["task_index"].item()), "")

        speed_bin: int | None = None
        speed_bins = self._speed_bins[ds_idx]
        if (
            cfg.use_speed_metadata
            and speed_bins is not None
            and random.random() >= cfg.speed_dropout_prob
        ):
            speed_bin = speed_bins.get(ep)
        success = None
        if cfg.condition_on_mistake:
            success = interval_success(
                self._subtask_lookup[ds_idx], self._mistake_intervals[ds_idx], ep, fi
            )
        mirrored = bool(item.get("is_mirrored"))
        prompt = self._format_prompt(ds_idx, task_text, speed_bin, success, mirrored)

        subtask_text = None
        if cfg.predict_subtask or cfg.subtask_in_user_stream:
            subtask_text = interval_text(self._subtask_lookup[ds_idx], ep, fi)
        if self._state_embodiments is not None:
            embodiment = self._state_embodiments[ds_idx]
            if drop_state or embodiment is None:
                item["state_mlp_embodiment"] = None
            else:
                item["state_mlp_embodiment"] = mirrored_name(embodiment) if mirrored else embodiment
        assert self._tokenizer is not None
        self._tokenizer.tokenize(item, images, camera_indices, prompt, subtask_text)

    def external_policy_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """One global normalizer for an external policy, which has a single normalizer.

        With one leaf, or one qk band shared by all leaves, it is that band. Otherwise the stats
        are recomputed over the rows of all leaves, because a mean of per-leaf quantiles is not
        a quantile. Every leaf must have the same state and action dims.
        """
        if len(self.stats) == 1:
            return with_qk_as_q(self.stats[0])

        if qk_identical_across(self.stats, (OBS_STATE, ACTION)):
            return with_qk_as_q(self.stats[0])

        from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats

        pooled: dict[str, dict[str, torch.Tensor]] = {}
        for key in (OBS_STATE, ACTION):
            dims = {int(st[key]["mean"].numel()) for st in self.stats if key in st}
            if len(dims) != 1:
                raise ValueError(
                    f"external_policy needs one normalizer, so every dataset must share "
                    f"{key} dim; got {sorted(dims)}"
                )
            data = np.concatenate(
                [
                    np.asarray(ds.select_columns([key])[key], dtype=np.float32)
                    for ds in self.datasets
                ]
            )
            raw = get_feature_stats(
                data, axis=0, keepdims=data.ndim == 1, quantile_list=DEFAULT_QUANTILES
            )
            pooled[key] = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in raw.items()}
        return pooled

    def external_chunk_size(self) -> int:
        """Action chunk length for an external adapter. All leaves must share it. Use this, not
        ``cfg.chunk_size``: ``chunk_duration_s`` overrides ``cfg.chunk_size``."""
        sizes = set(self.chunk_sizes)
        if len(sizes) != 1:
            raise ValueError(
                f"external_policy requires a single action-chunk length; got {sorted(sizes)}"
            )
        return sizes.pop()

    def _build_external_item(
        self,
        item: dict[str, Any],
        ds_idx: int,
        images: list[Any],
        kept_cam_keys: list[str],
    ) -> dict[str, Any]:
        """Raw item for an external adapter: decoded RGB at the adapter's resolution,
        unnormalized state/action, task text. Normalization, state encoding and text
        tokenization all belong to the adapter's own pretrained processors."""
        assert self._preprocessor is not None
        ep = item_int(item, "episode_index")
        override = self._task_override[ds_idx].get(ep)
        task_text = override or self._tasks[ds_idx].get(int(item["task_index"].item()), "")
        if len(images) != len(kept_cam_keys):
            raise ValueError(
                f"external path is single-frame: expected one image per camera, got "
                f"{len(images)} images for {len(kept_cam_keys)} cameras"
            )
        if self._external_cam_keys is None:
            self._external_cam_keys = list(kept_cam_keys)
        elif set(kept_cam_keys) != set(self._external_cam_keys):
            raise ValueError(
                f"external items must carry a stable camera set; first item had "
                f"{sorted(self._external_cam_keys)}, this one {sorted(kept_cam_keys)} — a "
                "partial mismatch silently becomes a masked empty camera in the adapter"
            )
        pixels = self._preprocessor.preprocess_images(images)
        return {
            OBS_STATE: item[OBS_STATE],
            ACTION: item[ACTION],
            "task": task_text,
            "external_images": dict(zip(kept_cam_keys, pixels)),
            "chunk_size": item.get("chunk_size"),
        }

    def _build_register_group(
        self,
        item: dict[str, Any],
        ds_idx: int,
        images: list[torch.Tensor],
        camera_indices: list[int],
        k_eff: int,
    ) -> dict[str, Any]:
        """Register memory: one single-frame item per history frame.

        Each frame is an independent VLM segment with its own prompt, images, state row,
        subtask turn and action chunk. History reaches the policy only through the expert-side
        registers. All frames use the same cameras. The state-dropout draw is per frame. The
        subtask and mistake lookups use each frame's own frame_index.
        Returns ``{"group_items": [oldest, ..., anchor]}``. Collate makes consecutive segments
        from it, in frame order."""
        n_cams = len(camera_indices)
        cs_ds = int(item["chunk_size"])
        stride_frames = self._stride_frames[ds_idx]
        fi = item_int(item, "frame_index")
        actions = item.pop(ACTION, None)
        pad = item.pop(ACTION_IS_PAD, None)
        valid = item.pop(ACTION_DIM_VALID, None)
        if actions is not None:
            if actions.shape[0] % cs_ds != 0:
                raise ValueError(
                    f"register memory: ACTION rows {actions.shape[0]} not divisible by "
                    f"chunk_size {cs_ds} (dataset {ds_idx})"
                )
            k_chunks = actions.shape[0] // cs_ds
            if k_chunks < k_eff:
                raise ValueError(
                    f"register memory: {k_chunks} fetched action chunks < k_eff={k_eff}"
                )
            actions = actions.reshape(k_chunks, cs_ds, actions.shape[-1])
            if pad is not None:
                pad = pad.reshape(k_chunks, cs_ds)
            if valid is not None:
                valid = valid.reshape(k_chunks, cs_ds, valid.shape[-1])
        state = item.pop(OBS_STATE, None)
        if state is not None and state.ndim == 2 and state.shape[0] != k_eff:
            raise ValueError(
                f"register memory: per-frame state rows {state.shape[0]} != k_eff={k_eff}"
            )

        p_drop = self.cfg.state_dropout_prob
        members: list[dict[str, Any]] = []
        for f in range(k_eff):
            m = dict(item)
            m["frame_index"] = torch.tensor(max(0, fi - (k_eff - 1 - f) * stride_frames))
            if actions is not None:
                m[ACTION] = actions[-k_eff:][f]
                if pad is not None:
                    m[ACTION_IS_PAD] = pad[-k_eff:][f]
                if valid is not None:
                    m[ACTION_DIM_VALID] = valid[-k_eff:][f]
            if state is not None:
                m[OBS_STATE] = state[f] if state.ndim == 2 else state
            self._build_lm_item(
                m,
                ds_idx,
                images[f * n_cams : (f + 1) * n_cams],
                camera_indices,
                drop_state=p_drop > 0.0 and random.random() < p_drop,
            )
            m["camera_indices"] = torch.tensor(camera_indices, dtype=torch.long)
            m["k_eff"] = 1
            members.append(m)
        return {"group_items": members}


def lookup_header(filename: str, loaded: int, total: int) -> str:
    """The header of the per-dataset subtask lookup table (printed only when a leaf has one)."""
    return (
        f"Subtask lookup per dataset ({loaded}/{total} leaves have one; each reads {filename}"
        + (f", else {DEFAULT_LOOKUP}" if filename != DEFAULT_LOOKUP else "")
        + "):"
    )


def _task_lookup(ds: LeRobotDataset, normalize: bool = True) -> dict[int, str]:
    """Task index → task text (normalized, else stripped), from ``meta.tasks`` (task text as a
    column or as the index)."""
    tasks = getattr(ds.meta, "tasks", None)
    if tasks is None:
        return {}
    text = normalize_task if normalize else str.strip
    if "task" in tasks.columns:
        return {int(row["task_index"]): text(str(row["task"])) for _, row in tasks.iterrows()}
    return {int(row["task_index"]): text(str(idx)) for idx, row in tasks.iterrows()}


def kept_frames(
    bounds: tuple[list[int], list[int], list[int]],
    dropped: set[int],
    keep_ranges: dict[int, list[list[int]]] | None,
    intervals: dict[int, list[tuple[int, int, str]]] | None,
    sample_outside_intervals: bool,
    untracked: np.ndarray | None = None,
    chunk_size: int = 1,
) -> np.ndarray:
    """The sorted leaf frame indices a leaf trains on (sampler and stats).

    ``bounds`` is ``(first frame, end frame, episode index)`` per episode. Excluded: the
    frames of ``dropped`` episodes, the frames whose action chunk (the ``chunk_size`` frames
    from it, within its episode) has only ``untracked`` rows (per leaf frame: an egocentric
    frame with no tracked hand, ``transforms.untracked_frames``) and, unless
    ``sample_outside_intervals``, the frames outside an episode's keep range (inclusive) or
    outside every subtask interval (end-exclusive). An episode without keep range or intervals
    keeps all its frames.
    """
    from_idx, to_idx, ep_indices = bounds
    keep = np.ones(max(to_idx, default=0), dtype=bool)
    for ep, f, t in zip(ep_indices, from_idx, to_idx):
        if ep in dropped:
            keep[f:t] = False
            continue
        if untracked is not None:
            tracked = np.concatenate([[0], np.cumsum(~untracked[f:t])])
            starts = np.arange(t - f)
            keep[f:t] &= tracked[np.minimum(starts + chunk_size, t - f)] > tracked[starts]
        if sample_outside_intervals:
            continue
        ranges = keep_ranges.get(ep) if keep_ranges is not None else None
        if ranges is not None:
            in_range = np.zeros(t - f, dtype=bool)
            for s, e in ranges:
                in_range[max(s, 0) : max(e + 1, 0)] = True
            keep[f:t] &= in_range
        episode_intervals = intervals.get(ep) if intervals is not None else None
        if episode_intervals:
            in_interval = np.zeros(t - f, dtype=bool)
            for s, e, _ in episode_intervals:
                in_interval[max(s, 0) : max(e, 0)] = True
            keep[f:t] &= in_interval
    return np.flatnonzero(keep)


def _untracked(ds: LeRobotDataset) -> np.ndarray:
    """``untracked_frames`` of a leaf's stored action column."""
    column = ds.hf_dataset.data.column(ACTION).combine_chunks()
    return untracked_frames(np.asarray(column.flatten()).reshape(len(column), -1))


def check_subtask_coverage(names: list[str], used_files: list[str | None]) -> None:
    """Every dataset of the mix (``names``: one per leaf) has a subtask lookup in a leaf iff its
    registry entry sets ``subtask_lookup``."""
    has = {n for n, f in zip(names, used_files) if f is not None}
    mix = list(dict.fromkeys(names))
    missing = [n for n in mix if n not in has and DATASETS[n].subtask_lookup]
    stale = [n for n in mix if n in has and not DATASETS[n].subtask_lookup]
    if missing or stale:
        raise ValueError(
            "predict_subtask: "
            + (f"no subtask lookup in {missing}, but their registry entry sets "
               "subtask_lookup=True (build it, or set it to False); " if missing else "")
            + (f"{stale} have a subtask lookup, but their registry entry does not set "
               "subtask_lookup=True" if stale else "")
        )  # fmt: skip


def _dropped_episodes(ds: LeRobotDataset, read_sidecar: bool) -> set[int]:
    """The episodes of ``meta/dropped_episodes.json`` (none without the file or the sidecar)."""
    path = Path(ds.root) / "meta" / "dropped_episodes.json"
    if not read_sidecar or not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def _episode_bounds(ds: LeRobotDataset) -> tuple[list[int], list[int], list[int]]:
    """(first frame, end frame, episode index) of every episode of a leaf."""
    eps = ds.meta.episodes
    from_idx = [int(f) for f in eps["dataset_from_index"]]
    ep_indices = (
        [int(e) for e in eps["episode_index"]]
        if "episode_index" in eps.column_names
        else list(range(len(from_idx)))
    )
    return from_idx, [int(t) for t in eps["dataset_to_index"]], ep_indices


def _speed_bins(ds: LeRobotDataset, bin_width: int) -> dict[int, int]:
    """Episode → speed bin: the episode length in frames over ``bin_width``, rounded."""
    from_idx, to_idx, ep_indices = _episode_bounds(ds)
    return {ep: round((t - f) / bin_width) for ep, f, t in zip(ep_indices, from_idx, to_idx)}
