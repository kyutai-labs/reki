"""The inference spec of a checkpoint: what ``RekiPolicy`` needs beyond the weights (dataset,
embodiment, prompt fields, cameras, chunk and cadence, state/action layout) and the
normalization stats. ``scripts/export_checkpoint.py`` builds and saves it next to the weights
(``reki_inference.json`` + ``reki_stats.safetensors``); for a raw training checkpoint it is
built from the finetuning dataset's ``meta/``, as training builds it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from reki.data.robotics.normalization import quantile_keys
from reki.data.robotics.registry import DATASETS, recenters_state_roll
from reki.data.robotics.transforms import rot_blocks
from reki.inference.benchmarks import BENCHMARKS, benchmark_tasks
from reki.policy.constants import ACTION, OBS_STATE
from reki.policy.train_config import TrainConfig

INFERENCE_SPEC_FILE = "reki_inference.json"
INFERENCE_STATS_FILE = "reki_stats.safetensors"

FeatStats = dict[str, torch.Tensor]
Stats = dict[str, tuple[FeatStats, FeatStats]]
SHARED_STATS_KEY = "shared"


def leaf_norm_stats(name: str, leaf: Path, train_cfg: TrainConfig) -> tuple[FeatStats, FeatStats]:
    """(state, action) stats of one training leaf, exactly as training builds them
    (``CombinedLeRobotDataset``: euler sidecar, slicing), as float32, the dtype
    they are applied in."""
    from reki.data.robotics.lerobot_io import leaf_meta_stats
    from reki.data.robotics.normalization import prepare_leaf_stats, slice_leaf_stats

    spec = DATASETS[name]
    unify = train_cfg.unify_ee_euler_rotation
    stats = prepare_leaf_stats(
        leaf_meta_stats(name, leaf),
        root=leaf,
        action_rot_rep=spec.action_rot_rep if unify else None,
        state_rot_rep=spec.state_rot_rep if unify else None,
        recenter_roll=recenters_state_roll(name, train_cfg),
    )
    stats = slice_leaf_stats(stats, spec.state_slice, spec.action_slice)
    return (
        {k: v.to(torch.float32) for k, v in stats[OBS_STATE].items()},
        {k: v.to(torch.float32) for k, v in stats[ACTION].items()},
    )


def collapse_stats(leaves: list[tuple[FeatStats, FeatStats]]) -> tuple[FeatStats, FeatStats] | None:
    """The one (state, action) band every leaf shares, or None when any leaf's band differs.
    Compares only the fields normalization reads, bitwise; the shared set keeps only them."""
    out = []
    for i in (0, 1):
        keys = quantile_keys(leaves[0][i])
        for leaf in leaves[1:]:
            quantile_keys(leaf[i])
            if not all(torch.equal(leaf[i][k], leaves[0][i][k]) for k in keys):
                return None
        out.append({k: leaves[0][i][k] for k in keys})
    return out[0], out[1]


def trained_leaves(dataset: str, train_cfg: TrainConfig) -> list[Path]:
    """Leaf dirs of ``dataset``, which the checkpoint must have been trained on
    (``train_cfg.dataset``)."""
    from reki.data.robotics.lerobot_io import training_leaves
    from reki.data.robotics.registry import expand_spec

    trained = expand_spec(train_cfg.dataset)
    if dataset not in trained:
        raise ValueError(f"checkpoint was trained on {sorted(trained)}, not {dataset!r}")
    return [p for _, p in training_leaves([dataset])]


def dataset_stats(
    dataset: str,
    leaves: list[Path],
    train_cfg: TrainConfig,
    task_keys: list[tuple[str, Path]] | None = None,
) -> Stats:
    """Normalization stats of ``dataset``, built by the training code for each of its leaves.
    One shared set (``{"shared": ...}``) when all leaves have the same band; otherwise one set
    per leaf, keyed by ``task_keys`` (task name, leaf) or by the leaf directory name."""
    per_leaf = [leaf_norm_stats(dataset, p, train_cfg) for p in leaves]
    shared = collapse_stats(per_leaf)
    if shared is not None:
        return {SHARED_STATS_KEY: shared}
    keys = task_keys if task_keys else [(p.name, p) for p in leaves]
    by_path = dict(zip(leaves, per_leaf, strict=True))
    return {k: by_path[p] for k, p in keys}


@dataclass(frozen=True)
class DatasetFields:
    """Inference fields of one training dataset, derived as training derives them."""

    embodiment: str
    robot_type: str
    control_mode: str
    camera_keys: list[str]
    camera_indices: list[int] | None
    fps: float
    euler_rot_dims: list[int] | None
    state_dim: int
    state_layout: str
    state_names: list[str] | None
    action_names: list[str] | None
    camera_names: list[str]


def feature_names(info: dict[str, Any], key: str) -> list[str] | None:
    """Per-dim names of a ``meta/info.json`` feature, or None when the dataset has none."""
    feat = info["features"][key]
    names = feat.get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]  # RoboTwin stores [[name, ...]]
    if not isinstance(names, list) or len(names) != feat["shape"][0]:
        return None
    return [str(n) for n in names]


def _sliced_action_names(dataset: str, info: dict[str, Any]) -> list[str] | None:
    names = feature_names(info, ACTION)
    if names is None:
        return None
    sl = DATASETS[dataset].action_slice
    return [names[i] for i in sl] if sl is not None else names


def _state_layout(info: dict[str, Any]) -> str:
    """Readable layout of the dataset's ``observation.state``: its feature names."""
    names = feature_names(info, OBS_STATE)
    if names is None:
        names = [f"state[{i}]" for i in range(info["features"][OBS_STATE]["shape"][0])]
    return f"[{', '.join(names)}]"


def dataset_fields(dataset: str, leaves: list[Path], train_cfg: TrainConfig) -> DatasetFields:
    """Embodiment, prompt fields, cameras, fps and state layout of ``dataset``, from the
    registry and the leaves' ``meta/info.json``, as ``CombinedLeRobotDataset`` sets them.
    Leaves that disagree on fps, state dim, robot type or cameras raise."""
    from reki.data.robotics.cameras import CAMERA_KEY_TO_INDEX, select_top_cameras
    from reki.data.robotics.registry import AXIS_ANGLE, canon_robot_type

    if not leaves:
        raise ValueError(f"{dataset}: no leaves")
    if dataset not in DATASETS:
        raise KeyError(f"{dataset}: not in DATASETS; register the dataset (docs/data.md)")
    spec = DATASETS[dataset]
    infos = [json.loads((p / "meta" / "info.json").read_text()) for p in leaves]
    per_leaf = {
        (
            float(i["fps"]),
            i["features"][OBS_STATE]["shape"][0],
            canon_robot_type(i.get("robot_type"), dataset) or str(i.get("robot_type")),
            tuple(
                select_top_cameras(
                    spec.cameras, train_cfg.max_cameras, available=set(i["features"])
                )
            ),
        )
        for i in infos
    }
    if len(per_leaf) != 1:
        raise ValueError(
            f"{dataset}: leaves differ in (fps, state dim, robot type, cameras): {per_leaf}"
        )
    fps, state_dim, robot_type, dataset_cams = next(iter(per_leaf))
    cams = [spec.cameras[k] for k in dataset_cams]
    if not cams:
        raise ValueError(f"{dataset}: no camera of the leaves is in DATASETS[{dataset!r}].cameras")
    embodiment = spec.embodiment
    if embodiment is None and train_cfg.state_encoding == "adaln":
        raise ValueError(f"{dataset}: no state embodiment, but the checkpoint encodes state")
    indices = [CAMERA_KEY_TO_INDEX[c] for c in cams]
    euler_rot_dims = None
    if train_cfg.unify_ee_euler_rotation and spec.action_rot_rep == AXIS_ANGLE:
        blocks = rot_blocks(embodiment)
        if len(blocks) != 1:
            raise ValueError(
                f"{dataset}: decoding {len(blocks)} euler rotation blocks is not supported"
            )
        euler_rot_dims = list(blocks[0])
    return DatasetFields(
        embodiment=embodiment or "",
        robot_type=robot_type,
        control_mode=spec.control_mode,
        camera_keys=[c.rsplit(".", 1)[-1] for c in cams],
        camera_indices=None if indices == list(range(len(indices))) else indices,
        fps=fps,
        euler_rot_dims=euler_rot_dims,
        state_dim=state_dim,
        state_layout=_state_layout(infos[0]),
        state_names=feature_names(infos[0], OBS_STATE),
        action_names=_sliced_action_names(dataset, infos[0]),
        camera_names=[k.rsplit(".", 1)[-1] for k in dataset_cams],
    )


def _fields_and_stats(
    name: str, train_cfg: TrainConfig, benchmark: str | None
) -> tuple[DatasetFields, list[tuple[str, Path]], Stats]:
    leaves = trained_leaves(name, train_cfg)
    fields = dataset_fields(name, leaves, train_cfg)
    tasks = benchmark_tasks(benchmark) if benchmark is not None else []
    return fields, tasks, dataset_stats(name, leaves, train_cfg, tasks)


def history_stream_n_action_steps(
    train_cfg: TrainConfig,
    trained_k: int,
    n_action_steps: int,
    stride_steps: int,
    chunk_size: int,
    tag: str,
) -> int:
    """Predict cadence of a history checkpoint: one predict per history step.

    A model with trained K > 1 was trained on frames ``history_stride_s`` apart, and the register
    memory adds one frame per predict. For it, return ``stride_steps`` and print a message when
    ``n_action_steps`` differs. For trained K == 1, return ``n_action_steps``. A shorter eval
    window of a K > 1 model keeps the trained cadence. Raise when the stride is longer than the
    chunk. Callers cap ``n_action_steps`` at the chunk size before the call.
    """
    if trained_k <= 1:
        return n_action_steps
    if stride_steps > chunk_size:
        raise ValueError(
            f"[{tag}] history eval: history stride ({stride_steps} steps = "
            f"history_stride_s={train_cfg.history_stride_s}s × fps) exceeds the action "
            f"chunk ({chunk_size} steps) — a single chunk can't span one history step. "
            "Raise chunk_duration_s or lower history_stride_s."
        )
    if n_action_steps != stride_steps:
        print(
            f"[{tag}] history eval (trained K={trained_k}): forcing n_action_steps "
            f"{n_action_steps} → {stride_steps} (= history stride, "
            f"history_stride_s={train_cfg.history_stride_s}s) so the window slides one "
            "frame per predict.",
            flush=True,
        )
    return stride_steps


def stats_dim(stats: FeatStats) -> int:
    """The feature dimension of one stats set."""
    return max(v.numel() for v in stats.values())


def select_stats(
    stats: Stats, per_task_stats: bool, stats_key: str | None
) -> tuple[FeatStats, FeatStats]:
    """The (state, action) stats for one call. A shared set takes no key; per-task stats
    need the task name."""
    if not per_task_stats:
        if stats_key is not None:
            raise ValueError(
                f"this checkpoint has one shared stats set; do not pass stats_key ({stats_key!r})"
            )
        return stats[SHARED_STATS_KEY]
    if stats_key is None:
        raise ValueError(
            f"this checkpoint has per-task stats; pass stats_key (LeRobot plugin: "
            f"--policy.stats_key) in {sorted(stats)[:5]}…"
        )
    if stats_key not in stats:
        raise KeyError(f"stats_key {stats_key!r} not in {sorted(stats)[:5]}…")
    return stats[stats_key]


@dataclass
class InferenceSpec:
    """What ``RekiPolicy`` needs beyond the weights; written by the export as
    ``reki_inference.json`` next to ``reki_stats.safetensors``."""

    benchmark: str | None
    """The sim benchmark preset of the export (``reki.inference.benchmarks``), else None."""
    dataset: str
    """The registered training dataset whose conventions (stats, gripper flips) apply."""
    embodiment: str
    """The state-MLP embodiment."""
    robot_type: str
    control_mode: str
    prompt_fps: float
    """Prompt fields: the model was trained with these in its text prompt."""
    chunk_size: int
    """Actions per predicted chunk."""
    n_action_steps: int
    """Actions returned (and executed) per ``predict``: the trained history stride."""
    camera_keys: list[str]
    """The ``images`` keys of ``predict``, in the order the model sees them."""
    camera_indices: list[int] | None
    """Canonical camera index of each camera (None: ``0..n-1``)."""
    state_dim: int
    """Dims of the training dataset's ``observation.state``: the ``predict`` input."""
    state_slice: list[int | None] | None
    """Input state → model state, as training maps it: the registry slice (None: zero pad),
    then axis-angle → euler xyz on ``state_euler_blocks``, then the roll shift by π when
    ``state_recenter_roll``."""
    state_euler_blocks: list[list[int]]
    """[lo, lo + 3) model state blocks that are axis-angle in the input, euler xyz in the model."""
    state_recenter_roll: bool
    """True: the roll of each euler block is shifted by π, wrapped to [-π, π]."""
    action_dim: int
    euler_rot_dims: list[int] | None
    """[a, b]: the model's action dims a:b are euler xyz, returned as axis-angle."""
    state_layout: str
    """Readable layout of the input state."""
    per_task_stats: bool
    """True when the stats are per task: ``predict`` then needs ``stats_key``."""
    stats_keys: list[str]
    tasks: list[str]
    """Task names of a per-task benchmark."""
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    camera_names: list[str] | None = None


def build_inference_spec(
    train_cfg: TrainConfig, *, dataset: str | None = None, benchmark: str | None = None
) -> tuple[InferenceSpec, Stats]:
    """The spec and stats of a checkpoint for ``dataset`` or for a sim ``benchmark`` preset.

    Pass exactly one. ``dataset`` is any registered dataset the checkpoint was trained on. The
    values are the ones ``scripts/eval/rollout.py`` computes at eval time. A preset adds the
    task names, the env fps and a state description. Its other fields must equal
    the ones derived from the dataset. Any ``history_k`` predicts once per ``history_stride_s``,
    as ``scripts/eval/rollout.py`` does by default."""
    if (dataset is None) == (benchmark is None):
        raise ValueError("pass exactly one of dataset= or benchmark=")
    bench = BENCHMARKS[benchmark] if benchmark is not None else None
    name = bench.dataset if bench is not None else dataset
    assert name is not None
    fields, tasks, stats = _fields_and_stats(name, train_cfg, benchmark)
    fps = fields.fps
    state_layout = fields.state_layout
    if bench is not None:
        derived = (fields.embodiment, fields.robot_type, fields.control_mode)
        preset = (bench.embodiment, bench.robot_type, bench.control_mode)
        if derived != preset:
            raise ValueError(f"{benchmark}: registry gives {derived}, preset says {preset}")
        if bench.camera_keys and list(bench.camera_keys) != fields.camera_keys:
            raise ValueError(
                f"{benchmark}: cameras {fields.camera_keys} != preset {list(bench.camera_keys)}"
            )
        fps = bench.fps if bench.fps is not None else fps
        state_layout = bench.state_layout
    if train_cfg.chunk_duration_s is None:
        raise ValueError("the export needs chunk_duration_s in train_config.yaml")
    chunk_size = max(1, round(train_cfg.chunk_duration_s * fps))
    stride_steps = max(1, round(train_cfg.history_stride_s * fps))
    n_action_steps = history_stream_n_action_steps(
        train_cfg, train_cfg.history_k, stride_steps, stride_steps, chunk_size, benchmark or name
    )
    state0, action0 = next(iter(stats.values()))
    for key, (s, a) in stats.items():
        if stats_dim(s) != stats_dim(state0) or stats_dim(a) != stats_dim(action0):
            raise ValueError(f"stats {key!r} dims differ from {next(iter(stats))!r}")
    ds = DATASETS[name]
    state_slice = list(ds.state_slice) if ds.state_slice is not None else None
    if (len(state_slice) if state_slice is not None else fields.state_dim) != stats_dim(state0):
        raise ValueError(f"{name}: model state dim != state stats dim {stats_dim(state0)}")
    euler = train_cfg.unify_ee_euler_rotation and ds.state_rot_rep is not None
    spec = InferenceSpec(
        benchmark=benchmark,
        dataset=name,
        embodiment=fields.embodiment,
        robot_type=fields.robot_type,
        control_mode=fields.control_mode,
        prompt_fps=fps,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        camera_keys=fields.camera_keys,
        camera_indices=fields.camera_indices,
        state_dim=fields.state_dim,
        state_slice=state_slice,
        state_euler_blocks=[list(b) for b in rot_blocks(fields.embodiment)] if euler else [],
        state_recenter_roll=recenters_state_roll(name, train_cfg),
        action_dim=stats_dim(action0),
        euler_rot_dims=fields.euler_rot_dims,
        state_layout=state_layout,
        per_task_stats=SHARED_STATS_KEY not in stats,
        stats_keys=sorted(stats),
        tasks=[t for t, _ in tasks],
    )
    if bench is None:
        for what, names, dim in (
            ("state", fields.state_names, spec.state_dim),
            ("action", fields.action_names, spec.action_dim),
        ):
            if names is not None and len(names) != dim:
                raise ValueError(f"{name}: {len(names)} {what} names for {dim} {what} dims")
        spec.state_names = fields.state_names
        spec.action_names = fields.action_names
        spec.camera_names = fields.camera_names
    return spec, stats


def benchmark_stats(benchmark: str, train_cfg: TrainConfig) -> tuple[float, Stats]:
    """(prompt fps, stats) of a checkpoint finetuned for ``benchmark``."""
    bench = BENCHMARKS[benchmark]
    fields, _, stats = _fields_and_stats(bench.dataset, train_cfg, benchmark)
    fps = bench.fps if bench.fps is not None else fields.fps
    return fps, stats


def save_inference_spec(out_dir: Path, spec: InferenceSpec, stats: Stats) -> None:
    """Write ``reki_inference.json`` and ``reki_stats.safetensors`` into ``out_dir``."""
    tensors = {}
    for key, (state, action) in stats.items():
        for part, feat in (("state", state), ("action", action)):
            for name, t in feat.items():
                tensors[f"{key}/{part}/{name}"] = t.contiguous()
    save_file(tensors, str(out_dir / INFERENCE_STATS_FILE))
    (out_dir / INFERENCE_SPEC_FILE).write_text(json.dumps(asdict(spec), indent=2) + "\n")


def _check_state_map(spec: InferenceSpec, spec_p: Path) -> None:
    """The state slice and euler blocks fit the input and model state dims."""
    sl = spec.state_slice
    if sl is not None and any(i is not None and not 0 <= i < spec.state_dim for i in sl):
        raise ValueError(f"{spec_p}: state_slice {sl} indexes past state_dim {spec.state_dim}")
    model_dim = len(sl) if sl is not None else spec.state_dim
    if any(not (0 <= lo and hi == lo + 3 <= model_dim) for lo, hi in spec.state_euler_blocks):
        raise ValueError(
            f"{spec_p}: state_euler_blocks {spec.state_euler_blocks} must be 3-wide [lo, lo + 3) "
            f"blocks inside the model state dim {model_dim}"
        )
    if spec.state_recenter_roll and not spec.state_euler_blocks:
        raise ValueError(f"{spec_p}: state_recenter_roll needs state_euler_blocks")


def load_inference_spec(ckpt_dir: Path) -> tuple[InferenceSpec, Stats]:
    """Read the spec and stats that ``save_inference_spec`` wrote."""
    spec_p, stats_p = ckpt_dir / INFERENCE_SPEC_FILE, ckpt_dir / INFERENCE_STATS_FILE
    for p in (spec_p, stats_p):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing: export the checkpoint with scripts/export_checkpoint.py "
                "--dataset <name> (or --benchmark <sim>)"
            )
    try:
        spec = InferenceSpec(**json.loads(spec_p.read_text()))
    except TypeError as e:
        raise ValueError(
            f"{spec_p} is from another reki version ({e}): re-export the checkpoint with "
            "scripts/export_checkpoint.py"
        ) from e
    _check_state_map(spec, spec_p)
    flat = load_file(str(stats_p))
    if spec.per_task_stats == (spec.stats_keys == [SHARED_STATS_KEY]):
        raise ValueError(
            f"{spec_p}: per_task_stats={spec.per_task_stats} contradicts stats_keys "
            f"{spec.stats_keys[:3]}"
        )
    stats: Stats = {}
    for key in spec.stats_keys:
        parts: dict[str, dict[str, torch.Tensor]] = {"state": {}, "action": {}}
        for name, t in flat.items():
            k, part, stat = name.rsplit("/", 2)
            if k == key:
                parts[part][stat] = t
        if not parts["state"] or not parts["action"]:
            raise KeyError(f"{stats_p} has no state/action stats for {key!r}")
        stats[key] = (parts["state"], parts["action"])
    return spec, stats


def inference_spec_for(
    ckpt_dir: Path, train_cfg: TrainConfig, dataset: str | None
) -> tuple[InferenceSpec, Stats]:
    """Spec + stats of ``ckpt_dir``: the stored ones for an exported or Hub checkpoint
    (``reki_inference.json``), or, for a raw training checkpoint, the ones the export would
    write, built from the finetuning ``dataset`` (only its ``meta/`` is read)."""
    if (ckpt_dir / INFERENCE_SPEC_FILE).exists():
        if dataset is not None:
            raise ValueError(
                f"{ckpt_dir} has {INFERENCE_SPEC_FILE}; do not pass a dataset ({dataset!r})"
            )
        return load_inference_spec(ckpt_dir)
    if dataset is None:
        raise ValueError(
            f"{ckpt_dir} has no {INFERENCE_SPEC_FILE}: a raw training checkpoint needs its "
            "finetuning dataset (dataset=<name>, --dataset <name>). The pretrained kyutai/reki-2b "
            "has no robot spec: finetune it on your robot first."
        )
    return build_inference_spec(train_cfg, dataset=dataset)
