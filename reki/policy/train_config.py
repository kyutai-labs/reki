"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

import yaml

from reki.utils.cli import fire_bool

STATE_MLP_EMBODIMENTS: dict[str, int] = {
    "g2a": 24,
    "franka": 8,
    "ur5e": 7,
    "single_arm_ee_euler": 8,
    "single_arm_ee_quat": 8,
    "bimanual_joint": 16,
    "bimanual_dexterous": 26,
    "agilex": 14,
    "r1lite": 17,
    "aloha": 14,
    "aloha_mirrored": 14,
    "single_arm_ee_euler_mirrored": 8,
    "panda_omron_mirrored": 16,
    "yam": 14,
    "gr1": 29,
    "panda_omron": 16,
    "airbot_mmk2": 24,
    "unitree_g1": 28,
    "so101": 6,
}


@dataclass
class TrainConfig:
    """All training options, set from a yaml config and CLI overrides."""

    output_dir: str = ""

    vlm_model_id: str = "Qwen/Qwen3.5-2B"

    dataset: str | list[str] | tuple[str, ...] = "libero"
    chunk_size: int = 50
    chunk_duration_s: float | None = 1.0
    dataset_weighting: Literal["power_law", "manual"] = "power_law"
    frames_alpha: float = 0.5
    weight_overrides: dict[str, float] = field(default_factory=dict)
    max_epochs_per_dataset: int = 100

    video_backend: Literal["pyav", "torchcodec"] = "torchcodec"

    steps: int = 100_000
    batch_size: int = 30
    num_workers: int = 12
    seed: int = 1000
    gradient_accumulation_steps: int = 1

    lr: float = 1e-4
    min_lr: float = 2e-6
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_schedule: Literal["cosine", "wsd"] = "wsd"
    decay_steps: int = 0
    ema_decay: float | None = 0.999

    max_text_tokens: int | None = 30000

    pretrained_path: str | None = None
    train_expert_only: bool = False
    train_vision_blocks: int = 4
    backbone_lr_scale: float = 1.0
    gradient_checkpointing: bool = True
    max_image_tokens: int | None = None
    image_size: int = 448

    external_policy: str | None = None

    aug_color_jitter: float | None = 0.2
    aug_random_crop: float | None = None
    aug_random_scale: bool = False

    mirror_prob: float = 0.0

    flow_draws: int = 8
    flow_loss_dof_balanced: bool = True
    action_expert_hidden_size: int = 768
    action_expert_intermediate_size: int = 3072
    action_expert_num_layers: int = 10
    action_expert_num_heads: int = 12

    state_encoding: Literal["none", "adaln"] = "adaln"
    unify_ee_euler_rotation: bool = True
    prompt_verbosity: int = 1
    state_mlp_embodiments: dict[str, int] = field(
        default_factory=lambda: dict(STATE_MLP_EMBODIMENTS)
    )
    state_dropout_prob: float = 0.20

    history_k: int = 3
    history_stride_s: float = 1.0
    max_cameras: int | None = None
    camera_dropout_prob: float = 0.2
    pin_overhead_camera: bool = True
    shuffle_cameras: bool = True
    use_camera_role_tokens: bool = True
    action_expert_num_registers: int = 32

    ddp_find_unused_parameters: bool = True

    ddp_static_graph: bool = False
    ddp_bucket_cap_mb: int | None = 256
    optimizer_master_weights: bool = True

    use_speed_metadata: bool = True
    speed_bin_width: int = 100
    speed_dropout_prob: float = 0.20
    embodiment_dropout_prob: float = 0.15
    fps_dropout_prob: float = 0.20

    save_steps: int = 5000
    logging_steps: int = 20
    save_optimizer_state: bool = True

    def __post_init__(self):
        if isinstance(self.dataset, tuple):
            self.dataset = list(self.dataset)
        if self.max_cameras is not None and self.max_cameras < 1:
            raise ValueError(f"max_cameras must be >= 1 or None, got {self.max_cameras!r}.")
        if self.prompt_verbosity not in (0, 1):
            raise ValueError(f"prompt_verbosity must be 0 or 1, got {self.prompt_verbosity!r}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps!r}"
            )
        if self.external_policy is not None:
            from reki.policy.external import get_external_policy

            get_external_policy(self.external_policy)
            if self.history_k != 1:
                raise ValueError(
                    f"external_policy={self.external_policy!r} is single-frame "
                    f"(no history); got history_k={self.history_k!r}."
                )
        if not 0.0 <= self.mirror_prob <= 1.0:
            raise ValueError(f"mirror_prob must be in [0, 1], got {self.mirror_prob!r}")
        if self.mirror_prob > 0.0:
            if self.state_encoding == "none":
                raise ValueError(
                    "mirror_prob>0 with state_encoding='none' gives the mirror no separate state "
                    "MLP, so the mirror-image robot and the real one become one embodiment — "
                    "the isolation the mirror relies on. Use 'adaln'."
                )
        if self.state_encoding == "adaln" and not self.state_mlp_embodiments:
            raise ValueError(
                "state_encoding='adaln' requires state_mlp_embodiments (the pooled "
                "per-embodiment state MLP is the cond source)."
            )
        if self.ema_decay is None:
            self.ema_decay = 0.999
        from reki.data.robotics.augmentations import (
            AUG_COLOR_JITTER_DEFAULT,
            AUG_CROP_FRACTION_DEFAULT,
        )

        if isinstance(self.aug_color_jitter, bool):
            self.aug_color_jitter = AUG_COLOR_JITTER_DEFAULT if self.aug_color_jitter else None
        if isinstance(self.aug_random_crop, bool):
            self.aug_random_crop = AUG_CROP_FRACTION_DEFAULT if self.aug_random_crop else None

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def _check_kwargs(cls, data: object, path: str | Path) -> dict:
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{path}: unknown TrainConfig keys {unknown}")
        return data

    @classmethod
    def load_yaml_kwargs(cls, path: str | Path) -> dict:
        """Strict yaml → kwargs dict: any key that is not a TrainConfig field raises."""
        with open(path) as f:
            return cls._check_kwargs(yaml.safe_load(f), path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """A training configuration from a strict YAML mapping."""
        return cls(**cls.load_yaml_kwargs(path))

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path) -> "TrainConfig":
        """The ``train_config.yaml`` of a checkpoint dir (or of its run-level parent)."""
        path = checkpoint_config_path(ckpt_dir)
        if path is None:
            raise FileNotFoundError(f"train_config.yaml not in {ckpt_dir} or parent")
        return cls(**cls._check_kwargs(read_checkpoint_config(path), path))

    @classmethod
    def from_config_and_overrides(cls, config: str | Path | None, overrides: dict) -> "TrainConfig":
        """A configuration from an optional strict YAML file and explicit field overrides.
        ``--help``/``-h`` prints the fields (with ``config`` applied), then exits."""
        kwargs = cls.load_yaml_kwargs(config) if config else {}
        if overrides.get("help") or overrides.get("h"):
            print(f"TrainConfig fields (--name=value), values from {config or 'the defaults'}:")
            cls(**kwargs).print()
            print("Script flags (--config, ...): pass `-- --help`.")
            raise SystemExit(0)
        unknown = sorted(set(overrides) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown TrainConfig overrides {unknown}")
        kwargs.update(overrides)
        for f in fields(cls):
            if f.type == "bool" and f.name in overrides:
                kwargs[f.name] = fire_bool(f.name, overrides[f.name])
        return cls(**kwargs)

    def print(self) -> None:
        for k, v in self.to_dict().items():
            print(f"  {k}: {v}")


def checkpoint_config_path(ckpt_dir: str | Path) -> Path | None:
    """``train_config.yaml`` inside a local checkpoint dir, else its run-level parent."""
    p = Path(ckpt_dir)
    return next(
        (c for c in (p / "train_config.yaml", p.parent / "train_config.yaml") if c.exists()), None
    )


def read_checkpoint_config(path: str | Path) -> dict:
    """A checkpoint's ``train_config.yaml`` as a dict (not checked against the TrainConfig
    fields)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
    return data
