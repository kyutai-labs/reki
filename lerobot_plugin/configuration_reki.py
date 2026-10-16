"""LeRobot config of an exported reki checkpoint.

The model loads from ``train_config.yaml`` + ``reki_inference.json`` (``reki.RekiPolicy``); this
config declares the LeRobot features of the training dataset and maps its image keys to reki
cameras (``lerobot_policy_reki.export``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NoReturn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


@PreTrainedConfig.register_subclass("reki")
@dataclass
class RekiConfig(PreTrainedConfig):
    """Configuration for running an exported REKI checkpoint through LeRobot."""

    # LeRobot observation image key → reki camera key (reki_inference.json camera_keys)
    camera_map: dict[str, str] = field(default_factory=dict)
    # per-dim names of the dataset state and of the returned action, when the dataset has them
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    # the checkpoint's control rate and actions per predict (reki_inference.json)
    fps: float = 0.0
    n_action_steps: int = 1
    # task stats key, only for a checkpoint with per-task stats
    stats_key: str | None = None
    # Run the expert and the prefill as CUDA graphs (reki.RekiPolicy.from_pretrained)
    compile: bool = True

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> NoReturn:
        raise NotImplementedError("the reki LeRobot plugin is inference-only")

    def get_scheduler_preset(self) -> NoReturn:
        raise NotImplementedError("the reki LeRobot plugin is inference-only")

    def validate_features(self) -> None:
        """Raise unless ``camera_map`` covers exactly the visual input features."""
        visual = {
            k for k, ft in (self.input_features or {}).items() if ft.type is FeatureType.VISUAL
        }
        if set(self.camera_map) != visual:
            raise ValueError(
                f"camera_map keys {sorted(self.camera_map)} != image features {sorted(visual)}"
            )
