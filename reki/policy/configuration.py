"""``RekiConfig``: the model hyperparameters saved as config.json."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import torch

from reki.policy.train_config import STATE_MLP_EMBODIMENTS


@dataclass
class RekiConfig:
    """REKI model hyperparameters (Qwen3.5 packed-stream action policy).

    The action expert is a diffusion transformer (DiT). Each block self-attends to its registers,
    its action tokens and the registers of past frames, then cross-attends to the final-layer
    output of the VLM. The memory of past frames lives only in these registers.

    ``from_train_config`` builds the config from a ``TrainConfig``. ``save_pretrained`` writes
    ``to_json_dict`` as config.json. Loading does not read config.json: it builds the config
    from train_config.yaml.
    """

    vlm_model_id: str = "Qwen/Qwen3.5-2B"
    dtype: str = "bfloat16"
    device: str | None = None

    flow_draws: int = 8
    flow_loss_dof_balanced: bool = True

    state_encoding: Literal["none", "adaln"] = "adaln"
    state_mlp_embodiments: dict[str, int] | None = dataclasses.field(
        default_factory=lambda: dict(STATE_MLP_EMBODIMENTS)
    )

    chunk_size: int = 50
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    time_embedding_min_period: float = 4e-3
    time_embedding_max_period: float = 4.0

    train_vision_blocks: int = 4
    train_expert_only: bool = False

    action_expert_hidden_size: int = 768
    action_expert_intermediate_size: int = 3072
    action_expert_num_layers: int = 10
    action_expert_num_heads: int = 12

    history_k: int = 3
    use_camera_role_tokens: bool = True
    action_expert_num_registers: int = 32

    @classmethod
    def from_train_config(cls, train_cfg: object, **overrides: object) -> "RekiConfig":
        """Build the config from a TrainConfig: copy every field with the same name, then apply
        ``overrides`` (e.g. device).

        Training, inference and evaluation all use this method. Fields only in this class keep
        their defaults. Fields only in TrainConfig (data pipeline) are ignored.
        """
        shared = {
            f.name: getattr(train_cfg, f.name)
            for f in dataclasses.fields(cls)
            if f.init and hasattr(train_cfg, f.name)
        }
        shared.update(overrides)
        return cls(**shared)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _valid = {"none", "adaln"}
        if self.state_encoding not in _valid:
            raise ValueError(
                f"state_encoding must be one of {_valid!r}, got {self.state_encoding!r}"
            )

        if self.action_expert_hidden_size % self.action_expert_num_heads != 0:
            raise ValueError(
                f"action_expert_hidden_size={self.action_expert_hidden_size} must be "
                f"divisible by action_expert_num_heads={self.action_expert_num_heads} "
                "(head_dim = hidden // heads)."
            )

        if self.history_k < 1:
            raise ValueError(f"history_k must be >= 1, got {self.history_k!r}.")
        if self.action_expert_num_registers <= 0:
            raise ValueError(
                "action_expert_num_registers must be > 0 (got "
                f"{self.action_expert_num_registers!r}); the expert always runs register memory."
            )

        if self.state_encoding == "adaln" and not self.state_mlp_embodiments:
            raise ValueError(
                "state_encoding='adaln' requires state_mlp_embodiments "
                "(per-embodiment state dims for the pooled cond MLP)."
            )

    def to_json_dict(self) -> dict[str, Any]:
        """The JSON-serializable model configuration."""
        return {"type": "reki", **dataclasses.asdict(self)}
