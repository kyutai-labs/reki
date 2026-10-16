"""LeRobot policy wrapper around ``reki.RekiPolicy`` (inference only).

``select_action`` keeps an action queue and predicts only when it is empty, so one predict
runs every ``n_action_steps`` (= history_stride_s × fps) env steps: the register-memory cadence
the model was trained with.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from torch import Tensor

from lerobot_policy_reki.configuration_reki import RekiConfig
from reki.inference.policy import RekiPolicy as _RekiPolicy


def to_uint8_hwc(img: Tensor) -> np.ndarray:
    """LeRobot's float CHW image in [0, 1] → the uint8 HWC frame the env rendered."""
    return (img.permute(1, 2, 0).float() * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


class RekiPolicy(PreTrainedPolicy):
    """A REKI inference policy exposed through LeRobot's policy interface."""

    config_class = RekiConfig
    name = "reki"

    def __init__(self, config: RekiConfig, reki_policy: _RekiPolicy | None = None, **kwargs: Any):
        super().__init__(config)
        if reki_policy is None:
            raise ValueError("reki is inference-only: load a checkpoint with --policy.path")
        self.reki = reki_policy
        self.model = reki_policy.policy
        self._queue: deque[Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> RekiPolicy:
        """Load through reki's own loader (train_config.yaml + reki_inference.json + weights)."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, revision=revision)
        assert isinstance(config, RekiConfig)
        # lerobot-eval turns TF32 on; reki eval runs fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = False
        inner = _RekiPolicy.from_pretrained(
            pretrained_name_or_path,
            device=config.device or "cuda",
            compile=config.compile,
            revision=revision,
        )
        spec = (inner.spec.n_action_steps, inner.spec.prompt_fps)
        if (config.n_action_steps, config.fps) != spec:
            raise ValueError(
                f"config.json (n_action_steps, fps) = {(config.n_action_steps, config.fps)} but "
                f"reki_inference.json gives {spec}: rerun lerobot_policy_reki.export"
            )
        return cls(config, inner).eval()

    def reset(self) -> None:
        self.reki.reset()
        self._queue.clear()

    def get_optim_params(self) -> dict:
        raise NotImplementedError("the reki LeRobot plugin is inference-only")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("the reki LeRobot plugin is inference-only")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """[1, n_action_steps, action_dim] env-ready actions (unnormalized, native rotation)."""
        if len(batch["task"]) != 1:
            raise ValueError("batched inference is not supported yet: use --eval.batch_size=1")
        images = {}
        for key, camera in self.config.camera_map.items():
            img = batch[key][0]
            want = tuple(self.config.input_features[key].shape)
            if tuple(img.shape) != want:
                raise ValueError(f"{key}: got {tuple(img.shape)}, the checkpoint expects {want}")
            images[camera] = to_uint8_hwc(img)
        state = batch[OBS_STATE][0].float().cpu().numpy()
        actions = self.reki.predict(
            images, state, batch["task"][0], stats_key=self.config.stats_key
        )
        return torch.from_numpy(actions)[None]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """The next queued action, predicting a new chunk when needed."""
        if not self._queue:
            self._queue.extend(self.predict_action_chunk(batch)[0])
        return self._queue.popleft()[None]
