"""The external-policy path trains through ``scripts/train.py`` on raw items."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import nn

from reki.policy.external import (
    ExternalDatasetPreprocessor,
    ExternalPolicy,
    register_external,
)
from reki.policy.train_config import TrainConfig

pytestmark = pytest.mark.cpu

DATASET = "robotwin2_clean"


class DummyPreprocessor(ExternalDatasetPreprocessor):
    raw_features = (OBS_STATE, ACTION)

    def preprocess_images(self, images: list) -> torch.Tensor:
        return torch.stack([torch.as_tensor(im, dtype=torch.float32) for im in images])


@register_external
class DummyMLPPolicy(ExternalPolicy):
    name = "dummy_mlp"
    raw_features = (OBS_STATE, ACTION)

    def __init__(self, stats: dict[str, dict[str, torch.Tensor]], chunk_size: int) -> None:
        super().__init__()
        self.state_mean = nn.Buffer(stats[OBS_STATE]["mean"].float())
        self.state_std = nn.Buffer(stats[OBS_STATE]["std"].float().clamp_min(1e-6))
        self.action_mean = nn.Buffer(stats[ACTION]["mean"].float())
        self.action_std = nn.Buffer(stats[ACTION]["std"].float().clamp_min(1e-6))
        self.chunk_size = chunk_size
        self.action_dim = self.action_mean.numel()
        self.net = nn.Sequential(
            nn.Linear(self.state_mean.numel(), 32),
            nn.GELU(),
            nn.Linear(32, chunk_size * self.action_dim),
        )

    @classmethod
    def build(
        cls, cfg: TrainConfig, dataset_stats: dict[str, Any], chunk_size: int
    ) -> DummyMLPPolicy:
        return cls(dataset_stats, chunk_size)

    @staticmethod
    def dataset_preprocessor(cfg: TrainConfig) -> Any:
        return DummyPreprocessor()

    @staticmethod
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            OBS_STATE: torch.stack([it[OBS_STATE] for it in items]),
            ACTION: torch.stack([it[ACTION] for it in items]),
            "task": [it["task"] for it in items],
        }

    def _predict(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        state = (batch[OBS_STATE].float() - self.state_mean) / self.state_std
        target = (batch[ACTION].float() - self.action_mean) / self.action_std
        pred = self.net(state).view(-1, self.chunk_size, self.action_dim)
        return pred, target

    def forward(
        self, batch: dict[str, Any], reduction: str = "mean"
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred, target = self._predict(batch)
        loss = ((pred.float() - target) ** 2).mean()
        return loss, {"flow_loss": loss.detach()}

    def freeze_vision_blocks(self, train_vision_blocks: int) -> None:
        pass

    def enable_gradient_checkpointing(self) -> None:
        pass

    def save_pretrained(self, save_dir: str) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(save_dir) / "dummy_mlp.pt")

    def param_groups(
        self, lr: float, weight_decay: float, backbone_lr_scale: float
    ) -> list[dict[str, Any]]:
        return [{"params": list(self.parameters()), "lr": lr, "weight_decay": weight_decay}]


def _overrides(output_dir: Path) -> dict[str, Any]:
    return dict(
        output_dir=str(output_dir),
        dataset=DATASET,
        external_policy="dummy_mlp",
        history_k=1,
        chunk_duration_s=1,
        max_cameras=2,
        aug_color_jitter=None,
        aug_random_crop=None,
        camera_dropout_prob=0.0,
        state_dropout_prob=0.0,
        batch_size=2,
        steps=2,
        save_steps=2,
        logging_steps=1,
        num_workers=0,
        ema_decay=0.0,
    )


def test_external_config_validation() -> None:
    with pytest.raises(ValueError, match="single-frame"):
        TrainConfig(output_dir="/tmp/x/run", external_policy="dummy_mlp", history_k=3)
    with pytest.raises(KeyError, match="unknown external_policy"):
        TrainConfig(output_dir="/tmp/x/run", external_policy="nope")


def test_external_items_are_raw(tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path) -> None:
    from reki.data.robotics.factory import make_combined_dataset

    tiny_lerobot_leaf(DATASET)
    cfg = TrainConfig(**_overrides(tmp_path / "run"))
    ds, _sampler = make_combined_dataset(cfg)
    item = ds[0]

    assert set(item) >= {OBS_STATE, ACTION, "task", "external_images"}
    assert len(item["external_images"]) == 2
    assert ds.external_chunk_size() == item[ACTION].shape[0]
    raw = ds.datasets[0][0]
    assert torch.equal(item[ACTION], raw[ACTION])
    assert torch.equal(item[OBS_STATE], raw[OBS_STATE])
    assert set(ds.external_policy_stats()) >= {OBS_STATE, ACTION}


def test_external_trains(tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path) -> None:
    from scripts.train import train

    tiny_lerobot_leaf(DATASET)
    out = tmp_path / "run"
    train(**_overrides(out))
    assert (out / "checkpoint-2" / "dummy_mlp.pt").exists()
    train_metrics = [
        metric
        for line in (out / "metrics.jsonl").read_text().splitlines()
        if (metric := json.loads(line)).get("loss") is not None
    ]
    assert set(train_metrics[-1]) >= {
        "step",
        "loss",
        "lr",
        "windows",
        "grad_norm",
        "step_s",
        "data_s",
    }
    assert isinstance(train_metrics[-1]["lr"], float)
