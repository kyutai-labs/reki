"""Adapters for external policies with the REKI training loop (no adapter ships here). The
data, split, batch, schedule and optimizer are the same as for REKI. Only the model changes.

To add a model, add one file here with a ``@register_external`` class and import it in
``__init__.py``.

The dataset gives the adapter raw state/action and decoded RGB. The adapter applies the
normalization, tokenization and resizing of its pretrained model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig


class ExternalPolicy(nn.Module, ABC):
    """The methods that the training loop calls on an external policy."""

    name: ClassVar[str]
    raw_features: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def build(
        cls, cfg: TrainConfig, dataset_stats: dict[str, Any], chunk_size: int
    ) -> ExternalPolicy:
        """Build the policy from ``cfg``, the stats of the one source dataset, and the chunk
        length of the dataset (from ``chunk_duration_s``)."""

    @staticmethod
    @abstractmethod
    def dataset_preprocessor(cfg: TrainConfig) -> Any:
        """The object that makes item fields from decoded frames and task text: an
        ``ExternalDatasetPreprocessor`` with the methods of ``Qwen3_5Preprocessor`` that the
        dataset calls."""

    @staticmethod
    @abstractmethod
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        """Batched tensors. External models do not use REKI's packed BS=1 layout."""

    @abstractmethod
    def forward(
        self, batch: dict[str, Any], reduction: str = "mean"
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The training loss and its named components."""

    @abstractmethod
    def freeze_vision_blocks(self, train_vision_blocks: int) -> None:
        """-1: all of the vision tower trains. 0: frozen. N: only the last N blocks train."""

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """Turn on gradient checkpointing in the adapter's model."""

    @abstractmethod
    def save_pretrained(self, save_dir: str) -> None:
        """Write a checkpoint in the native format of the external model. ``build`` reads
        it from ``cfg.pretrained_path``."""

    @abstractmethod
    def param_groups(
        self, lr: float, weight_decay: float, backbone_lr_scale: float
    ) -> list[dict[str, Any]]:
        """Optimizer groups. If the model has a backbone and an expert, use the same split
        and learning rates as REKI."""


class ExternalDatasetPreprocessor:
    """Base class of the dataset side of an adapter. With a preprocessor of this type,
    ``CombinedLeRobotDataset`` gives raw items (decoded RGB, unnormalized state/action)
    and does not apply REKI's VLM preprocessing. The features in ``raw_features`` are not
    normalized."""

    raw_features: tuple[str, ...] = ()

    def preprocess_images(self, images: list) -> "torch.Tensor":
        """The external policy's input tensor for decoded images."""
        raise NotImplementedError


EXTERNAL_POLICIES: dict[str, type[ExternalPolicy]] = {}
_PolicyT = TypeVar("_PolicyT", bound=type[ExternalPolicy])


def register_external(cls: _PolicyT) -> _PolicyT:
    """Register an external-policy adapter by its class-level name."""
    EXTERNAL_POLICIES[cls.name] = cls
    return cls


def get_external_policy(name: str) -> type[ExternalPolicy]:
    """The registered adapter class for ``name``."""
    if name not in EXTERNAL_POLICIES:
        raise KeyError(f"unknown external_policy={name!r}; registered: {sorted(EXTERNAL_POLICIES)}")
    return EXTERNAL_POLICIES[name]
