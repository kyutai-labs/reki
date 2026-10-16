"""Processor pipelines of the reki plugin.

``reki.RekiPolicy`` does its own state transform, normalization, prompt and action decode, so
both pipelines do almost nothing.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from lerobot_policy_reki.configuration_reki import RekiConfig


def make_reki_pre_post_processors(
    config: RekiConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """The (pre, post) pipelines. ``dataset_stats`` is unused: the policy normalizes with the
    stats of the checkpoint."""
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
