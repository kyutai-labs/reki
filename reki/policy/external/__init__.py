"""External-policy adapters. :mod:`reki.policy.external.base` has the contract."""

from __future__ import annotations

from reki.policy.external.base import (
    EXTERNAL_POLICIES,
    ExternalDatasetPreprocessor,
    ExternalPolicy,
    get_external_policy,
    register_external,
)

__all__ = [
    "EXTERNAL_POLICIES",
    "ExternalDatasetPreprocessor",
    "ExternalPolicy",
    "get_external_policy",
    "register_external",
]
