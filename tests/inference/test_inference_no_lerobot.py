"""The inference API imports no LeRobot module: ``pip install reki-vla`` does not install LeRobot."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu


# reki.inference.policy and the modules that RekiPolicy.from_pretrained and predict import
# inside functions, for an exported checkpoint.
INFERENCE_MODULES = [
    "reki.inference.policy",
    "reki.data.robotics.augmentations",
    "reki.policy.checkpoint",
    "reki.utils.dist_utils",
    "reki.vlm.common",
    "reki.vlm.qwen3_5.expert_graph",
    "reki.vlm.qwen3_5.modeling",
    "reki.vlm.qwen3_5.prefill_graph",
]


def test_inference_imports_no_lerobot() -> None:
    code = (
        f"import sys, {', '.join(INFERENCE_MODULES)}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
