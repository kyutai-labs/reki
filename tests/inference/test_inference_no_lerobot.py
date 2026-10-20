"""The inference API imports no LeRobot module: ``pip install reki-vla`` does not install LeRobot."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu


def test_inference_imports_no_lerobot() -> None:
    # reki.inference.policy, plus the modules RekiPolicy.from_pretrained and predict import lazily.
    modules = (
        "reki.inference.policy",
        "reki.inference.spec",
        "reki.inference.benchmarks",
        "reki.policy.checkpoint",
        "reki.policy.modeling",
        "reki.vlm.qwen3_5.modeling",
        "reki.vlm.qwen3_5.inference",
        "reki.vlm.qwen3_5.prefill_graph",
        "reki.vlm.qwen3_5.expert_graph",
    )
    code = (
        f"import sys, {', '.join(modules)}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
