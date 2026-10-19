"""The inference API imports no LeRobot module: ``pip install reki-vla`` does not install LeRobot."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu


def test_inference_imports_no_lerobot() -> None:
    code = (
        "import sys, reki.inference.policy; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
