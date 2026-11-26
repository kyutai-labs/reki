"""Resume raises when the checkpoint's optimizer state (ZeRO-1 or plain ``optimizer*.bin``) is
not the format of the run's optimizer: Accelerate would skip it and restart the Adam moments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.distributed.optim import ZeroRedundancyOptimizer

from scripts.train import ZERO1_OPTIMIZER, _load_checkpoint

pytestmark = pytest.mark.cpu


def _load(ckpt: Path, optimizer: Any) -> None:
    _load_checkpoint(cast(Any, None), cast(Any, None), optimizer, {}, str(ckpt))


def test_zero1_state_into_plain_optimizer_raises(tmp_path: Path) -> None:
    (tmp_path / ZERO1_OPTIMIZER).touch()
    plain = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(ValueError, match="use_zero1=True"):
        _load(tmp_path, plain)


def test_plain_state_into_zero1_optimizer_raises(tmp_path: Path) -> None:
    (tmp_path / "optimizer.bin").touch()
    zero1 = object.__new__(ZeroRedundancyOptimizer)  # isinstance only: no process group
    with pytest.raises(ValueError, match="use_zero1=False"):
        _load(tmp_path, zero1)
