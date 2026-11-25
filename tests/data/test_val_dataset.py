"""``make_val_dataset`` reads the val leaf from ``$REKI_DATASETS_ROOT``, as training does."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import QWEN

pytestmark = pytest.mark.cpu


def test_val_reads_datasets_root(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from reki.data.robotics.factory import make_val_dataset
    from reki.policy.train_config import TrainConfig
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    leaf = tiny_lerobot_leaf("libero")
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    cfg = TrainConfig(dataset="libero", vlm_model_id=QWEN, val_dataset="libero")
    val: Any = make_val_dataset(cfg, "libero")
    assert [Path(ds.root) for ds in val.datasets] == [leaf]
