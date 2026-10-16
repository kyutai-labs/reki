"""``expand_spec`` names the searched directories of the datasets it cannot find."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def test_missing_datasets_name_their_root(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    from reki.data.robotics.registry import expand_spec

    root = tiny_lerobot_leaf("robotwin2_clean").parents[1]
    assert expand_spec("robotwin") == ["robotwin2_clean"]
    with pytest.raises(FileNotFoundError) as err:
        expand_spec("robotwin2_clean,libero")
    assert str(root / "HuggingFaceVLA") in str(err.value)
    assert "docs/data.md#benchmark-datasets" in str(err.value)
