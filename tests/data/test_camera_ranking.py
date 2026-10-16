"""Unit tests for camera ranking + capping."""

from collections.abc import Callable
from pathlib import Path

import pytest

from reki.data.robotics.cameras import select_top_cameras

pytestmark = pytest.mark.cpu


def test_raw_keys_ranked_by_canonical_target():
    cam_map = {
        "observation.images.top_head": "observation.images.camera_top",
        "observation.images.hand_left": "observation.images.camera_wrist_left",
        "observation.images.hand_right": "observation.images.camera_wrist_right",
        "observation.images.head_left_fisheye": "observation.images.camera_side_1",
    }
    out = select_top_cameras(cam_map, max_cameras=3)
    assert out == [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]


def test_unmapped_cameras(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera missing from ``DatasetSpec.cameras`` is not used; no mapped camera raises."""
    import dataclasses

    from reki.data.robotics import registry
    from reki.data.robotics.lerobot_io import load_lerobot_dir

    leaf = tiny_lerobot_leaf("libero")
    spec = registry.DATASETS["libero"]
    first = dict(list(spec.cameras.items())[:1])
    monkeypatch.setitem(registry.DATASETS, "libero", dataclasses.replace(spec, cameras=first))
    load_lerobot_dir("libero", leaf)
    typo = {"observation.images.typo": "observation.images.camera_top"}
    monkeypatch.setitem(registry.DATASETS, "libero", dataclasses.replace(spec, cameras=typo))
    with pytest.raises(ValueError, match="no camera of the dataset"):
        load_lerobot_dir("libero", leaf)
