"""Only decode / I/O failures redraw another sample; any other error raises with its dataset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from av.error import InvalidDataError
from conftest import QWEN
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

pytestmark = pytest.mark.cpu


def _combined(leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch) -> Any:
    from reki.data.robotics.factory import VALIDATION_SETTINGS, make_combined_dataset
    from reki.policy.train_config import TrainConfig
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    leaf("libero", n_episodes=2, n_frames=10)
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    cfg = TrainConfig(
        dataset="libero", vlm_model_id=QWEN, steps=1, batch_size=1, **VALIDATION_SETTINGS
    )
    return make_combined_dataset(cfg)[0]


@pytest.mark.parametrize(
    "error",
    [
        OSError("network filesystem read error"),
        InvalidDataError(1094995529, "Invalid data found when processing input"),
    ],
)
def test_decode_error_is_retried(
    error: Exception, tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _combined(tiny_lerobot_leaf, monkeypatch)
    calls: list[int] = []

    def inner(idx: int) -> dict[str, Any]:
        calls.append(idx)
        if len(calls) == 1:
            raise error
        return {"idx": idx}

    monkeypatch.setattr(ds, "_getitem_inner", inner)
    assert ds[3] == {"idx": calls[1]} and len(calls) == 2


def test_torchcodec_failure_is_a_decode_error(tmp_path: Path) -> None:
    from reki.data.robotics.combined_dataset import DECODE_ERRORS
    from reki.data.robotics.lerobot_patches import _decode_video_frames_torchcodec_uint8

    video = tmp_path / "corrupt.mp4"
    video.write_bytes(b"\x00" * 4096)
    with pytest.raises(DECODE_ERRORS):
        _decode_video_frames_torchcodec_uint8(video, [0.0], 1e-4)


def test_other_error_raises_with_dataset(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _combined(tiny_lerobot_leaf, monkeypatch)

    def inner(idx: int) -> dict[str, Any]:
        raise KeyError("observation.state")

    monkeypatch.setattr(ds, "_getitem_inner", inner)
    with pytest.raises(RuntimeError, match=r"dataset=local/libero, local=13\).*KeyError") as info:
        ds[13]
    assert isinstance(info.value.__cause__, KeyError)


def _vl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import reki.data.text.llavaov_onefive as llava
    from reki.data.robotics.vl import VLSampleAdapter

    rows = [
        {
            "image": None if i == 1 else PILImage.new("RGB", (4, 4)),
            "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
        }
        for i in range(3)
    ]
    features = Features(
        {"image": Image(), "conversations": [{"from": Value("string"), "value": Value("string")}]}
    )
    Dataset.from_list(rows, features=features).save_to_disk(str(tmp_path / "svit"))
    monkeypatch.setattr(llava, "llava_ext_path", lambda: str(tmp_path))
    return VLSampleAdapter([llava.LLava15OV("svit", 1)], preprocessor=None)


def test_vl_null_images_dropped_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vl = _vl(tmp_path, monkeypatch)
    assert len(vl) == 2 and all(r is not None for r in vl._datasets[0].ds["image"])
    calls: list[int] = []

    def flaky(idx: int) -> dict[str, Any]:
        calls.append(idx)
        if len(calls) == 1:
            raise OSError("truncated image")
        return {"idx": idx}

    monkeypatch.setattr(vl, "_getitem_inner", flaky)
    assert vl[0] == {"idx": calls[1]}

    def broken(idx: int) -> dict[str, Any]:
        raise KeyError("conversations")

    monkeypatch.setattr(vl, "_getitem_inner", broken)
    with pytest.raises(RuntimeError, match=r"dataset=svit, local=1\).*KeyError") as info:
        vl[1]
    assert isinstance(info.value.__cause__, KeyError)
