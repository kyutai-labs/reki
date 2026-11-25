"""`load_vl_config` reads parquet and the three Arrow layouts to the same rows and decoded
images; only parquet writes to the HF datasets cache. A mixed or empty dir raises."""

from __future__ import annotations

import shutil
from pathlib import Path

import datasets
import numpy as np
import pytest
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from reki.data.utils import load_vl_config

pytestmark = pytest.mark.cpu


def _tiny(n: int) -> Dataset:
    rows = [
        {
            "images": [PILImage.fromarray(np.full((4, 4, 3), i, dtype=np.uint8))],
            "texts": [{"user": f"q{i}", "assistant": f"a{i}"}],
        }
        for i in range(n)
    ]
    features = Features(
        {
            "images": [Image()],
            "texts": [{"user": Value("string"), "assistant": Value("string")}],
        }
    )
    return Dataset.from_list(rows, features=features)


def _arrow_file(ds: Dataset, tmp: Path, dst: Path) -> None:
    """Place the single arrow file of a `save_to_disk` dump at `dst`."""
    ds.save_to_disk(str(tmp))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(next(tmp.glob("*.arrow")), dst)


def _check(loaded: Dataset, ref: Dataset) -> None:
    assert len(loaded) == len(ref)
    assert loaded["texts"] == ref["texts"]
    img = loaded[2]["images"][0]
    assert isinstance(img, PILImage.Image)
    assert np.asarray(img)[0, 0, 0] == 2


def test_layouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_cache")
    ref = _tiny(5)
    _arrow_file(ref, tmp_path / "dump0", tmp_path / "shards" / "data-00000-of-00001.arrow")
    ref.save_to_disk(str(tmp_path / "disk"))
    cfg = tmp_path / "builder" / "0.0.0" / "h"
    _arrow_file(ref, tmp_path / "dump", cfg / "fine_vision-train.arrow")
    _arrow_file(_tiny(3), tmp_path / "decoy", cfg / "cache-x.arrow")
    for name in ("shards", "disk", "builder"):
        _check(load_vl_config(str(tmp_path), name), ref)
    assert not (tmp_path / "hf_cache").exists()
    ref.to_parquet(str(tmp_path / "pq" / "train-00000-of-00001.parquet"))
    _check(load_vl_config(str(tmp_path), "pq"), ref)
    assert (tmp_path / "hf_cache").exists()


def test_mixed_formats_raise(tmp_path: Path) -> None:
    _tiny(2).to_parquet(str(tmp_path / "mix" / "train-00000-of-00001.parquet"))
    _arrow_file(_tiny(2), tmp_path / "dump", tmp_path / "mix" / "data-00000-of-00001.arrow")
    with pytest.raises(ValueError, match="both parquet and Arrow"):
        load_vl_config(str(tmp_path), "mix")


def test_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="empty.*pretrain_data.py build"):
        load_vl_config(str(tmp_path), "empty")
    for h in ("h1", "h2"):
        (tmp_path / "two" / "0.0.0" / h).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="found 2"):
        load_vl_config(str(tmp_path), "two")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_vl_config(str(tmp_path), "missing")


def test_alternating_rows_matches_formatter() -> None:
    from reki.data.text.llava_conversations import alternating_rows, format_llava_conversations

    def turn(key: str, speaker: str) -> dict[str, str | None]:
        other = "role" if key == "from" else "from"
        return {key: speaker, other: None, "value": "t", "content": "t"}

    rows = [
        [turn("from", "human"), turn("from", "gpt")],
        [turn("role", "user"), turn("role", "assistant"), turn("role", "user")],
        [turn("role", "assistant")],
        [],
        [turn("from", "human"), turn("from", "human")],
        [turn("role", "user"), turn("role", "assistant")],
    ]
    import pyarrow as pa

    table = pa.chunked_array([pa.array(rows[:3]), pa.array(rows[3:])])
    expected = []
    for row in rows:
        try:
            format_llava_conversations(row, None)
            expected.append(bool(row))
        except (AssertionError, IndexError):
            expected.append(False)
    assert alternating_rows(table).tolist() == expected == [True, True, False, False, False, True]
