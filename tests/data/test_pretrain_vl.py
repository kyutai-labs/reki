"""VL readers of the pretraining pipeline (``pretrain.readers.vl``) and the default VL paths.
Synthetic fixtures, no network, no data."""

from __future__ import annotations

import io
import json
import logging
import tarfile
import urllib.error
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pretrain.common.download import upstream_dir
from pretrain.pins import HfHead, HfSnapshot
from pretrain.readers import vl

from reki import paths

pytestmark = pytest.mark.cpu

SOURCE = HfSnapshot("org/vl", "a" * 40)
VL_VARS = ("FINEVISION_PATH", "LLAVA_EXT_PATH", "VL_SPATIAL_PARQUET_PATH")


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No VL path overrides; the HF datasets cache in ``tmp_path``."""
    for var in VL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_cache")


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _rows(out: Path) -> list[dict]:
    return [r for f in vl.shard_files(out) for r in vl.read_shard(f).to_pylist()]


def test_vl_paths_default_under_the_datasets_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REKI_DATASETS_ROOT", str(tmp_path))
    assert paths.finevision_path() == str(tmp_path / "vl" / "finevision")
    assert paths.llava_ext_path() == str(tmp_path / "vl" / "llava_onevision")
    assert paths.vl_spatial_parquet_path() == str(tmp_path / "vl" / "spatial")
    monkeypatch.setenv("FINEVISION_PATH", "/fv/")
    monkeypatch.setenv("LLAVA_EXT_PATH", "/llava/")
    monkeypatch.setenv("VL_SPATIAL_PARQUET_PATH", "/spatial")
    assert (paths.finevision_path(), paths.llava_ext_path(), paths.vl_spatial_parquet_path()) == (
        "/fv/",
        "/llava/",
        "/spatial",
    )


# ------------------------------------------------------------------ RoboPoint / RefSpatial


def _tarball(path: Path, names: list[str]) -> None:
    """A .tar.gz of ``names`` (PNG bytes) and a last, incompressible member, split into two
    ``.part_*`` files: the first ends inside the last member."""
    buf = io.BytesIO()
    pad = np.random.default_rng(0).bytes(1 << 16)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i, name in enumerate([*names, "pad.bin"]):
            data = _png(i) if name != "pad.bin" else pad
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / f"{path.name}.part_00").write_bytes(raw[: len(raw) // 2])
    (path.parent / f"{path.name}.part_01").write_bytes(raw[len(raw) // 2 :])


GROUPED = vl.ArchiveConfig(("qa.json",), ("img/a.tar.gz.part_*",), "fullpath", True, "obj/")
MULTI = vl.ArchiveConfig(("multi.json",), ("img/a.tar.gz.part_*",), "basename", False)


def _archive_upstream(root: Path) -> Path:
    up = upstream_dir(root, SOURCE)
    _tarball(up / "img" / "a.tar.gz", ["obj/1.png", "obj/2.png", "other/3.png"])
    qa = [
        {"image": "obj/1.png", "conversations": [{"from": "human", "value": "<image>\nWhere?"}]},
        {"image": "obj/2.png", "conversations": [{"from": "gpt", "value": "At (0.5, 0.25)."}]},
        {"image": "obj/1.png", "conversations": [{"from": "gpt", "value": "Here."}]},
        {"image": "other/3.png", "conversations": [{"from": "gpt", "value": "Not obj."}]},
        {"image": "obj/9.png", "conversations": [{"from": "gpt", "value": "Not in the tar."}]},
    ]
    multi = [
        {
            "id": "m0",
            "image": ["3.png", "1.png"],
            "conversations": [{"from": "human", "value": "Same?"}, {"from": "gpt", "value": "No"}],
        }
    ]
    (up / "qa.json").write_text(json.dumps(qa))
    (up / "multi.json").write_text(json.dumps(multi))
    return up


def test_archive_build_and_verify(tmp_path: Path) -> None:
    _archive_upstream(tmp_path)
    reader = vl.Archive({"grouped": GROUPED, "multi": MULTI})
    pin = vl.vl_pin((SOURCE,), {"grouped": 2, "multi": 1})
    reader.build("x", tmp_path, pin, (0, 1))
    grouped = _rows(tmp_path / "vl" / "spatial" / "grouped")
    assert [r["id"] for r in grouped] == ["obj/1.png", "obj/2.png"]
    assert grouped[0]["conversations"] == [
        {"role": "user", "content": "Where?"},
        {"role": "assistant", "content": "Here."},
    ]
    assert grouped[1]["conversations"][0]["content"] == "At (500, 250)."
    assert grouped[0]["images"][0]["bytes"] == _png(0) and grouped[0]["data_source"] == "grouped"
    (multi,) = _rows(tmp_path / "vl" / "spatial" / "multi")
    assert multi["id"] == "m0" and [i["path"] for i in multi["images"]] == ["3.png", "1.png"]
    assert [i["bytes"] for i in multi["images"]] == [_png(2), _png(0)]
    # the grouped JSON has 3 entries with an obj/ image; one image is not in the tarball
    with pytest.raises(ValueError, match="2 rows, 3 JSON entries"):
        reader.verify("x", tmp_path, pin)
    reader.verify("x", tmp_path, vl.vl_pin((SOURCE,), {"multi": 1}))


def test_archive_subset_reads_a_head_only(tmp_path: Path) -> None:
    up = _archive_upstream(tmp_path)
    (up / "img" / "a.tar.gz.part_01").unlink()
    reader = vl.Archive({"grouped": GROUPED})
    size = (up / "img" / "a.tar.gz.part_00").stat().st_size
    head = HfHead(SOURCE.repo, SOURCE.revision, "img/a.tar.gz.part_00", size)
    pin = vl.vl_pin((SOURCE, head), {"grouped": 1}, limit=1)
    reader.build("x", tmp_path, pin, (0, 1))
    reader.verify("x", tmp_path, pin)
    assert [r["id"] for r in _rows(tmp_path / "vl" / "spatial" / "grouped")] == ["obj/1.png"]
    reader.build("x", tmp_path, vl.vl_pin((SOURCE, head), {"grouped": 2}, limit=5), (0, 1))
    assert len(_rows(tmp_path / "vl" / "spatial" / "grouped")) == 2
    # without the head pin, a short tarball raises
    with pytest.raises((EOFError, tarfile.ReadError)):
        reader.build("x", tmp_path, vl.vl_pin((SOURCE,), {"grouped": 2}), (0, 1))


# ------------------------------------------------------------------ PixMo


def _pixmo_upstream(root: Path) -> tuple[Path, dict[str, bytes]]:
    up = upstream_dir(root, SOURCE)
    images = {f"sha{i}": _png(i) for i in range(3)}
    import hashlib

    shas = {k: hashlib.sha256(v).hexdigest() for k, v in images.items()}
    rows = [
        ("sha0", [{"x": 50.0, "y": 25.0}], None, "cup", "pointing"),
        ("sha1", [], 0.0, "dog", "counting"),
        ("sha0", [], None, "fork", "pointing"),
        ("sha2", [{"x": 1.0, "y": 2.0}], 1.0, "cat", "counting"),
    ]
    (up / "data").mkdir(parents=True)
    pd.DataFrame(
        {
            "image_url": [f"https://img/{r[0]}" for r in rows],
            "image_sha256": [shas[r[0]] for r in rows],
            "points": [r[1] for r in rows],
            "count": [r[2] for r in rows],
            "label": [r[3] for r in rows],
            "collection_method": [r[4] for r in rows],
        }
    ).to_parquet(up / "data" / "train-00000-of-00001.parquet")
    return up, {f"https://img/{k}": v for k, v in images.items()}


def _fake_urlopen(monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes]) -> None:
    def urlopen(request: object, timeout: float) -> io.BytesIO:
        url = request.full_url  # type: ignore[attr-defined]
        if url not in served:
            raise urllib.error.HTTPError(url, 404, "gone", {}, None)  # type: ignore[arg-type]
        return io.BytesIO(served[url])

    monkeypatch.setattr(vl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vl.time, "sleep", lambda _: None)


def test_pixmo_drops_the_rows_of_failed_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    up, images = _pixmo_upstream(tmp_path)
    del images["https://img/sha1"]
    _fake_urlopen(monkeypatch, images)
    pin = vl.vl_pin((SOURCE,), {"pixmo_points": None})
    reader = vl.Pixmo()
    with caplog.at_level(logging.INFO):
        vl.scrape(up, None)
    assert "2 images kept, 1 failed" in caplog.text
    failed = (up / "images" / "_failed.tsv").read_text().splitlines()
    assert len(failed) == 1 and failed[0].endswith("\tHTTP 404\thttps://img/sha1")
    reader.build("pixmo", tmp_path, pin, (0, 1))
    rows = _rows(tmp_path / "vl" / "spatial" / "pixmo_points")
    assert [r["conversations"][1]["content"] for r in rows] == [
        "[(500, 250)]",
        'There are no "fork" in the image.',
        "1",
    ]
    reader.verify("pixmo", tmp_path, pin)
    # an image that comes back online after the build: its rows are missing
    (up / "images" / f"{pd.read_parquet(up / 'data').image_sha256[1]}.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="3 kept images have 4"):
        reader.verify("pixmo", tmp_path, pin)


def test_pixmo_subset_takes_the_first_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    up, images = _pixmo_upstream(tmp_path)
    _fake_urlopen(monkeypatch, images)
    pin = vl.vl_pin((SOURCE,), {"pixmo_points": None}, limit=1)
    vl.scrape(up, pin.limit)
    assert len(list((up / "images").glob("*.jpg"))) == 1
    vl.Pixmo().build("pixmo", tmp_path, pin, (0, 1))
    vl.Pixmo().verify("pixmo", tmp_path, pin)
    assert len(_rows(tmp_path / "vl" / "spatial" / "pixmo_points")) == 2


# ------------------------------------------------------------------ FineVision / LLaVA-OV


def test_snapshot_copies_the_configs_to_arrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from datasets import Dataset, Features, Value
    from datasets import Image as HfImage

    from reki.data.robotics.vl import _build_finevision_inner
    from reki.data.utils import load_vl_config

    up = upstream_dir(tmp_path, SOURCE)
    features = Features(
        {
            "images": [HfImage()],
            "texts": [{"user": Value("string"), "assistant": Value("string")}],
        }
    )
    rows = [
        {
            "images": [{"bytes": _png(i), "path": None}],
            "texts": [{"user": f"q{i}", "assistant": "a"}],
        }
        for i in range(5)
    ]
    (up / "vsr").mkdir(parents=True)
    for k, part in enumerate((rows[:3], rows[3:])):
        path = up / "vsr" / f"train-{k}.parquet"
        Dataset.from_list(part, features=features).to_parquet(str(path))
    reader = vl.Snapshot(paths.FINEVISION_DIR, ("images", "texts"), ("vsr", "gqa"))
    pin = vl.vl_pin((SOURCE,), {"vsr": 5})
    reader.build("finevision", tmp_path, pin, (0, 1))
    reader.build("finevision", tmp_path, pin, (0, 1))  # kept
    out = tmp_path / "vl" / "finevision" / "vsr"
    assert [f.name for f in vl.shard_files(out)] == [
        "data-00000-of-00002.arrow",
        "data-00001-of-00002.arrow",
    ]
    loaded = load_vl_config(str(out.parent), "vsr")
    assert loaded["texts"] == [r["texts"] for r in rows]  # rows in order
    assert np.asarray(loaded[4]["images"][0])[0, 0, 0] == 4  # decoded image
    reader.verify("finevision", tmp_path, pin)
    with pytest.raises(ValueError, match="5 rows, pinned 4"):
        reader.verify("finevision", tmp_path, vl.vl_pin((SOURCE,), {"vsr": 4}))
    with pytest.raises(FileNotFoundError, match="no parquet files"):
        reader.build("finevision", tmp_path, vl.vl_pin((SOURCE,), {"gqa": 1}), (0, 1))
    # a build of fewer configs than the full pin marks a subset root: the loader reads those
    assert json.loads((out.parent / "SUBSET.json").read_text()) == ["gqa"]
    (out.parent / "SUBSET.json").write_text('["vsr"]')
    monkeypatch.setenv("FINEVISION_PATH", str(out.parent))
    (tmp_path / "llava").mkdir()
    (tmp_path / "llava" / "SUBSET.json").write_text("[]")
    monkeypatch.setenv("LLAVA_EXT_PATH", str(tmp_path / "llava"))
    with caplog.at_level(logging.WARNING):
        inner = _build_finevision_inner(1)
    assert [d.dataset_name for d in inner] == ["vsr"]
    assert "SUBSET root (SUBSET.json), 1 of 51 FineVision and 0 of 18 LLaVA-OV" in caplog.text


def test_verify_reads_a_parquet_config_dir(tmp_path: Path) -> None:
    from datasets import Dataset, Features, Value
    from datasets import Image as HfImage

    features = Features(
        {
            "images": [HfImage()],
            "texts": [{"user": Value("string"), "assistant": Value("string")}],
        }
    )
    rows = [
        {"images": [{"bytes": _png(i), "path": None}], "texts": [{"user": "q", "assistant": "a"}]}
        for i in range(3)
    ]
    out = tmp_path / "vl" / "finevision" / "vsr"
    out.mkdir(parents=True)
    for k, part in enumerate((rows[:2], rows[2:])):
        Dataset.from_list(part, features=features).to_parquet(str(out / f"train-{k}.parquet"))
    reader = vl.Snapshot(paths.FINEVISION_DIR, ("images", "texts"), ("vsr",))
    reader.verify("finevision", tmp_path, vl.vl_pin((SOURCE,), {"vsr": 3}))
    with pytest.raises(ValueError, match="3 rows, pinned 2"):
        reader.verify("finevision", tmp_path, vl.vl_pin((SOURCE,), {"vsr": 2}))
    (out / "data-00000-of-00001.arrow").touch()
    with pytest.raises(ValueError, match="both parquet and Arrow"):
        reader.verify("finevision", tmp_path, vl.vl_pin((SOURCE,), {"vsr": 3}))
    for f in out.iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError, match=r"no data-\*.arrow or \*.parquet"):
        reader.verify("finevision", tmp_path, vl.vl_pin((SOURCE,), {"vsr": 3}))


# ------------------------------------------------------------------ CLI


def test_vl_group_and_stages_without_effect(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts.data.pretrain_data import main, resolve

    names = ["finevision", "llava_onevision", "robopoint", "refspatial", "pixmo"]
    assert resolve("vl") == names
    assert resolve("pixmo,berkeley_cable_routing") == ["pixmo", "berkeley_cable_routing"]
    with caplog.at_level(logging.INFO):
        main(["curate", "vl", "--root", str(tmp_path)])
        main(["stats", "pixmo", "--root", str(tmp_path)])
    assert caplog.text.count("does nothing for VL data") == 6


def test_vl_pins() -> None:
    from pretrain import catalog

    for name in catalog.groups()["vl"]:
        full, subset = catalog.pin(name, False), catalog.pin(name, True)
        assert isinstance(full, vl.VlPin) and isinstance(subset, vl.VlPin)
        assert set(subset.rows) <= set(full.rows) and full.limit is None
    assert sum(n or 0 for n in vl.FINEVISION_ROWS.values()) == 6_349_586
    assert len(vl.LLAVA_ROWS) == 18
