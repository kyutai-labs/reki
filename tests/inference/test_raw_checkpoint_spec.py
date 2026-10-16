from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest
from test_export_dataset_spec import _cfg, _stats_equal

pytestmark = pytest.mark.cpu


def _named_libero(tiny_lerobot_leaf: Callable[..., Path]) -> Path:
    leaf = tiny_lerobot_leaf("libero")
    info_p = leaf / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    for key, prefix in (("observation.state", "s"), ("action", "a")):
        dim = info["features"][key]["shape"][0]
        info["features"][key]["names"] = [f"{prefix}{i}.pos" for i in range(dim)]
    info_p.write_text(json.dumps(info))
    return leaf


def test_raw_checkpoint_matches_export(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path
) -> None:
    """A raw checkpoint with its dataset gives the spec and stats the export stores, and
    reads only the dataset's meta/."""
    from reki.inference.spec import inference_spec_for, save_inference_spec

    leaf = _named_libero(tiny_lerobot_leaf)
    for sub in ("data", "videos"):
        shutil.rmtree(leaf / sub, ignore_errors=True)
    cfg = _cfg("libero", max_cameras=3)
    raw_dir, export_dir = tmp_path / "raw", tmp_path / "export"
    raw_dir.mkdir()
    export_dir.mkdir()

    raw_spec, raw_stats = inference_spec_for(raw_dir, cfg, "libero")
    save_inference_spec(export_dir, raw_spec, raw_stats)
    exp_spec, exp_stats = inference_spec_for(export_dir, cfg, None)
    assert asdict(raw_spec) == asdict(exp_spec)
    assert _stats_equal(raw_stats, exp_stats)
    assert raw_spec.state_names is not None and raw_spec.camera_names is not None


def test_raw_checkpoint_input_errors(
    tiny_lerobot_leaf: Callable[..., Path], tmp_path: Path
) -> None:
    from reki.inference.spec import inference_spec_for, save_inference_spec

    _named_libero(tiny_lerobot_leaf)
    cfg = _cfg("libero", max_cameras=3)
    raw_dir, export_dir = tmp_path / "raw", tmp_path / "export"
    raw_dir.mkdir()
    export_dir.mkdir()
    with pytest.raises(ValueError, match="raw training checkpoint needs its finetuning dataset"):
        inference_spec_for(raw_dir, cfg, None)
    save_inference_spec(export_dir, *inference_spec_for(raw_dir, cfg, "libero"))
    with pytest.raises(ValueError, match="do not pass a dataset"):
        inference_spec_for(export_dir, cfg, "libero")
