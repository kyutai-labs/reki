"""OXE readers of the pretraining pipeline (``pretrain.readers.oxe``)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from pretrain import catalog
from pretrain.common import verify
from pretrain.common.download import upstream_dir
from pretrain.common.v30 import Episode, episode_bounds, read_leaf, write_leaf
from pretrain.pins import HfSnapshot, Pin
from pretrain.readers import oxe

pytestmark = pytest.mark.cpu


def _episodes(n: int, frames: bool = False) -> list[Episode]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        length = 4 + i
        ep = Episode(
            i,
            rng.normal(size=(length, 3)).astype(np.float32),
            rng.normal(size=(length, 2)).astype(np.float32),
            f"task {i % 2}",
        )
        if frames:
            ep.frames["observation.images.top"] = rng.integers(
                0, 255, size=(length, 32, 32, 3), dtype=np.uint8
            )
        out.append(ep)
    return out


def test_pins_are_consistent() -> None:
    for name, pin in oxe.PINS.items():
        assert pin.subset is not None and pin.subset.select is not None, name
        assert len(pin.subset.select) == pin.subset.episodes, name
        assert list(pin.subset.select) == sorted(set(pin.subset.select)), name
        for source in (*pin.sources, *pin.subset.sources):
            assert isinstance(source, HfSnapshot)
            assert re.fullmatch(r"[0-9a-f]{40}", source.revision), name


@pytest.mark.parametrize("select", [None, (0, 2)])
def test_passthrough_keeps_values_and_video_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, select: tuple[int, ...] | None
) -> None:
    eps = _episodes(3, frames=True)
    source = HfSnapshot("test/bridge", "0" * 40)
    src = upstream_dir(tmp_path, source)
    write_leaf(eps, src, fps=10, robot_type="test", data_file_mb=0)
    full = Pin((source,), episodes=3, frames=15)
    kept = list(select) if select is not None else [0, 1, 2]
    pin = Pin(
        (source,),
        episodes=len(kept),
        frames=sum(len(eps[i].state) for i in kept),
        select=select,
    )
    monkeypatch.setattr(catalog, "pins", lambda: {"bridge_v2": full})

    reader = oxe.PassthroughV30("bridge_v2_lerobot")
    reader.build("bridge_v2", tmp_path, pin, (0, 1))
    (out,) = reader.leaves("bridge_v2", tmp_path, pin)

    arrays = read_leaf(out, ["observation.state", "action", "episode_index", "index"])
    np.testing.assert_array_equal(
        arrays["observation.state"], np.concatenate([eps[i].state for i in kept])
    )
    np.testing.assert_array_equal(arrays["action"], np.concatenate([eps[i].action for i in kept]))
    lengths = [len(eps[i].state) for i in kept]
    np.testing.assert_array_equal(arrays["episode_index"], np.repeat(range(len(kept)), lengths))
    np.testing.assert_array_equal(arrays["index"], np.arange(sum(lengths)))
    from_idx, to_idx, ep_ids = episode_bounds(out)
    assert ep_ids == list(range(len(kept)))
    assert to_idx == list(np.cumsum(lengths))
    info = json.loads((out / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (len(kept), sum(lengths))
    verify.check_videos(out)


def test_passthrough_rejects_wrong_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = HfSnapshot("test/bridge", "0" * 40)
    write_leaf(_episodes(2), upstream_dir(tmp_path, source), fps=10, robot_type="test")
    pin = Pin((source,), episodes=2, frames=9)
    monkeypatch.setattr(catalog, "pins", lambda: {"bridge_v2": pin})
    wrong = Pin((source,), episodes=1, frames=99, select=(1,))
    with pytest.raises(ValueError, match="pinned"):
        oxe.PassthroughV30("bridge_v2_lerobot").build("bridge_v2", tmp_path, wrong, (0, 1))
