"""RoboCOIN reader of the pretraining pipeline (``pretrain.readers.robocoin``): pins table,
collection routing, the airbot degree → radian transform."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pretrain.readers import robocoin

pytestmark = pytest.mark.cpu


def test_pins_table() -> None:
    from reki.data.robotics.registry import spec_names

    leaves = robocoin.leaf_pins()
    assert len(leaves) == 672
    assert len({p.leaf for p in leaves}) == len({p.repo for p in leaves}) == 672
    assert sorted(robocoin.PINS) == sorted(spec_names("robocoin"))
    assert all(len(p.revision) == 40 for p in leaves)
    by_leaf = {p.leaf: p for p in leaves}
    assert (
        by_leaf["AI2_Alphabot_2_move_test_tube"].repo == "RoboCOIN/AI2_Alphabot_2_move_test_tube_1"
    )
    assert by_leaf["AI2_Alphabot_2_organize_cosmetics"].revision.startswith("d5bde6f5c0")
    renamed = [
        p.repo
        for p in leaves
        if p.repo != f"RoboCOIN/{p.leaf}" and p.leaf != "AI2_Alphabot_2_move_test_tube"
    ]
    assert len(renamed) == 18
    assert all(r.startswith("RoboCOIN/Agilex_Cobot_Magic_") for r in renamed)
    assert sum(p.episodes for p in leaves) == 137847


def test_routing_matches_pins() -> None:
    names = {p.leaf: p.collection for p in robocoin.leaf_pins()}
    assert robocoin.collection_of("AIRBOT_MMK2_x", 36, ()) == names["AIRBOT_MMK2_desktop_storage"]
    assert robocoin.collection_of("Split_aloha_x", 26, (6, 19)) == "robocoin_agilex_26"
    assert robocoin.collection_of("Agilex_Cobot_Magic_x", 26, (12, 25)) == "robocoin_agilex_26b"
    assert robocoin.collection_of("R1_Lite_x", 16, ()) is None
    with pytest.raises(ValueError):
        robocoin.collection_of("Agilex_x", 26, (0, 1))


def _v21(root: Path, hands: list[np.ndarray]) -> Path:
    """A minimal LeRobot v2.1 leaf: 36-dim state/action, one episode per ``hands`` entry."""
    (root / "meta").mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    lines = []
    for i, hand in enumerate(hands):
        arr = np.zeros((len(hand), 36), dtype=np.float32)
        arr[:, 12:36] = hand[:, None]
        col = pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1)), 36)
        path = root / info["data_path"].format(episode_chunk=0, episode_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"observation.state": col, "action": col}), path)
        lines.append(json.dumps({"episode_index": i, "tasks": ["t"], "length": len(hand)}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_hand_degrees_and_idempotence(tmp_path: Path) -> None:
    deg = [np.array([0.0, 90.0, 117.0], np.float32), np.array([10.0, 60.0], np.float32)]
    assert robocoin.hand_degrees(_v21(tmp_path / "deg", deg))
    rad = [
        robocoin.to_radians(h[:, None].repeat(36, 1), h[:, None].repeat(36, 1))[0][:, 12]
        for h in deg
    ]
    assert max(r.max() for r in rad) < 2.1
    assert not robocoin.hand_degrees(_v21(tmp_path / "rad", rad))
    with pytest.raises(ValueError, match="in degrees"):
        robocoin.hand_degrees(_v21(tmp_path / "mixed", [deg[0], rad[0]]))


def test_to_radians_hand_dims_only() -> None:
    x = np.full((2, 36), 90.0, dtype=np.float32)
    state, action = robocoin.to_radians(x, x)
    np.testing.assert_array_equal(state[:, :12], x[:, :12])
    assert state.dtype == np.float32
    np.testing.assert_array_equal(action[:, 12:], np.float32(90.0) * np.float32(np.pi / 180))
    assert x[0, 12] == 90.0
