"""Frames outside keep ranges / subtask intervals are excluded at construction, like dropped
episodes: the sampler never draws them and the ratios use the kept frame counts."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import QWEN

pytestmark = pytest.mark.cpu

# 3 episodes x 20 frames: ep 0 dropped, ep 1 keep range [2, 15] and intervals [0, 5) + [10, 18),
# ep 2 unlabelled.
LOOKUP = [
    {
        "episode_index": 1,
        "intervals": [[0, 5, "a"], [10, 18, "b"]],
        "action_start": 2,
        "action_end": 15,
    },
]
KEPT = [20 + f for f in (2, 3, 4, 10, 11, 12, 13, 14, 15)] + list(range(40, 60))


def test_kept_frames() -> None:
    from reki.data.robotics.combined_dataset import kept_frames

    bounds = ([0, 20, 40], [20, 40, 60], [0, 1, 2])
    keep_ranges = {1: [[2, 15]]}
    intervals = {1: [(0, 5, "a"), (10, 18, "b")], 2: []}
    assert kept_frames(bounds, {0}, keep_ranges, intervals, False).tolist() == KEPT
    outside = kept_frames(bounds, {0}, keep_ranges, intervals, True)
    assert outside.tolist() == list(range(20, 60)), "only the dropped episode is excluded"


def test_sampler_draws_kept_frames_only(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import reki.data.robotics.factory as factory
    from reki.data.robotics.factory import VALIDATION_SETTINGS, make_combined_dataset
    from reki.policy.train_config import TrainConfig
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    leaf = tiny_lerobot_leaf("libero", n_episodes=3, n_frames=20)
    (leaf / "meta" / "subtask_lookup.jsonl").write_text("\n".join(map(json.dumps, LOOKUP)))
    (leaf / "meta" / "dropped_episodes.json").write_text("[0]")
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    ratio_sizes: list[list[int]] = []
    ratios = factory.robot_sampling_ratios

    def spy(cfg: Any, names: list[str], sizes: list[int]) -> list[float]:
        ratio_sizes.append(list(sizes))
        return ratios(cfg, names, sizes)

    monkeypatch.setattr(factory, "robot_sampling_ratios", spy)
    kw: dict[str, Any] = dict(VALIDATION_SETTINGS, subtask_lookup_filename="subtask_lookup.jsonl")
    cfg = TrainConfig(dataset="libero", vlm_model_id=QWEN, steps=1, batch_size=1, **kw)
    ds: Any
    sampler: Any
    ds, sampler = make_combined_dataset(cfg)
    assert ratio_sizes == [[len(KEPT)]] and len(sampler) == len(KEPT)
    random.seed(0)
    assert {i for _ in range(30) for i in sampler} == set(KEPT)
    for i in (KEPT[0], KEPT[3], KEPT[-1]):
        item = ds[i]
        assert int(item["frame_index"]) == i % 20 and int(item["episode_index"]) == i // 20
    with pytest.raises(AssertionError, match="not a kept frame"):
        ds[25]
    redraws = {ds._random_kept_index() for _ in range(500)}
    assert redraws <= set(KEPT) and np.isin(KEPT, list(redraws)).mean() > 0.9


def test_frames_with_an_untracked_chunk_are_not_kept() -> None:
    from reki.data.robotics.combined_dataset import kept_frames
    from reki.data.robotics.transforms import untracked_frames

    action = np.ones((10, 32), np.float32)
    action[0:4, :] = 0
    action[4, 0:16] = 0
    action[7:10, :] = 0
    np.testing.assert_array_equal(untracked_frames(action), [1, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    bounds = ([0], [10], [0])
    untracked = untracked_frames(action)
    assert kept_frames(bounds, set(), None, None, True, untracked).tolist() == [4, 5, 6]
    # a chunk of 3 frames reaches a tracked row from frame 2; the chunk ends at the episode end
    assert kept_frames(bounds, set(), None, None, True, untracked, 3).tolist() == [2, 3, 4, 5, 6]
    two_eps = kept_frames(([0, 5], [5, 10], [0, 1]), set(), None, None, True, untracked, 3)
    assert two_eps.tolist() == [2, 3, 4, 5, 6]


def test_sampler_drops_untracked_chunks_and_the_ego_stats_follow(
    tiny_lerobot_leaf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler drops the frames whose action chunk has no tracked hand; ``ego_stats`` is
    over the same frames, and the band leaves the untracked hand rows out."""
    import sys

    from reki.data.robotics.factory import VALIDATION_SETTINGS, make_combined_dataset
    from reki.policy.train_config import TrainConfig
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "data"))
    from pretrain.common.v30 import read_leaf

    leaf = tiny_lerobot_leaf("xperience", n_episodes=2, n_frames=20)
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    kw: dict[str, Any] = dict(VALIDATION_SETTINGS, subtask_lookup_filename="subtask_lookup.jsonl")
    kw.update(chunk_duration_s=None, chunk_size=3, state_encoding="none", history_k=1)
    cfg = TrainConfig(dataset="xperience", vlm_model_id=QWEN, steps=1, batch_size=1, **kw)
    from pretrain.readers import egocentric as ego

    ego.ego_wrist("xperience", leaf.parents[1], [leaf], False, cfg, windows=2, workers=1)
    ds: Any
    sampler: Any
    ds, sampler = make_combined_dataset(cfg)
    expected = [i for i in range(40) if i % 20 >= 4]
    (kept,) = ds.kept_local_frames()
    assert kept.tolist() == expected and len(sampler) == len(expected)

    ego.ego_stats("xperience", leaf.parents[1], [leaf], False, cfg)
    band = json.loads((leaf / "meta" / "stats.json").read_text())["action"]
    action = read_leaf(leaf, ("action",))["action"].astype(np.float32)[kept]
    assert band["count"] == [len(kept)]
    np.testing.assert_array_equal(band["mean"], action.astype(np.float64).mean(axis=0))
    for lo, hi in ((0, 16), (16, 32)):
        rows = action[action[:, lo:hi].any(axis=1), lo:hi]
        np.testing.assert_array_equal(band["qk01"][lo:hi], np.percentile(rows, 1, axis=0))
        np.testing.assert_array_equal(band["qk99"][lo:hi], np.percentile(rows, 99, axis=0))


@pytest.mark.parametrize("with_lookup", [False, True])
def test_lookup_table_is_printed_only_when_a_leaf_has_one(
    with_lookup: bool,
    tiny_lerobot_leaf: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from reki.data.robotics.factory import VALIDATION_SETTINGS, make_combined_dataset
    from reki.policy.train_config import TrainConfig
    from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor

    leaf = tiny_lerobot_leaf("libero", n_episodes=3, n_frames=20)
    if with_lookup:
        (leaf / "meta" / "subtask_lookup.jsonl").write_text("\n".join(map(json.dumps, LOOKUP)))
    monkeypatch.setattr(Qwen3_5Preprocessor, "from_pretrained", lambda *a, **k: None)
    kw: dict[str, Any] = dict(VALIDATION_SETTINGS, subtask_lookup_filename="subtask_lookup.jsonl")
    make_combined_dataset(TrainConfig(dataset="libero", vlm_model_id=QWEN, steps=1, **kw))
    out = capsys.readouterr().out
    assert ("Subtask lookup per dataset (1/1 leaves have one" in out) == with_lookup
    assert ("Subtask lookup" in out) == with_lookup
