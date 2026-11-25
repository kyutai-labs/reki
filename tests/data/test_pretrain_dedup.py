"""Episode keys and lists of excluded episodes of the pretraining pipeline
(``pretrain.common.dedup``, ``pretrain.common.drops``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pretrain.common import dedup, drops
from pretrain.common.v30 import Episode, write_leaf

pytestmark = pytest.mark.cpu


def _ep(seed: int, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)).astype(np.float32), rng.normal(size=(n, 2)).astype(np.float32)


def _groups(items: list[tuple[int, int, str, tuple[np.ndarray, np.ndarray]]]) -> dict:
    out: dict = {}
    for leaf, ep, label, arrays in items:
        out.setdefault(dedup.episode_key(*arrays), []).append((leaf, ep, label))
    return out


def test_key_is_float32_content_and_length() -> None:
    state, action = _ep(0)
    assert dedup.episode_key(state.astype(np.float64), action) == dedup.episode_key(state, action)
    nudged = state.copy()
    nudged[0, 0] = np.nextafter(nudged[0, 0], np.float32(np.inf))
    assert dedup.episode_key(nudged, action) != dedup.episode_key(state, action)
    assert dedup.episode_key(state, action).endswith("-6")


def test_detect_keeps_one_copy_and_drops_hits() -> None:
    a, b, c = _ep(0), _ep(1), _ep(2)
    groups = _groups([(0, 0, "pick", a), (0, 1, "x", b), (1, 0, "pick", a), (1, 1, "y", c)])
    listed = drops.detect(groups, {(1, 1)}, None)
    assert listed == drops.Drops([dedup.episode_key(*c)], {dedup.episode_key(*a): "pick"})
    assert drops.apply(listed, groups, 2) == [[], [0, 1]]


def test_detect_keeps_no_hit_copy_and_the_old_label() -> None:
    a = _ep(0)
    groups = _groups([(0, 0, "fold", a), (0, 1, "pick", a), (0, 2, "write", a)])
    key = dedup.episode_key(*a)
    assert drops.detect(groups, {(0, 0)}, None).kept == {key: "pick"}
    old = drops.Drops([], {key: "write"})
    listed = drops.detect(groups, set(), old)
    assert listed.kept == {key: "write"}
    assert drops.apply(listed, groups, 1) == [[0, 1]]
    assert drops.detect(groups, {(0, 0), (0, 1), (0, 2)}, old) == drops.Drops([key], {})


def test_apply_rejects_another_build() -> None:
    a, b = _ep(0), _ep(1)
    groups = _groups([(0, 0, "pick", a), (0, 1, "pick", a)])
    with pytest.raises(ValueError, match="1 listed episodes not found"):
        drops.apply(
            drops.Drops([dedup.episode_key(*b)], {dedup.episode_key(*a): "pick"}), groups, 1
        )
    with pytest.raises(ValueError, match="1 duplicates not listed"):
        drops.apply(drops.Drops([], {}), groups, 1)
    with pytest.raises(ValueError, match="no copy has the kept label"):
        drops.apply(drops.Drops([], {dedup.episode_key(*a): "place"}), groups, 1)


def test_groups_span_leaves_with_reader_labels(tmp_path: Path) -> None:
    a, b = _ep(0), _ep(1)
    write_leaf([Episode(0, *a, "pick it"), Episode(1, *b, "fold")], tmp_path / "l0", 10, "t")
    write_leaf([Episode(0, *a, "pick that")], tmp_path / "l1", 10, "t")
    leaves = [tmp_path / "l0", tmp_path / "l1"]
    assert drops.groups(leaves, lambda leaf: None)[dedup.episode_key(*a)] == [
        (0, 0, "pick it"),
        (1, 0, "pick that"),
    ]
    folders = {"l0": ["pick", "fold"], "l1": ["pick"]}
    groups = drops.groups(leaves, lambda leaf: folders[leaf.name])
    assert groups[dedup.episode_key(*a)] == [(0, 0, "pick"), (1, 0, "pick")]
    assert dedup.shared_across({"d0": set(groups), "d1": {dedup.episode_key(*a)}}) == 1


def test_key_columns_group_episodes_that_differ_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pretrain_data
    from pretrain.recipe import Reader, Recipe

    class Keyed(Reader):
        def key_columns(self) -> dedup.Columns:
            return (0, 2), (1,)

    state, action = _ep(0)
    other_state, other_action = state.copy(), action.copy()
    other_state[:, 1] += 1.0
    other_action[:, 0] += 1.0
    write_leaf([Episode(0, state, action, "pick")], tmp_path / "l0", 10, "t")
    write_leaf([Episode(0, other_state, other_action, "pick")], tmp_path / "l1", 10, "t")
    leaves = [tmp_path / "l0", tmp_path / "l1"]
    key = dedup.episode_key(state, action, ((0, 2), (1,)))
    assert key == dedup.episode_key(other_state, other_action, ((0, 2), (1,)))
    assert len(drops.groups(leaves, lambda leaf: None)) == 2

    monkeypatch.setattr(pretrain_data.catalog, "pin", lambda name, subset: None)
    for reader, dropped in ((Keyed(), [[], [0]]), (Reader(), [[], []])):
        recipes = {"d": Recipe(reader, curate=(), stats=())}
        monkeypatch.setattr(pretrain_data.catalog, "recipes", lambda: recipes)
        assert pretrain_data.leaf_drops("d", tmp_path, True, leaves)[0] == dropped


def test_save_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drops, "DIR", tmp_path)
    assert drops.load("d") is None
    listed = drops.Drops(["k1-3"], {"k2-4": "pick"})
    drops.save("d", listed)
    assert drops.load("d") == listed


def test_verify_compares_with_the_drop_list(tmp_path: Path) -> None:
    import json

    from pretrain.common import verify
    from pretrain.pins import Pin

    a = _ep(0)
    write_leaf([Episode(0, *a, "pick"), Episode(1, *a, "pick")], tmp_path / "l0", 10, "t")
    leaves = [tmp_path / "l0"]
    (leaves[0] / "meta" / "dropped_episodes.json").write_text(json.dumps([1]))
    pin = Pin((), episodes=2, frames=12)
    verify.check_counts("d", leaves, pin, [[1]])
    with pytest.raises(ValueError, match="differs from the list of excluded episodes"):
        verify.check_counts("d", leaves, pin, [[0, 1]])
