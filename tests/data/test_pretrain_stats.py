"""Stats ops of the pretraining pipeline (``pretrain.common.stats``) against numpy references
over the kept frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
from pretrain.common import stats
from pretrain.common.v30 import Episode, write_leaf

pytestmark = pytest.mark.cpu


def _leaf(path: Path, seed: int, drops: list[int], d: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A leaf of 5 episodes; returns its kept (state, action) frames."""
    rng = np.random.default_rng(seed)
    eps = [
        Episode(
            i,
            rng.normal(size=(8 + i, d)).astype(np.float32),
            (rng.normal(size=(8 + i, d)) * 0.4).astype(np.float32),
            "t",
        )
        for i in range(5)
    ]
    write_leaf(eps, path, fps=10, robot_type="test")
    (path / "meta" / "dropped_episodes.json").write_text(json.dumps(drops))
    kept = [e for e in eps if e.index not in drops]
    if not kept:
        return np.zeros((0, d)), np.zeros((0, d))
    return np.concatenate([e.state for e in kept]), np.concatenate([e.action for e in kept])


STATS_KEYS = {"min", "max", "mean", "std", "count", "q01", "q99", "qk01", "qk99"}


def _stats(leaf: Path, name: str = "stats.json") -> dict:
    return json.loads((leaf / "meta" / name).read_text())


def test_kept_qk_is_lerobot_stats_over_kept_frames(tmp_path: Path) -> None:
    state, action = _leaf(tmp_path / "a", 0, [1, 3])
    stats.kept_qk("test", tmp_path, [tmp_path / "a"], False)
    out = _stats(tmp_path / "a")
    for key, data in (("observation.state", state), ("action", action)):
        ref = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        np.testing.assert_array_equal(out[key]["qk01"], ref["q01"])
        np.testing.assert_array_equal(out[key]["qk99"], ref["q99"])
        np.testing.assert_array_equal(out[key]["max"], data.max(0))
        np.testing.assert_array_equal(out[key]["mean"], data.astype(np.float64).mean(0))
        np.testing.assert_array_equal(out[key]["std"], data.astype(np.float64).std(0))
        assert out[key]["count"] == [len(data)]
        assert out[key]["q01"] == out[key]["qk01"] and out[key]["q99"] == out[key]["qk99"]
        assert set(out[key]) == STATS_KEYS


def test_kept_qk_exact_is_np_quantile(tmp_path: Path) -> None:
    state, _ = _leaf(tmp_path / "a", 0, [0])
    stats.kept_qk("test", tmp_path, [tmp_path / "a"], False, exact=True)
    out = _stats(tmp_path / "a")["observation.state"]
    np.testing.assert_array_equal(out["qk01"], np.quantile(state, 0.01, axis=0))
    np.testing.assert_array_equal(out["qk99"], np.quantile(state, 0.99, axis=0))
    assert out["q01"] == out["qk01"]


def test_std_is_float64_without_cancellation(tmp_path: Path) -> None:
    """A dim with a large offset and a small spread: float32 sqrt(E[x²]−m²) gives 0."""
    rng = np.random.default_rng(0)
    state = (1.71 + 9e-4 * rng.normal(size=(400, 2))).astype(np.float32)
    eps = [
        Episode(i, state[100 * i : 100 * i + 100], state[100 * i : 100 * i + 100], "t")
        for i in range(4)
    ]
    write_leaf(eps, tmp_path / "a", fps=10, robot_type="test")
    stats.kept_qk("test", tmp_path, [tmp_path / "a"], False)
    out = _stats(tmp_path / "a")["observation.state"]
    np.testing.assert_allclose(out["std"], state.astype(np.float64).std(0), rtol=1e-12)
    assert min(out["std"]) > 8e-4


def test_global_qk_pools_kept_frames(tmp_path: Path) -> None:
    _, aa = _leaf(tmp_path / "a", 0, [2])
    _, ab = _leaf(tmp_path / "b", 1, [])
    leaves = [tmp_path / "a", tmp_path / "b"]
    stats.kept_qk("test", tmp_path, leaves, False)
    stats.global_qk("test", tmp_path, leaves, False)
    pooled = np.concatenate([aa, ab])
    for leaf in leaves:
        out = _stats(leaf)["action"]
        np.testing.assert_array_equal(out["qk01"], np.percentile(pooled, 1, axis=0))
        np.testing.assert_array_equal(out["qk99"], np.percentile(pooled, 99, axis=0))


def test_global_qk_alone_writes_kept_leaf_stats(tmp_path: Path) -> None:
    """A recipe with only ``global_qk`` (rdt, ABC, ego) still gets kept-frame basics."""
    sa, aa = _leaf(tmp_path / "a", 0, [1, 3])
    stats.global_qk("test", tmp_path, [tmp_path / "a"], False)
    for key, data in (("observation.state", sa), ("action", aa)):
        out = _stats(tmp_path / "a")[key]
        assert set(out) == STATS_KEYS
        assert out["count"] == [len(data)]
        np.testing.assert_array_equal(out["std"], data.astype(np.float64).std(0))
        ref = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        np.testing.assert_array_equal(out["q01"], ref["q01"])
        np.testing.assert_array_equal(out["qk01"], np.percentile(data, 1, axis=0))


def test_global_qk_pool_raises_when_a_pool_dataset_is_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from pretrain import catalog

    _leaf(tmp_path / "a", 0, [])
    by_name = {"da": [tmp_path / "a"], "db": [tmp_path / "b"]}
    reader = SimpleNamespace(leaves=lambda name, root, pin: by_name[name])
    monkeypatch.setattr(
        catalog, "recipes", lambda: {n: SimpleNamespace(reader=reader) for n in by_name}
    )
    monkeypatch.setattr(catalog, "pin", lambda name, subset: None)
    with pytest.raises(FileNotFoundError, match=r"(?s)\['db'\].*build db --subset --root"):
        stats.global_qk("da", tmp_path, by_name["da"], True, pool=("da", "db"))
    before = (tmp_path / "a" / "meta" / "stats.json").read_bytes()
    recipe = (stats.op("kept_qk"), stats.op("global_qk", pool=("da", "db")))
    with pytest.raises(FileNotFoundError, match=r"\['db'\]"):
        stats.run("da", tmp_path, by_name["da"], recipe, True)
    assert (tmp_path / "a" / "meta" / "stats.json").read_bytes() == before


def test_global_qk_zero_blocks_skip_all_zero_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    action = rng.normal(size=(50, 8)).astype(np.float32)
    action[::3, 0:4] = 0  # block [0, 4) untracked on these rows
    eps = [
        Episode(i, np.zeros((10, 8), np.float32), action[10 * i : 10 * i + 10], "t")
        for i in range(5)
    ]
    write_leaf(eps, tmp_path / "a", fps=10, robot_type="test")
    stats.global_qk("test", tmp_path, [tmp_path / "a"], False, zero_blocks=((0, 4),))
    out = _stats(tmp_path / "a")["action"]
    tracked = action[action[:, 0:4].any(axis=1), 0:4]
    np.testing.assert_array_equal(out["qk01"][0:4], np.percentile(tracked, 1, axis=0))
    np.testing.assert_array_equal(out["qk99"][0:4], np.percentile(tracked, 99, axis=0))
    np.testing.assert_array_equal(out["qk99"][4:8], np.percentile(action[:, 4:8], 99, axis=0))


def test_global_qk_pool_writes_only_own_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from pretrain import catalog

    _, aa = _leaf(tmp_path / "a", 0, [2])
    _, ab = _leaf(tmp_path / "b", 1, [])
    by_name = {"da": [tmp_path / "a"], "db": [tmp_path / "b"]}
    reader = SimpleNamespace(leaves=lambda name, root, pin: by_name[name])
    monkeypatch.setattr(
        catalog, "recipes", lambda: {n: SimpleNamespace(reader=reader) for n in by_name}
    )
    monkeypatch.setattr(catalog, "pin", lambda name, subset: None)
    stats.kept_qk("db", tmp_path, by_name["db"], False)
    own_b = _stats(tmp_path / "b")["action"]["qk01"]
    stats.kept_qk("da", tmp_path, by_name["da"], False)
    stats.global_qk("da", tmp_path, by_name["da"], False, pool=("da", "db"))
    pooled = np.concatenate([aa, ab])
    np.testing.assert_array_equal(
        _stats(tmp_path / "a")["action"]["qk01"], np.percentile(pooled, 1, axis=0)
    )
    assert _stats(tmp_path / "b")["action"]["qk01"] == own_b
    first = _stats(tmp_path / "a")
    stats.global_qk("db", tmp_path, by_name["db"], False, pool=("db", "da"))
    for col in ("action", "observation.state"):
        assert _stats(tmp_path / "b")[col]["qk01"] == first[col]["qk01"]
        assert _stats(tmp_path / "b")[col]["qk99"] == first[col]["qk99"]


def test_global_qk_pool_dims_and_clip(tmp_path: Path) -> None:
    sa, _ = _leaf(tmp_path / "a", 0, [])
    sb, _ = _leaf(tmp_path / "b", 1, [4])
    leaves = [tmp_path / "a", tmp_path / "b"]
    stats.kept_qk("test", tmp_path, leaves, False)
    per_leaf_b = _stats(tmp_path / "b")["observation.state"]["qk01"]
    stats.global_qk(
        "test",
        tmp_path,
        leaves,
        False,
        ["observation.state"],
        pool_dims=4,
        clip_min=0.0,
        clip_dims=[1],
    )
    pooled = np.concatenate([sa, sb])[:, :4]
    pooled[:, 1] = np.clip(pooled[:, 1], 0.0, None)
    out = _stats(tmp_path / "b")["observation.state"]["qk01"]
    np.testing.assert_array_equal(out[:4], np.percentile(pooled, 1, axis=0))
    assert out[4:] == per_leaf_b[4:]


def test_euler_per_leaf_and_pooled(tmp_path: Path) -> None:
    from scipy.spatial.transform import Rotation as R

    _, aa = _leaf(tmp_path / "a", 0, [1])
    _, ab = _leaf(tmp_path / "b", 1, [])
    leaves = [tmp_path / "a", tmp_path / "b"]

    def ref(action: np.ndarray) -> dict:
        eul = action.astype(np.float64)
        eul[:, 3:6] = R.from_rotvec(eul[:, 3:6]).as_euler("xyz")
        return get_feature_stats(eul, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)

    stats.euler("berkeley_rpt_delta", tmp_path, leaves, False)
    out = _stats(tmp_path / "a", "norm_stats_euler.json")["action"]
    np.testing.assert_array_equal(out["qk01"], ref(aa)["q01"])

    stats.euler("interndata_franka", tmp_path, leaves, False, pooled=True)
    pooled = ref(np.concatenate([aa, ab]))
    for leaf, own in ((tmp_path / "a", ref(aa)), (tmp_path / "b", ref(ab))):
        out = _stats(leaf, "norm_stats_euler.json")["action"]
        np.testing.assert_array_equal(out["qk99"][:6], pooled["q99"][:6])
        assert out["qk99"][6] == own["q99"][6]  # gripper_unit_index stays per leaf


def test_kept_mask_follows_the_subtask_lookup(tmp_path: Path) -> None:
    _leaf(tmp_path / "a", 0, [2])
    lookup = {"episode_index": 0, "intervals": [[1, 3, "grasp"]]}
    (tmp_path / "a" / "meta" / "subtask_lookup.jsonl").write_text(json.dumps(lookup) + "\n")
    keep = stats.kept_mask(tmp_path / "a")
    assert np.flatnonzero(keep[:8]).tolist() == [1, 2]
    assert keep[8:17].all() and not keep[17:27].any()


def test_empty_leaf_gets_all_frame_stats_and_stays_out_of_the_band(tmp_path: Path) -> None:
    _, aa = _leaf(tmp_path / "a", 0, [])
    _leaf(tmp_path / "b", 1, [0, 1, 2, 3, 4])
    sb, ab = _leaf(tmp_path / "c", 1, [])  # the same frames as b, all kept
    leaves = [tmp_path / "a", tmp_path / "b"]
    stats.kept_qk("test", tmp_path, leaves, False)
    stats.global_qk("test", tmp_path, leaves, False)
    stats.euler("interndata_franka", tmp_path, leaves, False, pooled=True)
    out = _stats(tmp_path / "b")
    assert out["action"]["count"] == [len(ab)]
    np.testing.assert_array_equal(out["observation.state"]["max"], sb.max(0))
    np.testing.assert_array_equal(out["action"]["qk01"], np.percentile(aa, 1, axis=0))
    euler_b = _stats(tmp_path / "b", "norm_stats_euler.json")["action"]
    euler_a = _stats(tmp_path / "a", "norm_stats_euler.json")["action"]
    assert euler_b["qk99"][:6] == euler_a["qk99"][:6]  # the band of a alone
