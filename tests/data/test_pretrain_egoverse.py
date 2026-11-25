"""EgoVerse reader on synthetic inputs: the pin, the object filter and digest, the checked
download, the Aria → otaheri keypoints, the head pose, the per-leaf fit and the leaf split."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pretrain import catalog
from pretrain.readers import egocentric as ego
from pretrain.readers import egoverse
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.cpu


def test_pin_lists_the_aria_episodes_and_a_two_leaf_subset() -> None:
    full, subset = catalog.pin("egoverse", False), catalog.pin("egoverse", True)
    assert isinstance(full, egoverse.EgoversePin) and isinstance(subset, egoverse.EgoversePin)
    assert (full.episodes, full.frames) == (2514, 11_670_746)
    assert list(full.names) == sorted(set(full.names)) and len(full.names) == full.episodes
    assert set(subset.names) <= set(full.names) and subset.leaves == 2
    assert (subset.episodes, subset.frames) == (5, 7130)
    assert "egoverse" in catalog.recipes()


def test_otaheri_order_maps_twenty_aria_points() -> None:
    mapped = [a for a in egoverse.OTAHERI_FROM_ARIA if a is not None]
    assert len(egoverse.OTAHERI_FROM_ARIA) == 21 and sorted(mapped) == list(range(20))
    assert egoverse.OTAHERI_FROM_ARIA[0] == 5 and egoverse.OTAHERI_FROM_ARIA[13] is None
    aria = np.arange(3 * 21 * 3, dtype=np.float32).reshape(3, 21, 3) + 1
    aria[2] = 0
    out = egoverse.otaheri_keypoints(aria)
    np.testing.assert_array_equal(out[0, 0], aria[0, 5])
    np.testing.assert_array_equal(out[1, 16:], aria[1, 0:5])
    assert np.isnan(out[:2, 13]).all() and np.isfinite(np.delete(out[:2], 13, axis=1)).all()
    assert np.isnan(out[2]).all()


def test_head_pose_quaternion_is_wxyz() -> None:
    rot = Rotation.random(4, random_state=0)
    xyzw = rot.as_quat()
    head = np.concatenate([np.ones((4, 3)), xyzw[:, [3, 0, 1, 2]]], axis=1)
    t, aa = egoverse.camera_pose(head)
    np.testing.assert_array_equal(t, 1.0)
    np.testing.assert_allclose(Rotation.from_rotvec(aa).as_matrix(), rot.as_matrix(), atol=1e-9)
    head[1, 3:] = 0
    with pytest.raises(ValueError, match="non-unit"):
        egoverse.camera_pose(head)


def test_task_text() -> None:
    assert egoverse.task_text({"task_description": " folding clothes"}) == "Folding clothes."
    assert egoverse.task_text({"task_description": "", "task_name": "cup_on_saucer"}) == (
        "Cup on saucer."
    )
    assert egoverse.task_text({"task_description": "Fold it."}) == "Fold it."
    assert egoverse.task_text({}) == ego.FALLBACK_TASK


def test_wanted_objects_and_digest() -> None:
    keep = [
        "e.mp4",
        "e.zarr/zarr.json",
        "e.zarr/obs_head_pose/c/0/0",
        "e.zarr/left.obs_aria_keypoints/zarr.json",
        "e.zarr/right.obs_aria_keypoints/c/0/0",
    ]
    drop = [
        "e.zarr/images.front_1/c/0",
        "e.zarr/left.obs_wrist_pose/c/0/0",
        "e.zarr/obs_eye_gaze/zarr.json",
        "e.flow3d/x/metadata.json",
    ]
    assert all(map(egoverse.wanted, keep)) and not any(map(egoverse.wanted, drop))
    objects = [("b", 2, "y"), ("a", 1, "x")]
    assert egoverse.digest(objects) == egoverse.digest(objects[::-1])
    assert egoverse.digest(objects) != egoverse.digest([("b", 3, "y"), ("a", 1, "x")])


class FakeBucket:
    """The S3 client calls ``listing`` and ``fetch_episode`` make, over in-memory objects."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.fetched: list[str] = []

    def get_paginator(self, op: str) -> Any:
        assert op == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str) -> list[dict]:
        contents = [
            {"Key": k, "Size": len(v), "ETag": f'"{hashlib.md5(v).hexdigest()}"'}
            for k, v in sorted(self.objects.items())
            if k.startswith(Prefix)
        ]
        return [{"Contents": contents}]

    def download_file(self, bucket: str, key: str, dst: str) -> None:
        self.fetched.append(key)
        Path(dst).write_bytes(self.objects[key])


def test_fetch_checks_the_pin_and_the_files(tmp_path: Path) -> None:
    p = egoverse.PREFIX
    objects = {
        f"{p}e.mp4": b"video",
        f"{p}e.zarr/zarr.json": b"{}",
        f"{p}e.zarr/obs_head_pose/c/0/0": b"head",
        f"{p}e.zarr/images.front_1/c/0": b"jpegs",
        f"{p}e.flow3d/m.json": b"flow",
    }
    bucket = FakeBucket(objects)
    pinned = egoverse.digest(egoverse.listing(bucket, "b", "e"))
    egoverse.fetch_episode(bucket, "b", "e", pinned, tmp_path)
    assert sorted(bucket.fetched) == sorted(k for k in objects if egoverse.wanted(k[len(p) :]))
    assert (tmp_path / "e.mp4").read_bytes() == b"video"
    egoverse.fetch_episode(bucket, "b", "e", pinned, tmp_path)
    assert len(bucket.fetched) == 3

    bucket.objects[f"{p}e.mp4"] = b"other"
    with pytest.raises(ValueError, match="differ from the pin"):
        egoverse.fetch_episode(bucket, "b", "e", pinned, tmp_path)

    corrupt = FakeBucket(objects)
    corrupt.download_file = lambda bucket, key, dst: Path(dst).write_bytes(b"x")  # type: ignore[method-assign]
    pinned = egoverse.digest(egoverse.listing(corrupt, "b", "e"))
    with pytest.raises(ValueError, match="differs from its listing"):
        egoverse.fetch_episode(corrupt, "b", "e", pinned, tmp_path / "new")


def test_missing_credentials_fail_before_any_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in egoverse.ENV:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(PermissionError, match="R2_ENDPOINT_URL, R2_ACCESS_KEY_ID"):
        egoverse._client()


def _raw(name: str, n: int, sides: tuple[str, ...], rng: np.random.Generator) -> egoverse.Raw:
    head = np.zeros((n, 7))
    head[:, :3] = rng.normal(size=(n, 3))
    head[:, 3] = 1.0
    keypoints = {s: rng.normal(size=(n, 21, 3)) for s in sides}
    return egoverse.Raw(name, "Task.", head, keypoints)


def test_each_hand_of_a_leaf_is_fitted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from pretrain.readers import mano_fit

    calls: list[tuple[bool, list[int]]] = []

    def fit_hand(kp: np.ndarray, lengths: list[int], is_right: bool, models: Path) -> dict:
        calls.append((is_right, lengths))
        n = len(kp)
        err = np.full(n, 0.01, np.float32)
        err[1] = egoverse.FIT_ERR
        return {
            "finger_aa": np.zeros((n, 45), np.float32),
            "wrist_t": np.arange(n, dtype=np.float32)[:, None].repeat(3, 1),
            "wrist_aa": np.zeros((n, 3), np.float32),
            "fit_err": err,
        }

    monkeypatch.setattr(mano_fit, "fit_hand", fit_hand)
    rng = np.random.default_rng(0)
    raws = [_raw("a", 3, ("left", "right"), rng), _raw("b", 4, ("right",), rng)]
    raws[1].keypoints["right"][2] = 0
    fits = egoverse.fit_hands(raws, Path("models"))
    assert calls == [(False, [3]), (True, [3, 4])]
    assert fits[0]["left_tracked"].tolist() == [True, False, True]
    assert fits[1]["right_tracked"].tolist() == [True, True, False, True]
    np.testing.assert_array_equal(fits[1]["right_wrist_t"][:, 0], [3, 4, 5, 6])

    state, action = egoverse.episode_arrays(raws[1], fits[1], ego.Eigengrasps.load())
    assert state.shape == (4, 12) and action.shape == (4, 32) and action.dtype == np.float32
    np.testing.assert_allclose(state[:, :3], raws[1].head[:, :3], rtol=1e-6)
    assert not action[:, :16].any()
    np.testing.assert_array_equal(action[:, 16], [3, 4, 0, 6])


def test_build_shards_split_the_leaves_round_robin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = tuple(f"e{i}" for i in range(7))
    pin = egoverse.EgoversePin((), episodes=7, frames=None, names=names, leaves=3)
    reader = egoverse.Egoverse()
    written: dict[int, list] = {}

    def write_leaf(name: str, root: Path, leaf: int, episodes: Any, fps: int) -> None:
        written[leaf] = episodes.args[0]

    monkeypatch.setattr(reader, "write_leaf", write_leaf)
    reader.build("egoverse", tmp_path, pin, (1, 2))
    assert written == {1: [(1, "e1"), (4, "e4")]}
    reader.build("egoverse", tmp_path, pin, (0, 1))
    assert written[0] == [(0, "e0"), (3, "e3"), (6, "e6")]
    assert [p.name for p in reader.leaves("egoverse", tmp_path, pin)] == [
        "egoverse_00",
        "egoverse_01",
        "egoverse_02",
    ]
