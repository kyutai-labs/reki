"""Egocentric readers (egodex, xperience) on synthetic inputs: resampling, subtask intervals,
the lookup's PTS fallback, the egodex episode assembly and the ``ego_wrist`` stats op."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from pretrain import catalog
from pretrain.common.v30 import Episode, write_leaf
from pretrain.readers import egocentric as ego
from pretrain.readers import egodex, xperience
from scipy.spatial.transform import Rotation

from reki.data.robotics.transforms import interpolate_wrist_chunk, wrist_to_camera_frame

pytestmark = pytest.mark.cpu


def test_catalog_has_both_datasets_with_subset_pins() -> None:
    for name in ("egodex", "xperience"):
        assert catalog.pin(name, True) is not catalog.pin(name, False)
        assert name in catalog.recipes()
    egodex_subset, xperience_subset = catalog.pin("egodex", True), catalog.pin("xperience", True)
    assert isinstance(egodex_subset, egodex.EgodexPin)
    assert isinstance(xperience_subset, xperience.XperiencePin)
    assert egodex_subset.folders == (("part1", "fry_bread"), ("part4", "play_reset_connect_four"))
    assert all(f.startswith("part4/play_reset_connect_four/") for f in egodex_subset.files)
    assert xperience_subset.leaves == 2


def test_segments_split_at_gaps_and_drop_short() -> None:
    # 2 s at 25 fps, a 0.5 s gap, 0.6 s more (too short), then nothing
    pts = np.concatenate([np.arange(50) / 25, 2.5 + np.arange(15) / 25])
    segs = list(xperience.segments(pts))
    assert len(segs) == 1
    a, b, grid = segs[0]
    assert (a, b) == (0, 50)
    np.testing.assert_allclose(grid, np.minimum(np.arange(40) / 20, pts[49]))
    assert list(xperience.segments(pts[:19])) == []


def test_nearest_source_and_hand_resampling() -> None:
    pts = np.array([0.0, 0.04, 0.12, 0.16])
    grid = np.array([0.0, 0.05, 0.1, 0.15])
    np.testing.assert_array_equal(xperience.nearest_source(pts, 0, 4, grid), [0, 1, 2, 3])
    rot = Rotation.from_rotvec([[0, 0, 0], [0, 0, 0.4], [0, 0, 1.2], [0, 0, 1.6]]).as_rotvec()
    ann = {
        "left_wrist_t": np.stack([pts * 10, pts, pts], 1).astype(np.float32),
        "left_wrist_aa": rot.astype(np.float32),
        "left_finger_aa": np.zeros((4, 45), np.float32),
        "left_valid": np.array([True, True, True, False]),
    }
    pca = ego.Eigengrasps.load()
    block = xperience.resample_hand(ann, "left", pca, pts, 0, 4, grid)
    assert block.shape == (4, 16) and block.dtype == np.float32
    np.testing.assert_allclose(block[:2, 0], [0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(block[1, 5], 0.5, atol=1e-6)
    # the frame bracketed by the untracked source frame 3 is zero
    assert not block[3].any() and block[:3].any(axis=1).all()


def _rigid(rot: Rotation, t: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = rot.as_matrix(), t
    return out


def test_world_poses_compose_to_the_observation_c0_frame() -> None:
    """Stored world camera + wrist, through the loader compose, give the wrist of frame t+k in
    c0 at frame t: ``B·T_bw(t)·T_bw(t+k)⁻¹·B⁻¹·T_hand(t+k)``."""
    rng = np.random.default_rng(0)
    n = 5
    slam = Rotation.random(n, random_state=1)
    B = _rigid(Rotation.random(1, random_state=2), rng.normal(size=(1, 3)) * 0.2)[0]
    T_bw = _rigid(slam, rng.normal(size=(n, 3)) * 3)
    hand = _rigid(Rotation.random(n, random_state=3), rng.normal(size=(n, 3)) * 0.3)
    q = slam.as_quat()
    ann = {
        "slam_quat_wxyz": q[:, [3, 0, 1, 2]],
        "slam_trans_xyz": T_bw[:, :3, 3],
        "T_c0_b": B,
        "left_wrist_c0": hand[:, :3, 3],
        "left_wrist_aa": Rotation.from_matrix(hand[:, :3, :3]).as_rotvec(),
        "right_wrist_c0": hand[::-1, :3, 3],
        "right_wrist_aa": Rotation.from_matrix(hand[::-1, :3, :3]).as_rotvec(),
    }
    world = xperience.world_poses(ann)
    action = torch.zeros(n, 32, dtype=torch.float64)
    for s, side in ((0, "left"), (16, "right")):
        action[:, s : s + 3] = torch.tensor(world[f"{side}_wrist_t"])
        action[:, s + 3 : s + 6] = torch.tensor(world[f"{side}_wrist_aa"])
        action[:, s + 6 : s + 16] = 1.0
    for t in range(n):
        cam = torch.tensor(np.concatenate([world["camera_t"][t], world["camera_aa"][t]]))
        out = wrist_to_camera_frame(action, cam).numpy()
        for k in range(n):
            M = B @ T_bw[t] @ np.linalg.inv(T_bw[k]) @ np.linalg.inv(B) @ hand[k]
            np.testing.assert_allclose(out[k, 0:3], M[:3, 3], atol=1e-9)
            got = Rotation.from_rotvec(out[k, 3:6]).as_matrix()
            np.testing.assert_allclose(got, M[:3, :3], atol=1e-9)
        # k = t: the stored c0 pose itself
        np.testing.assert_allclose(out[t, 0:3], hand[t, :3, 3], atol=1e-9)
    with pytest.raises(ValueError, match="non-unit"):
        xperience.world_poses({**ann, "slam_quat_wxyz": np.zeros((n, 4))})


def _ann(device_ts: np.ndarray, actions: list[tuple[int, int, str]], task: str = "Sort") -> dict:
    return {
        "device_ts": device_ts,
        "act_start": np.array([a[0] for a in actions], np.int64),
        "act_end": np.array([a[1] for a in actions], np.int64),
        "act_label": np.array([a[2] for a in actions], dtype=str),
        "main_task": np.array(task),
    }


def _caption(*actions: tuple) -> dict:
    acts = [{"start_frame": s, "end_frame": e, "label": t} for s, e, t in actions]
    return {"segments": [{"Current Action": acts}]}


def test_current_actions_drop_a_caption_with_a_frame_outside_int64() -> None:
    start, end, label, dropped = xperience.current_actions(
        _caption((5, 10, "reach"), (-3, 2**62, "grasp"), ("40", 30, " place "), (1, 2**70, ""))
    )
    np.testing.assert_array_equal(start, [5, -3, 40])
    np.testing.assert_array_equal(end, [10, 2**62, 30])
    assert label.tolist() == ["reach", "grasp", "place"] and dropped == []

    caption = _caption((5, 10, "reach"), (73763454783454786297, 73763454786297, "pick"))
    start, end, label, dropped = xperience.current_actions(caption)
    assert len(start) == len(end) == len(label) == 0
    assert dropped == [(5, 10, "reach"), (73763454783454786297, 73763454786297, "pick")]

    for bad in (1.5, None, True, "12a"):
        with pytest.raises(ValueError):
            xperience.current_actions(_caption((bad, 10, "reach")))


def test_clip_lookup_intervals_are_end_exclusive_runs() -> None:
    pts = np.arange(40) / 20
    ts = np.arange(40, dtype=np.int64) * 50
    ann = _ann(ts, [(0, 500, "reach"), (500, 1000, "grasp"), (1500, 2000, "place")])
    (entry,) = xperience.clip_lookup(ann, pts, 40)
    assert entry == {
        "intervals": [[0, 10, "reach"], [10, 20, "grasp"], [30, 40, "place"]],
        "task": "Sort",
    }
    (no_task,) = xperience.clip_lookup(_ann(ts, [], task=""), pts, 40)
    assert no_task == {"intervals": []}


def test_clip_lookup_checks_device_times_only_for_kept_clips() -> None:
    assert xperience.clip_lookup(_ann(np.zeros(3, np.int64), []), np.arange(10) / 20, 10) == []
    with pytest.raises(ValueError, match="device times"):
        xperience.clip_lookup(_ann(np.zeros(30, np.int64), []), np.arange(40) / 20, 40)


def test_curate_falls_back_to_decoder_pts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = xperience.Xperience()
    leaf = tmp_path / "egocentric_raw" / "xperience_lerobot_v30" / "xperience_00"
    (leaf / "meta").mkdir(parents=True)
    (leaf / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))
    monkeypatch.setattr(reader, "_clips", lambda root, pin: [["a", "b"]])

    def lookup(root: Path, clips: list[str], pts_fn: Callable) -> list[dict]:
        n = 2 if pts_fn is xperience.demux_pts else 1
        return [{"intervals": [[0, 3, pts_fn.__name__]]}] * n

    monkeypatch.setattr(xperience, "leaf_lookup", lookup)
    reader.curate("xperience", tmp_path, catalog.pin("xperience", True))
    lines = (leaf / "meta" / "subtask_lookup.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [
        {"episode_index": 0, "intervals": [[0, 3, "decoder_pts"]]}
    ]


def test_egodex_keypoints_follow_otaheri_order() -> None:
    names = egodex._keypoint_names("left")
    assert len(set(names)) == 21
    assert names[0] == "leftHand" and names[7] == "leftLittleFingerKnuckle"
    assert names[13:17] == [
        f"leftThumb{p}" for p in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
    ]
    assert names[17:] == [f"left{f}FingerTip" for f in ("Index", "Middle", "Ring", "Little")]


def test_egodex_episode_arrays() -> None:
    rng = np.random.default_rng(0)
    lengths = np.array([3, 4])
    n = lengths.sum()
    fits = {"lengths": lengths}
    for side in ("left", "right"):
        fits[f"{side}_wrist_t"] = rng.normal(size=(n, 3)).astype(np.float32)
        fits[f"{side}_wrist_aa"] = rng.normal(size=(n, 3)).astype(np.float32)
        fits[f"{side}_finger_aa"] = rng.normal(size=(n, 45)).astype(np.float32)
        fits[f"{side}_fit_err"] = np.full(n, 0.01, np.float32)
    fits["right_fit_err"][4] = egodex.FIT_ERR
    fits["camera_t"] = rng.normal(size=(n, 3)).astype(np.float32)
    fits["camera_aa"] = rng.normal(size=(n, 3)).astype(np.float32)
    pca = ego.Eigengrasps.load()
    state, action = egodex.episode_arrays(fits, 1, pca)
    assert state.shape == (4, 12) and action.shape == (4, 32)
    np.testing.assert_array_equal(state[:, :3], fits["camera_t"][3:])
    assert not state[:, 6:].any()
    np.testing.assert_array_equal(action[:, :3], fits["left_wrist_t"][3:])
    np.testing.assert_array_equal(
        action[:, 6:16], pca.transform(fits["left_finger_aa"][3:]).astype(np.float32)
    )
    assert not action[1, 16:].any() and action[0, 16:].any()


def test_egodex_text_skips_the_upstream_placeholder() -> None:
    placeholder = {"llm_description": "Description unavailable.", "llm_description2": "None"}
    assert egodex._text(placeholder, "insert_remove_bookshelf") == "Insert remove bookshelf."
    assert egodex._text({**placeholder, "description": " Pour water. "}, "pour") == "Pour water."
    assert egodex._text({"llm_description": "Fold the towel."}, "fold") == "Fold the towel."
    assert egodex._text({"llm_description": "None"}, "roll_ball") == "Roll ball."


@pytest.mark.parametrize(
    ("text", "stripped"),
    [
        (
            "Put block pieces into a jenga tower on a blue tablecloth with a pink background "
            "while sitting.",
            "Put block pieces into a jenga tower.",
        ),
        (
            "Insert own plug into socket while sitting at a table with a blue tablecloth and "
            "pink background.",
            "Insert own plug into socket.",
        ),
        (
            "Add the lid onto each of the three cups placed on the wooden table with a red "
            "background.",
            "Add the lid onto each of the three cups.",
        ),
        ("Fold a small tshirt sitting on a green tablecloth.", "Fold a small tshirt."),
        (
            "Insert a block into a cabinet with both doors open in a sitting position.",
            "Insert a block into a cabinet with both doors open.",
        ),
        ("Unfold the red tablecloth.", "Unfold the red tablecloth."),
        ("Pour.", "Pour."),
    ],
)
def test_egodex_text_loses_its_scene_clauses(text: str, stripped: str) -> None:
    assert egodex.strip_scene(text) == stripped


def test_egodex_archive_reads_extracted_members(tmp_path: Path) -> None:
    for m in ("part1/b_task/0.hdf5", "part1/a_task/1.hdf5", "part1/a_task/0.hdf5"):
        (tmp_path / m).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / m).write_bytes(m.encode())
    archive = egodex.Archive(tmp_path)
    assert archive.folders("part1") == {
        "a_task": ["part1/a_task/0.hdf5", "part1/a_task/1.hdf5"],
        "b_task": ["part1/b_task/0.hdf5"],
    }
    assert archive.read("part1/a_task/1.hdf5") == b"part1/a_task/1.hdf5"


def _ego_leaf(path: Path, seed: int, untracked: bool = False) -> list[Episode]:
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(3):
        n = 60 + 10 * i
        action = np.zeros((n, 32), np.float32)
        action[:, 0:3] = np.cumsum(rng.normal(size=(n, 3)) * 0.01, 0)
        action[:, 16:19] = np.cumsum(rng.normal(size=(n, 3)) * 0.01, 0)
        action[:, 6:16] = rng.normal(size=(n, 10))
        if untracked:
            action[5:25, 0:16] = 0
            action[40:, 16:32] = 0
        state = ego.camera_state(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)) * 0.3)
        eps.append(Episode(i, state, action, "t"))
    write_leaf(eps, path, fps=30, robot_type=ego.ROBOT_TYPE)
    return eps


def _sidecar(leaf: Path, name: str) -> dict:
    return json.loads((leaf / "meta" / name).read_text())


def test_ego_wrist_writes_the_horizon_sidecar(tmp_path: Path) -> None:
    from reki.policy.train_config import TrainConfig

    leaves = [tmp_path / "x_00", tmp_path / "x_01"]
    eps = [_ego_leaf(leaf, seed) for seed, leaf in enumerate(leaves)]
    train = TrainConfig(chunk_duration_s=0.4, action_pace={"egodex": 0.6})
    ego.ego_wrist("egodex", tmp_path, leaves, False, train, windows=2, workers=1)
    rows = []
    for leaf_eps in eps:
        for ep in leaf_eps:
            n = len(ep.action)
            for s in np.linspace(0, n - 2, 2).astype(int):
                chunk = interpolate_wrist_chunk(torch.tensor(ep.action[s : s + 12]), 0.6)
                wp = wrist_to_camera_frame(chunk, torch.tensor(ep.state[s, :6])).numpy()
                rows.append(wp[:, ego.WRIST_DIMS])
    ref = np.concatenate(rows)
    for leaf in leaves:
        out = _sidecar(leaf, "norm_stats_ego_wrist_0.4s_pace0.6.json")
        assert (out["window"], out["fps"], out["pace"], out["chunk_duration_s"]) == (
            12,
            30,
            0.6,
            0.4,
        )
        assert out["sampling"]["windows"] == 2 and out["sampling"]["rows"] == len(ref)
        np.testing.assert_array_equal(out["qk99"], np.percentile(ref, 99, 0))
        assert "qk01" not in json.loads((leaf / "meta" / "stats.json").read_text())["action"]


def test_ego_wrist_processes_give_the_serial_band(tmp_path: Path) -> None:
    from reki.policy.train_config import TrainConfig

    leaves = [tmp_path / f"x_{i:02d}" for i in range(3)]
    for seed, leaf in enumerate(leaves):
        _ego_leaf(leaf, seed, untracked=seed == 1)
    train = TrainConfig()
    name = "norm_stats_ego_wrist_1s_pace1.json"
    ego.ego_wrist(
        "xperience", tmp_path, leaves, False, train, windows=3, interleave=True, workers=1
    )
    serial = _sidecar(leaves[0], name)
    ego.ego_wrist(
        "xperience", tmp_path, leaves, False, train, windows=3, interleave=True, workers=3
    )
    for leaf in leaves:
        assert _sidecar(leaf, name) == serial


def test_ego_wrist_leaves_out_untracked_rows(tmp_path: Path) -> None:
    from reki.policy.train_config import TrainConfig

    leaf = tmp_path / "x_00"
    eps = _ego_leaf(leaf, 0, untracked=True)
    train = TrainConfig(chunk_duration_s=None, chunk_size=50)
    ego.ego_wrist("xperience", tmp_path, [leaf], False, train, windows=3, workers=1)
    left, right = [], []
    for ep in eps:
        for s in np.linspace(0, len(ep.action) - 2, 3).astype(int):
            chunk = torch.tensor(ep.action[s : s + 50])
            wp = wrist_to_camera_frame(chunk, torch.tensor(ep.state[s, :6])).numpy()
            left.append(wp[chunk[:, 0:16].numpy().any(1), 0:6])
            right.append(wp[chunk[:, 16:32].numpy().any(1), 16:22])
    out = _sidecar(leaf, "norm_stats_ego_wrist_50f_pace1.json")
    ref = np.concatenate([np.percentile(np.concatenate(h), 1, 0) for h in (left, right)])
    np.testing.assert_array_equal(out["qk01"], ref)


def test_loader_stamps_the_matching_sidecar_or_names_the_command(tmp_path: Path) -> None:
    from reki.data.robotics.normalization import apply_ego_wrist_stats

    (tmp_path / "meta").mkdir()
    stats = {"action": {"qk01": torch.full((32,), -9.0), "qk99": torch.full((32,), 9.0)}}
    kw = dict(name="egodex", chunk_duration_s=1.0, chunk_size=50, window=30, fps=30, pace=0.6)
    with pytest.raises(
        FileNotFoundError,
        match="pretrain_data.py stats egodex --config",
    ):
        apply_ego_wrist_stats(stats, tmp_path, **kw)  # type: ignore[arg-type]
    side = {"window": 30, "fps": 30, "pace": 0.6, "qk01": [-1.0] * 12, "qk99": [1.0] * 12}
    path = tmp_path / "meta" / "norm_stats_ego_wrist_1s_pace0.6.json"
    path.write_text(json.dumps({**side, "fps": 20}))
    with pytest.raises(ValueError, match="training needs"):
        apply_ego_wrist_stats(stats, tmp_path, **kw)  # type: ignore[arg-type]
    path.write_text(json.dumps(side))
    apply_ego_wrist_stats(stats, tmp_path, **kw)  # type: ignore[arg-type]
    assert stats["action"]["qk99"][ego.WRIST_DIMS].eq(1.0).all()
    assert (
        stats["action"]["qk99"][6:16].eq(9.0).all() and stats["action"]["qk99"][22:].eq(9.0).all()
    )
    with pytest.raises(FileNotFoundError, match="norm_stats_ego_wrist_20f_pace1.json is missing"):
        apply_ego_wrist_stats(
            stats, tmp_path, **{**kw, "chunk_duration_s": None, "chunk_size": 20, "pace": 1.0}
        )  # type: ignore[arg-type]


def test_mano_chunks_pack_whole_episodes_with_a_padding_row() -> None:
    from pretrain.readers.mano_fit import CHUNK, chunks

    cap = CHUNK - 1
    assert chunks([3, 4]) == [[(0, 3), (3, 4)]]
    assert chunks([cap - 2, 2, 1]) == [[(0, cap - 2), (cap - 2, 2)], [(cap, 1)]]
    assert chunks([5, cap + 3, 2]) == [[(0, 5)], [(5, cap)], [(5 + cap, 3)], [(8 + cap, 2)]]


def test_egodex_dedup_labels_are_task_folders(tmp_path: Path) -> None:
    for m in ("part1/fry_bread/0.hdf5", "part1/fry_bread/1.hdf5", "part1/fry_egg/0.hdf5"):
        (tmp_path / "upstream" / "egodex" / m).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "upstream" / "egodex" / m).write_bytes(b"")
    pin = egodex.EgodexPin(
        egodex.PIN.sources, 3, None, folders=(("part1", "fry_bread"), ("part1", "fry_egg"))
    )
    reader = egodex.Egodex()
    assert reader.dedup_labels("egodex", tmp_path, pin, tmp_path / "egodex_00") == [
        "part1/fry_bread"
    ]
    assert reader.dedup_labels("egodex", tmp_path, pin, tmp_path / "egodex_02") == ["part1/fry_egg"]


def test_egodex_shards_only_fit_and_the_unsharded_build_writes_every_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    folders = (("part1", "fry_bread"), ("part1", "fry_egg"), ("part2", "wipe"))
    for part, task in folders:
        for k in range(2):
            m = tmp_path / "upstream" / "egodex" / part / task / f"{k}.hdf5"
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_bytes(b"")
    pin = egodex.EgodexPin(egodex.PIN.sources, 6, None, folders=folders)
    fitted: list[str] = []

    def fit(archive: egodex.Archive, members: list[str], out: Path, models: Path) -> None:
        fitted.append(out.relative_to(out.parents[1]).as_posix())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")

    written: dict[int, list[int]] = {}

    def write(
        reader: egodex.Egodex, name: str, root: Path, src: Path, leaf: int, refs: list
    ) -> int:
        written[leaf] = [g for g, _ in refs]
        return leaf

    monkeypatch.setattr(egodex, "fit_folder", fit)
    monkeypatch.setattr(egodex, "mano_dir", lambda root: tmp_path)
    monkeypatch.setattr(egodex, "_write_leaf", write)
    monkeypatch.setattr(
        egodex, "ProcessPoolExecutor", lambda workers, mp_context: ThreadPoolExecutor(workers)
    )
    reader = egodex.Egodex()
    reader.build("egodex", tmp_path, pin, (1, 2))
    assert fitted == ["part1/fry_egg.npz"] and not written
    assert [p.name for p in reader.leaves("egodex", tmp_path, pin)] == [
        f"egodex_{i:02d}" for i in range(6)
    ]
    reader.build("egodex", tmp_path, pin, (0, 1))
    assert fitted == ["part1/fry_egg.npz", "part1/fry_bread.npz", "part2/wipe.npz"]
    assert written == {g: [g] for g in range(6)}
