"""The InternData-A1 reader: column merge, franka EE delta, episodes meta from frame counts."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pretrain.readers import interndata
from scipy.spatial.transform import Rotation as R

pytestmark = pytest.mark.cpu


def _pose(rot: R, xyz: np.ndarray) -> np.ndarray:
    xyzw = rot.as_quat()
    return np.concatenate([xyz, xyzw[:, [3]], xyzw[:, :3]], axis=1).astype(np.float32)


def test_franka_ee_delta() -> None:
    rng = np.random.default_rng(0)
    state_rot, delta = R.random(5, random_state=1), R.from_rotvec(rng.normal(0, 0.1, (5, 3)))
    xyz = rng.normal(size=(5, 3))
    state = _pose(state_rot, xyz)
    action = _pose(delta * state_rot, xyz + 0.01)
    grip = rng.uniform(size=(5, 1)).astype(np.float32)
    out = interndata.franka_ee_delta(state, action, grip)
    assert out.shape == (5, 7) and out.dtype == np.float32
    np.testing.assert_allclose(out[:, :3], 0.01, atol=1e-6)
    np.testing.assert_allclose(out[:, 3:6], delta.as_rotvec(), atol=1e-5)
    np.testing.assert_array_equal(out[:, 6:], grip)


def _bimanual_table(n: int, joints: int) -> pa.Table:
    rng = np.random.default_rng(0)

    def fsl(k: int) -> pa.Array:
        flat = pa.array(rng.normal(size=n * k).astype(np.float32))
        return pa.FixedSizeListArray.from_arrays(flat, k)

    cols = {}
    for kind in ("states", "actions"):
        for side in ("left", "right"):
            cols[f"{kind}.{side}_joint.position"] = fsl(joints)
            cols[f"{kind}.{side}_gripper.position"] = pa.array(
                rng.uniform(size=n).astype(np.float32)
            )
    return pa.table(cols)


def test_split_aloha_negates_elbows() -> None:
    table = _bimanual_table(4, 6)
    state, names, action, _ = interndata.merge("split_aloha", table)
    left = np.stack(table.column("states.left_joint.position").to_pylist())
    right = np.stack(table.column("states.right_joint.position").to_pylist())
    assert state.shape == (4, 14) and len(names) == 14
    np.testing.assert_array_equal(state[:, 2], -left[:, 2])
    np.testing.assert_array_equal(state[:, 9], -right[:, 2])
    np.testing.assert_array_equal(state[:, [0, 1, 3, 4, 5]], left[:, [0, 1, 3, 4, 5]])
    raw_action = np.stack(table.column("actions.left_joint.position").to_pylist())
    np.testing.assert_array_equal(action[:, 2], -raw_action[:, 2])


def test_genie1_merge_untouched() -> None:
    table = _bimanual_table(3, 7)
    state, _, action, _ = interndata.merge("genie1", table)
    assert state.shape == action.shape == (3, 16)
    left = np.stack(table.column("states.left_joint.position").to_pylist())
    np.testing.assert_array_equal(state[:, :7], left)


def test_camera_columns() -> None:
    cols = interndata.camera_columns([2, 3, 4, 1], [5, 5], 10, "cam")
    assert cols["videos/cam/file_index"] == [0, 0, 1, 1]
    np.testing.assert_allclose(cols["videos/cam/from_timestamp"], [0, 0.2, 0, 0.4])
    np.testing.assert_allclose(cols["videos/cam/to_timestamp"], [0.2, 0.5, 0.4, 0.5])


def test_camera_columns_raises() -> None:
    with pytest.raises(ValueError, match="straddles"):
        interndata.camera_columns([3, 3], [4, 2], 10, "cam")
    with pytest.raises(ValueError, match="video frames"):
        interndata.camera_columns([3, 3], [4, 3], 10, "cam")


def test_untar_skips_dangling_episodes_link(tmp_path: Path) -> None:
    archive = tmp_path / "task.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = json.dumps({"total_episodes": 1}).encode()
        member = tarfile.TarInfo("task/meta/info.json")
        member.size = len(info)
        tf.addfile(member, io.BytesIO(info))
        link = tarfile.TarInfo("task/" + str(interndata.EPISODES_FILE))
        link.type = tarfile.SYMTYPE
        link.linkname = "/mnt/shared-storage-user/elsewhere/file-000.parquet"
        tf.addfile(link)
    task = interndata.untar(archive)
    assert (task / "meta" / "info.json").exists()
    assert not (task / interndata.EPISODES_FILE).is_symlink()


def test_untar_modes_follow_the_umask(tmp_path: Path) -> None:
    import os
    import stat

    archive = tmp_path / "task.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, mode in [("task/videos", 0o700), ("task/meta/info.json", 0o600)]:
            member = tarfile.TarInfo(name)
            member.mode = mode
            if name.endswith("videos"):
                member.type = tarfile.DIRTYPE
                tf.addfile(member)
            else:
                member.size = 2
                tf.addfile(member, io.BytesIO(b"{}"))
    mask = os.umask(0o027)
    try:
        task = interndata.untar(archive)
    finally:
        os.umask(mask)
    assert stat.S_IMODE((task / "meta" / "info.json").stat().st_mode) == 0o640
    assert stat.S_IMODE((task / "videos").stat().st_mode) == 0o750


def test_upstream_episodes_mismatch_raises(tmp_path: Path) -> None:
    runs = [(0, 0, 0, 3, ["do it"]), (1, 0, 3, 5, ["do it"])]
    cams = {"cam": interndata.camera_columns([3, 2], [5], 10, "cam")}
    table = interndata.episodes_table(runs, cams)
    path = tmp_path / interndata.EPISODES_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    interndata.check_upstream_episodes(tmp_path, table)
    moved = [(0, 0, 0, 3, ["do it"]), (1, 1, 3, 5, ["do it"])]
    interndata.check_upstream_episodes(tmp_path, interndata.episodes_table(moved, cams))
    bad = interndata.episodes_table([(0, 0, 0, 2, ["do it"]), (1, 0, 2, 5, ["do it"])], cams)
    with pytest.raises(ValueError, match="differs"):
        interndata.check_upstream_episodes(tmp_path, bad)


@pytest.mark.parametrize("robot", list(interndata.TASKS))
def test_tasks_in_upstream_order(robot: str) -> None:
    """Dedup keeps the first copy in leaf order: the leaves follow the repo path order."""
    tasks = interndata.TASKS[robot]
    assert list(tasks) == sorted(tasks) and len(set(tasks)) == len(tasks)
    assert len({rel.split("/")[1] for rel in tasks}) == len(tasks)
