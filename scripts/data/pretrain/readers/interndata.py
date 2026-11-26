"""Reader for InternData-A1 (``sim_updated_lerobotv30``): one tarball per task, each an
upstream LeRobot v3.0 leaf with InternUtopia's named columns.

Per task: untar, merge the named columns into ``observation.state`` / ``action``, rebuild
``meta/episodes`` from the data parquets and the videos' frame counts, write ``info.json`` /
``stats.json``. Videos are hard-linked from upstream; ``tasks.parquet`` is copied.

- franka: state = EE pose ``[xyz, rpy]`` + gripper (7); action = base-frame EE delta
  ``[dxyz, rotvec, gripper]`` from the ``ee_to_robot_pose`` quaternions (robosuite OSC_POSE
  delta convention, gripper absolute).
- split_aloha: L joints (6) + L gripper + R joints (6) + R gripper = 14; the elbows (dims 2, 9)
  are negated on state and action (a reflection about 0 vs RoboTwin's ``aloha`` convention).
- genie1: L joints (7) + L gripper + R joints (7) + R gripper = 16.

Upstream split_aloha / genie1 ship ``meta/episodes/chunk-000/file-000.parquet`` as a symlink
into the authors' storage (``/mnt/shared-storage-user/...``), so the episodes come from the
data parquets' ``episode_index`` runs, checked against ``info.json``. Each camera's video is a
multi-episode concatenation chunked into its own files; episodes are laid onto each camera's
files by their ``ffprobe`` frame counts. Where upstream has a real episodes parquet (franka),
the rebuilt table must equal it.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shared import link_tree

from ..common import download
from ..common.download import upstream_dir
from ..common.stats import op
from ..common.v30 import concat_columns, list_column, read_columns, table_columns, video_keys
from ..common.video import ffprobe
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir

logger = logging.getLogger(__name__)

REPO = "InternRobotics/InternData-A1"
REVISION = "6904fb6464a1096b4c1730f652ce40f879751051"
PREFIX = "sim_updated_lerobotv30"

STATE_COLS = {
    "franka": ["states.gripper.pose", "states.gripper.position"],
    "split_aloha": [
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.right_joint.position",
        "states.right_gripper.position",
    ],
    "genie1": [
        "states.left_joint.position",
        "states.left_gripper.position",
        "states.right_joint.position",
        "states.right_gripper.position",
    ],
}
ACTION_COLS = {
    "split_aloha": [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    ],
    "genie1": [
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    ],
}
FRANKA_POSE_COLS = [
    "states.ee_to_robot_pose",
    "actions.ee_to_robot_pose",
    "actions.gripper.position",
]
"""franka: the action's source columns (current pose, commanded pose, gripper)."""
FRANKA_ACTION_NAMES = ["d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper"]
NEGATE_DIMS = {"split_aloha": [2, 9]}
META_COLS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
DROPPED_FEATURE_PREFIXES = ("states.", "actions.", "master_actions.")
DROPPED_FEATURE_SUFFIXES = ("_intrinsics", "_extrinsics")
EPISODES_FILE = Path("meta/episodes/chunk-000/file-000.parquet")

# ------------------------------------------------------------------ tasks per collection

FRANKA_TASKS = (
    "articulation_tasks/close_the_electriccooker",
    "articulation_tasks/close_the_laptop",
    "articulation_tasks/close_the_microwave_part1",
    "articulation_tasks/close_the_microwave_part2",
    "articulation_tasks/close_the_pot",
    "articulation_tasks/close_the_trashcan",
    "articulation_tasks/close_the_washingmachine_left",
    "articulation_tasks/close_the_washingmachine_right",
    "articulation_tasks/open_the_laptop",
    "articulation_tasks/open_the_microwave_part1",
    "articulation_tasks/open_the_microwave_part2",
    "articulation_tasks/open_the_pot",
    "articulation_tasks/open_the_trashcan",
    "basic_tasks/collect_the_shoes_language",
    "basic_tasks/organize_the_alarm_clocks_language",
    "basic_tasks/organize_the_brushes_language",
    "basic_tasks/organize_the_colorful_cups_language",
    "basic_tasks/organize_the_glues_language",
    "basic_tasks/organize_the_teaboxes_language",
    "basic_tasks/pick_the_object_into_trashcan",
    "basic_tasks/pick_the_priced_item_language",
    "basic_tasks/select_a_drink_language",
    "basic_tasks/sort_the_table_waste_language",
    "basic_tasks/track_the_targets",
    "long_horizon_tasks/manipulate_the_microwave",
    "long_horizon_tasks/put_the_trashcan_in_trashcan_long",
    "long_horizon_tasks/sort_bowls_and_cups",
    "long_horizon_tasks/sweep_the_trash",
)
SPLIT_ALOHA_TASKS = (
    "articulation_tasks/close_the_microwave_left_arm",
    "articulation_tasks/close_the_microwave_right_arm",
    "articulation_tasks/open_the_microwave_from_scratch",
    "articulation_tasks/open_the_microwave_left_arm",
    "articulation_tasks/open_the_microwave_right_arm",
    *(
        f"articulation_tasks/{verb}_the_storagefurniture_{arm}_arm_{level}"
        for verb in ("pull", "push")
        for arm in ("left", "right")
        for level in ("bottom", "middle", "top")
    ),
    *(
        f"articulation_tasks/rotate_the_{hearth}_hearth_{arm}_arm_{way}"
        for hearth in ("left", "right")
        for arm in ("left", "right")
        for way in ("clockwise", "counterclockwise")
    ),
    "basic_tasks/collect_the_shoes",
    "basic_tasks/collect_the_shoes_language",
    "basic_tasks/gather_toothbrush_and_toothpaste_part1",
    "basic_tasks/gather_toothbrush_and_toothpaste_part2",
    "basic_tasks/hang_the_cup_on_rack_left_arm",
    "basic_tasks/hang_the_cup_on_rack_right_arm",
    "basic_tasks/insert_the_flower_in_vase_left_arm",
    "basic_tasks/insert_the_flower_in_vase_right_arm",
    "basic_tasks/insert_the_markpen_in_penholder_left_arm",
    "basic_tasks/insert_the_markpen_in_penholder_right_arm",
    "basic_tasks/organize_the_alarm_clocks",
    "basic_tasks/organize_the_alarm_clocks_language",
    "basic_tasks/organize_the_brushes",
    "basic_tasks/organize_the_brushes_language",
    "basic_tasks/organize_the_glues_language",
    "basic_tasks/organize_the_teaboxes_language",
    "basic_tasks/pick_beef_sandwich_on_conveyor",
    "basic_tasks/pick_ham_sandwich_on_conveyor",
    "basic_tasks/pick_the_priced_item",
    "basic_tasks/pick_the_priced_item_language",
    "basic_tasks/pour_Baijiu_left_arm",
    "basic_tasks/pour_Baijiu_right_arm",
    "basic_tasks/pour_redwine_left_arm",
    "basic_tasks/pour_redwine_right_arm",
    "basic_tasks/pour_water_left_arm",
    "basic_tasks/pour_water_right_arm",
    "basic_tasks/select_a_drink",
    "basic_tasks/select_a_drink_language",
    "basic_tasks/sort_the_table_waste",
    "basic_tasks/sort_the_table_waste_language",
    "basic_tasks/sort_the_tray_on_rack",
    *(f"basic_tasks/stack_the_boxes_part{i}" for i in range(1, 7)),
    "basic_tasks/track_the_targets",
    "basic_tasks/watering_plants_left_arm",
    "basic_tasks/watering_plants_right_arm",
    *(
        f"long_horizon_tasks/clean_the_dirt_with_{tool}_{arm}_arm"
        for tool in ("brown_cloth", "sponge", "white_cloth")
        for arm in ("left", "right")
    ),
    "long_horizon_tasks/collaborate_assemble_a_beef_sandwich",
    "long_horizon_tasks/collect_the_glues",
    "long_horizon_tasks/gather_the_teaboxes",
    "long_horizon_tasks/make_a_sandwich",
    *(f"long_horizon_tasks/pack_in_objects_part{i}" for i in range(1, 4)),
    *(f"long_horizon_tasks/pack_out_objects_part{i}" for i in range(1, 5)),
    *(f"long_horizon_tasks/sort_the_rubbish_part{i}" for i in range(1, 4)),
    "long_horizon_tasks/stack_a_beef_sandwich",
    "long_horizon_tasks/sweep_the_trash",
)
GENIE1_TASKS = tuple(
    f"basic_tasks/{t}"
    for t in (
        "scan_qrcode_part1",
        "scan_qrcode_part2",
        "store_the_eggs_part1",
        "store_the_eggs_part2",
        "take_shelf_items_to_cart_left_arm_part1",
        "take_shelf_items_to_cart_left_arm_part2",
        "take_shelf_items_to_cart_right_arm_part1",
        "take_shelf_items_to_cart_right_arm_part2",
    )
)
TASKS = {"franka": FRANKA_TASKS, "split_aloha": SPLIT_ALOHA_TASKS, "genie1": GENIE1_TASKS}
"""``<category>/<task>`` per robot: the 123 trained tasks. ``pick_and_place_tasks`` is not
used, nor ``split_aloha/make_a_sandwich_language`` (its tarball nests the task under an absolute
upstream path)."""

SUBSET_TASKS = {
    "franka": (
        "articulation_tasks/close_the_washingmachine_left",
        "articulation_tasks/close_the_pot",
        "long_horizon_tasks/manipulate_the_microwave",
    ),
    "split_aloha": ("basic_tasks/select_a_drink", "basic_tasks/select_a_drink_language"),
    "genie1": ("basic_tasks/take_shelf_items_to_cart_left_arm_part1",),
}
"""``--subset`` tarballs. franka: the smallest one, ``close_the_pot`` (a same-text duplicate
pair) and ``manipulate_the_microwave`` (an empty leaf: every episode a ``sudden_change`` hit);
split_aloha: the smallest twin pair (900 same-text duplicate pairs); genie1: the smallest."""


def tarball(robot: str, rel: str) -> str:
    """The repo path of one task's tarball."""
    category, task = rel.split("/")
    return f"{PREFIX}/{category}/{robot}/{task}.tar.gz"


def source(robot: str, tasks: tuple[str, ...]) -> HfSnapshot:
    return HfSnapshot(REPO, REVISION, tuple(tarball(robot, t) for t in tasks))


# ------------------------------------------------------------------ column merge


def franka_ee_delta(state_pose: np.ndarray, action_pose: np.ndarray, grip: np.ndarray):
    """``[dxyz, rotvec, gripper]``: commanded ``ee_to_robot_pose`` ⊖ current, in the base frame.
    Poses are ``[xyz, quat wxyz]``; the rotation delta is ``R_action · R_state⁻¹``."""
    from scipy.spatial.transform import Rotation as R

    dpos = action_pose[:, :3] - state_pose[:, :3]
    sq, aq = state_pose[:, [4, 5, 6, 3]], action_pose[:, [4, 5, 6, 3]]
    drot = (R.from_quat(aq) * R.from_quat(sq).inv()).as_rotvec()
    return np.concatenate([dpos, drot, grip], axis=1).astype(np.float32)


def merge(robot: str, table: pa.Table) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """``(state, state names, action, action names)`` of one upstream data table."""
    action_cols = ACTION_COLS.get(robot, FRANKA_POSE_COLS)
    cols = table_columns(table, [*STATE_COLS[robot], *action_cols])
    state, state_names = concat_columns(cols, STATE_COLS[robot])
    if robot == "franka":
        pose, target, grip = (concat_columns(cols, [c])[0] for c in action_cols)
        action = franka_ee_delta(pose, target, grip)
        action_names = list(FRANKA_ACTION_NAMES)
    else:
        action, action_names = concat_columns(cols, action_cols)
    for d in NEGATE_DIMS.get(robot, []):
        state[:, d] *= -1.0
        action[:, d] *= -1.0
    return state, state_names, action, action_names


# ------------------------------------------------------------------ episodes meta


def nb_frames(video: Path) -> int:
    """Frame count from the mp4 stream header."""
    out = ffprobe(video, "-show_entries", "stream=nb_frames", "-of", "csv=p=0")
    if not out.isdigit():
        raise RuntimeError(f"{video}: ffprobe nb_frames gave {out!r}")
    return int(out)


def camera_columns(lengths: list[int], frames: list[int], fps: int, cam: str) -> dict[str, list]:
    """Per episode (in order) the file, from and to timestamp of one camera whose files hold
    ``frames`` frames each. Each episode must lie wholly inside one file."""
    if sum(frames) != sum(lengths):
        raise ValueError(f"{cam}: {sum(frames)} video frames, episodes hold {sum(lengths)}")
    file_index: list[int] = []
    from_ts: list[float] = []
    to_ts: list[float] = []
    f = used = 0
    for i, n in enumerate(lengths):
        file_index.append(f)
        from_ts.append(used / fps)
        used += n
        to_ts.append(used / fps)
        if f < len(frames) - 1 and used >= frames[f]:
            if used != frames[f]:
                raise ValueError(f"{cam}: episode {i} straddles the end of file {f}")
            used = 0
            f += 1
    return {
        f"videos/{cam}/chunk_index": [0] * len(lengths),
        f"videos/{cam}/file_index": file_index,
        f"videos/{cam}/from_timestamp": from_ts,
        f"videos/{cam}/to_timestamp": to_ts,
    }


Run = tuple[int, int, int, int, list[str]]
"""One episode in the data: (episode, data file, first frame, end frame, task texts)."""


def episodes_table(runs: list[Run], cams: dict[str, dict[str, list]]) -> pa.Table:
    """The v3.0 ``meta/episodes`` table."""
    n = len(runs)
    cols: dict[str, pa.Array] = {
        "episode_index": pa.array([r[0] for r in runs], pa.int64()),
        "data/chunk_index": pa.array([0] * n, pa.int64()),
        "data/file_index": pa.array([r[1] for r in runs], pa.int64()),
        "dataset_from_index": pa.array([r[2] for r in runs], pa.int64()),
        "dataset_to_index": pa.array([r[3] for r in runs], pa.int64()),
        "length": pa.array([r[3] - r[2] for r in runs], pa.int64()),
        "tasks": pa.array([r[4] for r in runs], pa.list_(pa.string())),
        "meta/episodes/chunk_index": pa.array([0] * n, pa.int64()),
        "meta/episodes/file_index": pa.array([0] * n, pa.int64()),
    }
    for cam_cols in cams.values():
        for k, v in cam_cols.items():
            cols[k] = pa.array(v, pa.float64() if k.endswith("timestamp") else pa.int64())
    return pa.table(cols)


def check_upstream_episodes(src: Path, table: pa.Table) -> None:
    """Where upstream ships a real episodes parquet, the rebuilt table must equal it, except
    ``data/file_index``: upstream gives the first episode of each data file the previous file's
    index (the loader does not read it)."""
    path = src / EPISODES_FILE
    if path.is_symlink() or not path.exists():
        return
    up = pq.read_table(path)
    for col in table.column_names:
        if col == "data/file_index":
            continue
        a, b = table.column(col).to_pylist(), up.column(col).to_pylist()
        if col.endswith("timestamp"):
            same = np.allclose(a, b, rtol=0, atol=1e-6)
        else:
            same = a == b
        if not same:
            raise ValueError(f"{path}: rebuilt {col} differs from upstream")


# ------------------------------------------------------------------ untar


def _skip_dangling_episodes_link(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    """tarfile ``data`` filter, except upstream's absolute ``meta/episodes`` symlink is skipped
    (the episodes table is rebuilt)."""
    if member.issym() and member.name.endswith(str(EPISODES_FILE)):
        return None
    return tarfile.data_filter(member, dest)


def untar(archive: Path) -> Path:
    """Extract one task tarball next to itself, once; returns the task dir."""
    task_dir = archive.with_name(archive.name.removesuffix(".tar.gz"))
    download.untar(archive, archive.parent, member_filter=_skip_dangling_episodes_link)
    if not (task_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{archive}: no {task_dir.name}/meta/info.json inside")
    return task_dir


# ------------------------------------------------------------------ one leaf


def _drop_feature(key: str) -> bool:
    return key.startswith(DROPPED_FEATURE_PREFIXES) or key.endswith(DROPPED_FEATURE_SUFFIXES)


def build_leaf(robot: str, src: Path, out: Path) -> int:
    """Write one merged leaf; returns its episode count."""
    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats

    info = json.loads((src / "meta" / "info.json").read_text())
    missing = [
        c
        for c in STATE_COLS[robot] + ACTION_COLS.get(robot, FRANKA_POSE_COLS)
        if c not in info["features"]
    ]
    if missing:
        raise KeyError(f"{src}: missing columns {missing}")
    tasks = pq.read_table(src / "meta" / "tasks.parquet").to_pandas()
    task_text = dict(zip(tasks["task_index"], tasks.index))
    if out.exists() or out.is_symlink():
        shutil.rmtree(out)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    runs: list[Run] = []
    states, actions = [], []
    names: tuple[list[str], list[str]] = ([], [])
    data_files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    expected_files = [
        src / info["data_path"].format(chunk_index=0, file_index=i) for i in range(len(data_files))
    ]
    if data_files != expected_files:
        raise ValueError(f"{src}: data files are not chunk-000/file-000..N")
    offset = 0
    for file_index, path in enumerate(data_files):
        table = pq.read_table(path)
        state, state_names, action, action_names = merge(robot, table)
        names = (state_names, action_names)
        meta = {c: table.column(c) for c in META_COLS}
        (out / path.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"observation.state": list_column(state), "action": list_column(action), **meta}
            ),
            out / path.relative_to(src),
        )
        epi = read_columns(path, ["episode_index", "index", "task_index"])
        if not np.array_equal(epi["index"], np.arange(offset, offset + len(state))):
            raise ValueError(f"{path}: index is not contiguous from {offset}")
        starts = np.flatnonzero(np.diff(epi["episode_index"], prepend=-1))
        for s, e in zip(starts, [*starts[1:], len(state)]):
            ep = int(epi["episode_index"][s])
            if runs and runs[-1][0] == ep:
                raise ValueError(f"{path}: episode {ep} spans data files")
            texts = [task_text[t] for t in np.unique(epi["task_index"][s:e])]
            runs.append((ep, file_index, offset + int(s), offset + int(e), texts))
        offset += len(state)
        states.append(state)
        actions.append(action)

    eps = [r[0] for r in runs]
    if eps != list(range(len(eps))):
        raise ValueError(f"{src}: data episodes are not 0..{len(eps) - 1} in order")
    if (len(eps), offset) != (info["total_episodes"], info["total_frames"]):
        raise ValueError(
            f"{src}: data has {len(eps)} episodes / {offset} frames, info.json "
            f"{info['total_episodes']} / {info['total_frames']}"
        )

    lengths = [r[3] - r[2] for r in runs]
    cams = {}
    for cam in video_keys(info):
        files = sorted((src / "videos" / cam / "chunk-000").glob("file-*.mp4"))
        cams[cam] = camera_columns(lengths, [nb_frames(f) for f in files], info["fps"], cam)
    table = episodes_table(runs, cams)
    check_upstream_episodes(src, table)
    pq.write_table(table, out / EPISODES_FILE)

    link_tree(src / "videos", out / "videos")
    shutil.copy(src / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")

    info["codebase_version"] = "v3.0"
    info["features"] = {k: v for k, v in info["features"].items() if not _drop_feature(k)}
    for key, arr, feat_names in (
        ("observation.state", states, names[0]),
        ("action", actions, names[1]),
    ):
        info["features"][key] = {
            "dtype": "float32",
            "shape": [arr[0].shape[1]],
            "names": feat_names,
        }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    stats = json.loads((src / "meta" / "stats.json").read_text())
    stats = {k: v for k, v in stats.items() if not _drop_feature(k)}
    for key, arr in (("observation.state", states), ("action", actions)):
        s = get_feature_stats(
            np.concatenate(arr), axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES
        )
        stats[key] = {k: np.asarray(v).reshape(-1).tolist() for k, v in s.items()}
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    return len(runs)


# ------------------------------------------------------------------ reader


class InternData(Reader):
    """One collection: each task tarball → ``<root>/<spec.collection>/<task>``."""

    def __init__(self, robot: str) -> None:
        self.robot = robot

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [collection_dir(name, root) / rel.split("/")[1] for rel in tasks_of(pin)]

    def download(self, name: str, root: Path, pin: Pin) -> None:
        super().download(name, root, pin)
        up = upstream_dir(root, pin.sources[0])
        for rel in tasks_of(pin):
            untar(up / tarball(self.robot, rel))

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        up = upstream_dir(root, pin.sources[0])
        i, n = shard
        jobs = list(zip(tasks_of(pin), self.leaves(name, root, pin)))
        for rel, out in jobs[i::n]:
            src = up / tarball(self.robot, rel).removesuffix(".tar.gz")
            episodes = build_leaf(self.robot, src, out)
            logger.info("%s: %d episodes → %s", name, episodes, out)


def tasks_of(pin: Pin) -> list[str]:
    """The ``<category>/<task>`` tarballs a pin downloads."""
    (source,) = pin.sources
    assert isinstance(source, HfSnapshot) and source.allow is not None
    return ["/".join(p.removesuffix(".tar.gz").split("/")[1::2]) for p in source.allow]


SUBSET_COUNTS = {
    "franka": (5752, 3722475),
    "split_aloha": (4500, 1606904),
    "genie1": (1516, 497277),
}
"""Per collection, the ``--subset`` episodes and frames."""


def _pin(robot: str, episodes: int, frames: int) -> Pin:
    subset = Pin((source(robot, SUBSET_TASKS[robot]),), *SUBSET_COUNTS[robot])
    return Pin((source(robot, TASKS[robot]),), episodes, frames, subset=subset)


PINS = {
    "interndata_franka": _pin("franka", 85798, 52372554),
    "interndata_split_aloha": _pin("split_aloha", 150514, 121442702),
    "interndata_genie1": _pin("genie1", 13729, 7616836),
}
"""Episode / frame totals of the 123 trained leaves. Duplicates: franka 5 same-text pairs inside
track_the_targets / close_the_pot; split_aloha 7,248 same-text pairs between a task and its
``*_language`` (or renamed long-horizon) twin. Two franka leaves lose every episode to
``sudden_change``."""

CURATE = ("frozen", "sudden_change")
RECIPES = {
    "interndata_franka": Recipe(
        InternData("franka"),
        CURATE,
        (op("kept_qk"), op("global_qk"), op("euler", pooled=True)),
    ),
    "interndata_split_aloha": Recipe(
        InternData("split_aloha"), CURATE, (op("kept_qk"), op("global_qk"))
    ),
    "interndata_genie1": Recipe(InternData("genie1"), CURATE, (op("kept_qk"), op("global_qk"))),
}
