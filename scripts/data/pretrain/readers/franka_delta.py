"""Readers of the Franka joint datasets relabelled to single-arm ΔEEF (``single_arm_ee_euler``):
berkeley_rpt (IPEC LeRobot v2.0) and RoboSet (raw HDF5).

Upstream state = ``[7 abs joints, gripper]``, action = ``[7 joint targets or deltas, gripper]``.
Forward kinematics of the robosuite Panda (``mj_forward`` only, no renderer) gives the EE pose
of each state row:

    state[8]  = [x, y, z, rotvec(3), pad 0, gripper state]
    action[7] = [FK(t+1) - FK(t) position, rotvec(R(t+1) R(t)^T), gripper action]

The last row of an episode has zero motion and keeps its gripper action. Deltas never cross
episodes. Rotations are axis-angle; training converts them to euler (registry ``AXIS_ANGLE``).
FK needs the ``data`` env: robosuite 1.4.0 + mujoco 3.3.2, the versions the trained data was
made with (bit-exact on every frame).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import dedup, v30
from ..common.download import upstream_dir
from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir
from .lerobot_v2x import V2X_COLUMNS, LeRobotV2x, episode_files

logger = logging.getLogger(__name__)

Fk = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
AxisAngle = Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------------ forward kinematics


def make_fk() -> tuple[Fk, AxisAngle]:
    """``fk(7 joints) -> (grip-site position [3], rotation [3, 3])`` of the robosuite Panda, and
    robosuite's rotation matrix → axis-angle."""
    import mujoco  # type: ignore[import]
    import robosuite  # type: ignore[import]
    import robosuite.utils.transform_utils as T  # type: ignore[import]
    from robosuite.controllers import load_controller_config  # type: ignore[import]

    env = robosuite.make(
        "Lift",
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        ignore_done=True,
    )
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    adr = [model.joint(f"robot0_joint{i}").qposadr[0] for i in range(1, 8)]
    site = model.site("gripper0_grip_site").id

    def fk(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for k, a in enumerate(adr):
            data.qpos[a] = joints[k]
        mujoco.mj_forward(model, data)
        return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()

    def axis_angle(rot: np.ndarray) -> np.ndarray:
        return T.quat2axisangle(T.mat2quat(rot))

    return fk, axis_angle


def delta_ee(
    state: np.ndarray, action: np.ndarray, fk: Fk, axis_angle: AxisAngle
) -> tuple[np.ndarray, np.ndarray]:
    """One episode's ``[N, 8]`` joint state/action → ``([N, 8] EE state, [N, 7] ΔEE action)``."""
    state = state.astype(np.float64)
    n = len(state)
    pos = np.zeros((n, 3))
    rot = []
    for t in range(n):
        pos[t], r = fk(state[t, :7])
        rot.append(r)
    st = np.zeros((n, 8), np.float32)
    st[:, 0:3] = pos
    for t in range(n):
        st[t, 3:6] = axis_angle(rot[t])
    st[:, 7] = state[:, 7]
    ac = np.zeros((n, 7), np.float32)
    for t in range(n - 1):
        ac[t, 0:3] = pos[t + 1] - pos[t]
        ac[t, 3:6] = axis_angle(rot[t + 1] @ rot[t].T)
    ac[:, 6] = action[:, 7].astype(np.float64)
    return st, ac


KEY_COLUMNS: dedup.Columns = ((0, 1, 2, 7), (0, 1, 2, 6))
"""The episode key's columns (``Reader.key_columns``): the EE position and the raw gripper, of
state and action. The rotation is left out: ``T.mat2quat`` runs a float32 ``np.linalg.eigh``,
whose low bits differ across CPUs (OpenBLAS kernels)."""


class DeltaEe:
    """The per-episode ``(joint state, joint action) → (EE state, ΔEE action)`` transform; the
    robosuite model is built on the first call."""

    def __init__(self) -> None:
        self._fk: tuple[Fk, AxisAngle] | None = None

    def __call__(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._fk is None:
            self._fk = make_fk()
        return delta_ee(state, action, *self._fk)


# ------------------------------------------------------------------ berkeley_rpt

RPT = HfSnapshot("IPEC-COMMUNITY/berkeley_rpt_lerobot", "c4b7ed4b4bc05ef9475629ca7cfae1145f95550c")
RPT_SUBSET_TASK = "pick an object from the bin"
RPT_SUBSET_EPISODES = tuple(
    int(i)
    for i in (
        "1 2 30 37 48 53 63 70 78 81 84 86 88 99 100 102 106 108 116 120 125 127 138 145 146 149 "
        "150 151 155 157 172 175 183 190 194 197 202 205 209 227 236 254 256 258 260 267 270 272 "
        "273 275 280 285 287 288 290 294 296 297 302 303 307 309 319 326 327 340 342 344 348 349 "
        "361 364 371 373 380 383 385 387 397 403 409 419 426 430 433 437 439 444 445 455 456 461 "
        "466 469 472 474 477 484 486 489 495 496 497 499 506 508 518 523 528 530 534 536 537 540 "
        "541 544 549 559 564 567 570 581 582 585 590 593 594 608 610 616 617 619 621 628 629 636 "
        "651 663 677 682 693 702 704 706 711 714 715 719 726 729 731 745 749 754 761 769 777 781 "
        "784 786 792 793 796 798 799 803 804 805 809 817 818 821 822 829 835 838 847 849 859 868 "
        "872 879 882 885 891 900 903"
    ).split()
)
"""The upstream episodes of ``RPT_SUBSET_TASK`` (meta order)."""
RPT_SUBSET = HfSnapshot(
    RPT.repo, RPT.revision, episode_files(RPT_SUBSET_EPISODES, ("observation.images.hand_image",))
)


# ------------------------------------------------------------------ RoboSet

ROBOSET = HfSnapshot(
    "jdvakil/RoboSet-Teleoperation", "ab4719b8c5ff5deea405e5ade0be5cb676990535", ("*/*/*/*.h5",)
)
ROBOSET_FPS = 5
ROBOSET_CAMERAS = {
    "rgb_top": "observation.images.top",
    "rgb_wrist": "observation.images.wrist",
    "rgb_left": "observation.images.left",
    "rgb_right": "observation.images.right",
}
_SCENE_RE = re.compile(r"_scene_\d+$")

ROBOSET_TASKS: dict[str, str] = {
    "pick butter": "Pick up the butter.",
    "place butter": "Put down the butter.",
    "pick bowl": "Pick up the bowl.",
    "place bowl": "Put down the bowl.",
    "pick cup": "Pick up the cup.",
    "place cup": "Put down the cup.",
    "pick lid": "Pick up the lid.",
    "place lid": "Put down the lid.",
    "pick mug": "Pick up the mug.",
    "pick tea": "Pick up the tea box.",
    "place tea": "Put down the tea box.",
    "pick toast": "Pick up the toast.",
    "place toast": "Put down the toast.",
    "pick towel": "Pick up the towel.",
    "cap lid": "Put the lid on the container.",
    "cap mug": "Put the cap on the mug.",
    "uncap lid": "Take the lid off the container.",
    "wipe towel": "Wipe the surface with the towel.",
    "slide open drawer": "Slide the drawer open.",
    "slide close drawer": "Slide the drawer closed.",
    "slide in bowl": "Slide the bowl in.",
    "slide out bowl": "Slide the bowl out.",
    "open oven": "Open the oven door.",
    "close oven": "Close the oven door.",
    "flap open oven": "Open the oven door.",
    "flap close oven": "Close the oven door.",
    "plunge toaster": "Push down the toaster lever.",
}
"""Skill phrase of a task-scene dir → the task text."""


ROBOSET_DIRS: dict[str, tuple[int, int]] = {
    "baking_prep/scene_1/baking_prep_pick_butter_scene_1": (25, 250),
    "baking_prep/scene_1/baking_prep_place_butter_scene_1": (25, 250),
    "baking_prep/scene_1/baking_prep_slide_close_drawer_scene_1": (10, 250),
    "baking_prep/scene_1/baking_prep_slide_open_drawer_scene_1": (10, 250),
    "baking_prep/scene_4/baking_prep_pick_butter_scene_4": (25, 250),
    "baking_prep/scene_4/baking_prep_place_butter_scene_4": (25, 250),
    "baking_prep/scene_4/baking_prep_slide_close_drawer_scene_4": (10, 250),
    "baking_prep/scene_4/baking_prep_slide_open_drawer_scene_4": (10, 250),
    "clean_kitchen/scene_3/clean_kitchen_cap_mug_scene_3": (25, 250),
    "clean_kitchen/scene_3/clean_kitchen_flap_close_oven_scene_3": (10, 250),
    "clean_kitchen/scene_3/clean_kitchen_pick_mug_scene_3": (25, 250),
    "clean_kitchen/scene_3/clean_kitchen_pick_towel_scene_3": (25, 250),
    "clean_kitchen/scene_3/clean_kitchen_slide_close_drawer_scene_3": (10, 250),
    "clean_kitchen/scene_3/clean_kitchen_wipe_towel_scene_3": (25, 250),
    "heat_soup/scene_2/baking_close_oven_scene_2": (10, 250),
    "heat_soup/scene_2/baking_open_oven_scene_2": (10, 250),
    "heat_soup/scene_2/baking_pick_bowl_scene_2": (25, 250),
    "heat_soup/scene_2/baking_slide_in_bowl_scene_2": (25, 250),
    "heat_soup/scene_4/baking_open_oven_scene_4": (10, 250),
    "heat_soup/scene_4/baking_pick_bowl_scene_4": (25, 250),
    "heat_soup/scene_4/baking_slide_in_bowl_scene_4": (25, 250),
    "make_tea/scene_2/make_tea_cap_lid_scene_2": (25, 250),
    "make_tea/scene_2/make_tea_pick_lid_scene_2": (25, 250),
    "make_tea/scene_2/make_tea_pick_tea_scene_2": (25, 250),
    "make_tea/scene_2/make_tea_place_lid_scene_2": (25, 250),
    "make_tea/scene_2/make_tea_place_tea_scene_2": (25, 250),
    "make_tea/scene_2/make_tea_uncap_lid_scene_2": (25, 250),
    "make_toast/scene_1/make_toast_pick_toast_scene_1": (25, 250),
    "make_toast/scene_1/make_toast_place_toast_scene_1": (25, 250),
    "make_toast/scene_2/make_toast_plunge_toaster_scene_2": (10, 250),
    "serve_soup/scene_2/serve_soup_flap_close_oven_scene_2": (10, 250),
    "serve_soup/scene_2/serve_soup_flap_open_oven_scene_2": (10, 250),
    "serve_soup/scene_2/serve_soup_place_bowl_scene_2": (25, 250),
    "serve_soup/scene_2/serve_soup_slide_out_bowl_scene_2": (25, 250),
    "stow_bowl/scene_3/stow_bowl_pick_cup_scene_3": (25, 250),
    "stow_bowl/scene_3/stow_bowl_place_cup_scene_3": (25, 250),
    "stow_bowl/scene_3/stow_bowl_slide_close_drawer_scene_3": (10, 250),
    "stow_bowl/scene_3/stow_bowl_slide_open_drawer_scene_3": (10, 250),
}
"""Task-scene dir → (``.h5`` sessions, episodes) at the pin, in upstream (sorted) order. A
25-file dir stores each trial 10× (the 10 trials of a session are identical copies; dedup
drops 9); a 10-file dir holds 25 distinct trials per session."""
ROBOSET_SUBSET_DIRS = (
    "heat_soup/scene_4/baking_open_oven_scene_4",
    "make_tea/scene_2/make_tea_place_tea_scene_2",
)
"""One 10-file dir (distinct trials, trailing-NaN cuts, 1 frozen drop) and one 25-file dir
(10× copies, 50 frozen drops)."""


def roboset_episodes(dirs: Mapping[str, tuple[int, int]] = ROBOSET_DIRS) -> dict[str, range]:
    """Task-scene dir → its global upstream episode indices (sorted dir, sorted ``.h5``, HDF5
    trial order)."""
    out, start = {}, 0
    for d in sorted(dirs):
        out[d] = range(start, start + dirs[d][1])
        start += dirs[d][1]
    return out


def roboset_task(task_dir: str) -> str:
    """``heat_soup/scene_2/baking_close_oven_scene_2`` → "Close the oven door."

    The skill is the leaf name without ``_scene_N`` and the activity prefix (the top dir); a
    leaf whose prefix is another activity word (``baking_*`` under ``heat_soup``) loses it too.
    """
    activity, _, leaf = task_dir.split("/")
    skill = _SCENE_RE.sub("", leaf)
    if skill.startswith(activity + "_"):
        skill = skill[len(activity) + 1 :]
    skill = skill.replace("_", " ").strip()
    words = skill.split(" ", 1)
    if len(words) == 2 and words[1] in ROBOSET_TASKS:
        skill = words[1]
    return ROBOSET_TASKS[skill]


def trial_arrays(
    data: Mapping, with_frames: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """One HDF5 trial → ``(state [N, 8], action [N, 8], frames)``, float16 widened to float32.
    The trailing rows with NaN control (no command at the episode end) are cut; any other NaN
    raises."""
    state = np.concatenate(
        [
            np.asarray(data["qp_arm"], dtype=np.float32),
            np.asarray(data["qp_ee"], dtype=np.float32).reshape(-1, 1),
        ],
        axis=1,
    )
    action = np.concatenate(
        [
            np.asarray(data["ctrl_arm"], dtype=np.float32),
            np.asarray(data["ctrl_ee"], dtype=np.float32).reshape(-1, 1),
        ],
        axis=1,
    )
    valid = np.asarray(~np.isnan(action).any(axis=1))
    n = int(np.argmin(valid)) if not valid.all() else len(valid)
    if valid[n:].any() or n < 2 or np.isnan(state[:n]).any():
        raise ValueError(f"{getattr(data, 'name', '')}: NaN outside the trailing control rows")
    frames = (
        {key: np.asarray(data[raw], dtype=np.uint8)[:n] for raw, key in ROBOSET_CAMERAS.items()}
        if with_frames
        else {}
    )
    return state[:n], action[:n], frames


@dataclass(frozen=True)
class Roboset(Reader):
    """RoboSet teleop HDF5 → one v3.0 leaf per task-scene dir (``ROBOSET_DIRS``), FK to ΔEE,
    frames encoded. A leaf is built iff the pin selects its episodes; a selection must cover
    whole dirs. ``--shard i/n`` builds dirs ``[i::n]``."""

    def dirs(self, pin: Pin) -> list[str]:
        """The task-scene dirs the pin selects."""
        if pin.select is None:
            return sorted(ROBOSET_DIRS)
        select = set(pin.select)
        out = []
        for d, eps in roboset_episodes().items():
            hit = select.intersection(eps)
            if hit and len(hit) != len(eps):
                raise ValueError(f"{d}: the pin selects {len(hit)} of its {len(eps)} episodes")
            if hit:
                out.append(d)
        return out

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        dirs = self.dirs(pin)
        return [collection_dir(name, root) / d.rsplit("/", 1)[-1] for d in dirs]

    def files(self, root: Path, task_dir: str) -> list[Path]:
        """The ``.h5`` sessions of a dir, sorted; a count off the pin raises."""
        files = sorted((upstream_dir(root, ROBOSET) / task_dir).glob("*.h5"))
        if len(files) != ROBOSET_DIRS[task_dir][0]:
            raise ValueError(
                f"{task_dir}: {len(files)} .h5 files, pinned {ROBOSET_DIRS[task_dir][0]}"
            )
        return files

    def episodes(
        self, root: Path, task_dir: str, transform: DeltaEe | None = None
    ) -> Iterator[v30.Episode]:
        """The episodes of a dir in upstream order; with ``transform``, ΔEE arrays and frames."""
        import h5py  # type: ignore[import]

        task = roboset_task(task_dir)
        index = 0
        for path in self.files(root, task_dir):
            with h5py.File(path, "r") as f:
                for trial in f:
                    state, action, frames = trial_arrays(f[trial]["data"], transform is not None)
                    if transform is not None:
                        state, action = transform(state, action)
                    yield v30.Episode(index, state, action, task, frames=frames)
                    index += 1
        if index != ROBOSET_DIRS[task_dir][1]:
            raise ValueError(f"{task_dir}: {index} episodes, pinned {ROBOSET_DIRS[task_dir][1]}")

    def key_columns(self) -> dedup.Columns:
        return KEY_COLUMNS

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        i, n = shard
        transform = DeltaEe()
        dirs = self.dirs(pin)
        for task_dir, out in list(zip(dirs, self.leaves(name, root, pin)))[i::n]:
            v30.write_leaf(self.episodes(root, task_dir, transform), out, ROBOSET_FPS, "franka")
            logger.info("%s: built %s", name, out.name)


# ------------------------------------------------------------------ registration

PINS = {
    "berkeley_rpt_delta": Pin(
        (RPT,),
        episodes=908,
        frames=392578,
        subset=Pin((RPT_SUBSET,), 187, 54214, select=RPT_SUBSET_EPISODES),
    ),
    "roboset_delta": Pin(
        (ROBOSET,),
        episodes=9500,
        frames=379965,
        subset=Pin(
            tuple(
                HfSnapshot(ROBOSET.repo, ROBOSET.revision, (f"{d}/*.h5",))
                for d in ROBOSET_SUBSET_DIRS
            ),
            episodes=500,
            frames=19974,
            select=tuple(i for d in ROBOSET_SUBSET_DIRS for i in roboset_episodes()[d]),
        ),
    ),
}

RECIPES = {
    "berkeley_rpt_delta": Recipe(
        LeRobotV2x("berkeley_rpt_delta_lerobot", V2X_COLUMNS, transform=DeltaEe(), key=KEY_COLUMNS),
        curate=("short_video",),
        stats=(op("kept_qk"), op("euler")),
    ),
    "roboset_delta": Recipe(
        Roboset(),
        curate=("frozen",),
        stats=(op("kept_qk"), op("euler", pooled=True)),
    ),
}
