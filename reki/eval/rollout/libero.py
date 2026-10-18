"""LIBERO and LIBERO-Plus: the env runs in-process (lerobot's ``LiberoEnv``, MuJoCo on OSMesa).

Env vars (``MUJOCO_GL=osmesa``, ``LIBERO_CONFIG_PATH``, ...) must be set by the caller:
``launchers/eval_protocols.py`` builds the command, ``scripts/eval/SETUP.md`` the install.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from reki.data.robotics.registry import recenters_state_roll
from reki.data.robotics.transforms import recenter_euler_roll
from reki.eval.protocols import LIBERO_SUITES, LIBERO_TASK_IDS
from reki.eval.rollout.loop import Adapter, Unit
from reki.inference.features import EnvObs, decode_action
from reki.policy.train_config import TrainConfig

ENV_FPS = 10
LIBERO_EULER_DIMS = (3, 6)
"""The rotation block of the LIBERO action (euler xyz when ``unify_ee_euler_rotation``)."""


def axisangle_to_euler_np(aa: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector → euler 'xyz' (radians)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(aa, dtype=np.float64)).as_euler("xyz").astype(np.float32)


def decode_libero_action(
    action_norm: torch.Tensor,
    stats: dict[str, torch.Tensor],
    *,
    unify_ee_euler_rotation: bool,
) -> np.ndarray:
    """LIBERO decode. Gripper +1 is open for the model and close for robosuite OSC_POSE."""
    return decode_action(
        action_norm,
        stats,
        dataset="libero",
        euler_rot_dims=LIBERO_EULER_DIMS if unify_ee_euler_rotation else None,
    )


def quat_xyzw_to_axisangle(q: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) to axis-angle, as lerobot's ``LiberoProcessorStep._quat2axisangle``.
    q and -q are the same rotation. scipy returns the opposite sign to the dataset."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = q[:3] / den
    return (axis * angle).astype(np.float32)


def task_ids_for_axis(suite_name: str, axis: str) -> list[int]:
    """The task ids of a LIBERO-Plus suite in one perturbation category (``axis``, e.g.
    "Camera Viewpoints"), from the suite's ``benchmark/task_classification.json``."""
    from lerobot.envs.libero import _get_suite
    from libero.libero import get_libero_path  # type: ignore[import-not-found]

    tc_path = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    tc = json.loads(tc_path.read_text())
    if suite_name not in tc:
        raise ValueError(f"suite {suite_name!r} is not in {tc_path} (have {sorted(tc)})")
    matching_names = {e["name"] for e in tc[suite_name] if e["category"] == axis}
    if not matching_names:
        raise ValueError(
            f"no tasks for axis {axis!r} in suite {suite_name!r} "
            f"(categories: {sorted({e['category'] for e in tc[suite_name]})})"
        )
    suite = _get_suite(suite_name)
    return [i for i, t in enumerate(suite.tasks) if t.name in matching_names]


def _patch_libero_init_states_loader() -> None:
    """Load init states through the suite's ``get_task_init_states``: LIBERO-Plus maps its
    perturbed tasks to the 10 base init files, which lerobot's loader does not. Also defaults
    ``torch.load`` to ``weights_only=False`` during the call (LIBERO-Plus on torch >= 2.6)."""
    import lerobot.envs.libero as le

    if getattr(le, "_init_states_patched", False):
        return
    orig = le.get_task_init_states
    orig_torch_load = torch.load

    def torch_load_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    def patched(task_suite: Any, i: int) -> Any:
        if hasattr(task_suite, "get_task_init_states"):
            torch.load = torch_load_compat
            try:
                return task_suite.get_task_init_states(i)
            finally:
                torch.load = orig_torch_load
        return orig(task_suite, i)

    le.get_task_init_states = patched
    le._init_states_patched = True  # type: ignore[attr-defined]


def make_libero_env(task: str, task_id: int = 0, n_steps_wait: int = 10) -> Any:
    """One single-task LIBERO env (no vector wrapper)."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    _patch_libero_init_states_loader()
    return LiberoEnv(
        task_suite=_get_suite(task),
        task_id=task_id,
        task_suite_name=task,
        camera_name=("agentview_image", "robot0_eye_in_hand_image"),
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        n_envs=1,
        num_steps_wait=n_steps_wait,
        control_mode="relative",
    )


def libero_obs_to_envobs(
    env: Any,
    raw_obs: dict[str, Any],
    convert_rot_to_euler: bool = False,
    recenter_roll: bool = False,
) -> EnvObs:
    """The LIBERO obs in the training format (lerobot ``LiberoProcessorStep``): images rotated
    180°, state ``[eef_pos(3), axis-angle(3), 0, gripper_qpos[0]]``. ``convert_rot_to_euler``
    converts the rotation to euler xyz (for ``unify_ee_euler_rotation`` checkpoints).
    ``recenter_roll`` then shifts the roll by π (``DatasetSpec.recenter_state_roll``)."""
    pixels = raw_obs["pixels"]
    img_a = np.flip(np.asarray(pixels["image"]), axis=(0, 1)).copy()
    img_b = np.flip(np.asarray(pixels["image2"]), axis=(0, 1)).copy()
    rs = raw_obs["robot_state"]
    pos = np.asarray(rs["eef"]["pos"], dtype=np.float32)
    rot = quat_xyzw_to_axisangle(np.asarray(rs["eef"]["quat"], dtype=np.float32)).astype(np.float32)
    if convert_rot_to_euler:
        rot = axisangle_to_euler_np(rot)
        if recenter_roll:
            recenter_euler_roll(rot, [(0, 3)])
    gripper_qpos = np.asarray(rs["gripper"]["qpos"], dtype=np.float32)
    state = np.concatenate([pos, rot, np.zeros(1, dtype=np.float32), gripper_qpos[:1]])
    return EnvObs(
        images=[Image.fromarray(img_a), Image.fromarray(img_b)],
        state_raw=state.astype(np.float32),
        task_text=env.task_description,
    )


class LiberoEnv:
    """A LIBERO environment adapted to the shared rollout interface."""

    def __init__(self, suite: str, task_id: int, rot_to_euler: bool, recenter_roll: bool) -> None:
        self.env = make_libero_env(suite, task_id=task_id)
        self.max_episode_steps = int(self.env._max_episode_steps)
        self.rot_to_euler = rot_to_euler
        self.recenter_roll = recenter_roll
        self.scene: Any = None

    def reset(self, seed: int, episode: int = 0) -> dict:
        self.env.init_state_id = episode
        raw, _ = self.env.reset(seed=seed)
        return raw

    def step(self, action: np.ndarray, *, need_obs: bool, sim_state: bool) -> tuple:
        raw, _, terminated, _, info = self.env.step(action)
        return raw, bool(info.get("is_success", False)), bool(terminated), False

    def observe(self, raw: dict) -> EnvObs:
        return libero_obs_to_envobs(
            self.env, raw, convert_rot_to_euler=self.rot_to_euler, recenter_roll=self.recenter_roll
        )

    def frame(self, raw: dict) -> np.ndarray:
        """Agentview and wrist side by side, upright."""
        a = np.flip(np.asarray(raw["pixels"]["image"]), axis=(0, 1))
        b = np.flip(np.asarray(raw["pixels"]["image2"]), axis=(0, 1))
        return np.concatenate([a, b], axis=1).astype(np.uint8)

    def traj_start(self, raw: dict) -> dict:
        from reki.eval.mujoco_sim_state import SceneIndex

        self.scene = SceneIndex.build(self.env._env.env.sim.model._model)
        return {"articulation_joints": self.scene.art_joints}

    def traj_step(self, raw: dict) -> dict:
        from reki.eval.mujoco_sim_state import snapshot

        rs_env = self.env._env.env
        sim = rs_env.sim
        return {
            "sim_state": snapshot(sim.model._model, sim.data._data, self.scene, rs_env.timestep)
        }


class LiberoAdapter(Adapter):
    """LIBERO suites x task ids; LIBERO-Plus with ``perturbation_axis`` (the task ids of that
    axis) and ``suite_alias`` (the results dir of the suite)."""

    benchmark = "libero"
    n_cameras = 2

    def __init__(
        self,
        suites: list[str] | None,
        task_ids: list[int] | None,
        perturbation_axis: str | None = None,
        suite_alias: str | None = None,
    ) -> None:
        self.suites = suites or list(LIBERO_SUITES)
        self.task_ids = task_ids or list(LIBERO_TASK_IDS)
        self.perturbation_axis = perturbation_axis
        self.suite_alias = suite_alias
        self.rot_to_euler = False
        self.recenter_roll = False

    def configure(self, train_cfg: TrainConfig) -> None:
        self.rot_to_euler = train_cfg.unify_ee_euler_rotation
        self.recenter_roll = recenters_state_roll("libero", train_cfg)

    def prompt_fps(self, stats_fps: float) -> float:
        return stats_fps

    def env_fps(self, stats_fps: float) -> int:
        return ENV_FPS

    def units(self) -> list[Unit]:
        units = []
        for suite in self.suites:
            tids = self.task_ids
            if self.perturbation_axis is not None:
                tids = task_ids_for_axis(suite, self.perturbation_axis)
                print(
                    f"  perturbation_axis={self.perturbation_axis!r}: {len(tids)} tasks in {suite}"
                )
            for tid in tids:
                rel = f"{self.suite_alias or suite}/{tid:02d}"
                units.append(Unit(f"{suite}:{tid}", rel, suite, {"task": suite, "task_id": tid}))
        return units

    def session(self) -> AbstractContextManager:
        return nullcontext()

    def make_env(self, unit: Unit) -> LiberoEnv:
        return LiberoEnv(unit.task, unit.fields["task_id"], self.rot_to_euler, self.recenter_roll)

    def decode(self, action_norm: torch.Tensor, stats: dict) -> np.ndarray:
        return decode_libero_action(action_norm, stats, unify_ee_euler_rotation=self.rot_to_euler)
