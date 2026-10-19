"""LeRobot env type ``reki``: the simulators of ``scripts/eval/rollout.py`` under ``lerobot-eval``.

    lerobot-eval --policy.path=kyutai/reki-2b-robocasa365 --env.type=reki \\
        --env.benchmark=robocasa365 --env.task=CloseFridge --env.sim_port=5798 \\
        --eval.batch_size=1 --eval.n_episodes=50

One env runs the units and episodes of the rollout: the same task grid, episode seeds, LIBERO
init states, step cap and sim servers (``reki.eval.rollout``). Each reset seeds torch with the
episode seed, as the rollout does, so the policy draws the same flow noise. The env emits
frames in the format of the training dataset; ``reki.RekiPolicy`` converts them as for a
dataset frame. LIBERO and LIBERO-Plus run in-process; the other benchmarks drive a running sim
server (``--env.sim_host``/``--env.sim_port``).
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

GYM_ID = "reki/Sim-v0"


@dataclass(frozen=True)
class SimFrame:
    """The training-dataset frame that the env of a benchmark emits."""

    dataset: str
    images: dict[str, tuple[int, int]]  # dataset camera name → (height, width)
    state_dim: int  # the dataset's full state; the policy slices it
    action_dim: int
    fps: int


SIMS: dict[str, SimFrame] = {
    "libero": SimFrame("libero", dict.fromkeys(("image", "image2"), (256, 256)), 8, 7, 10),
    "robotwin": SimFrame(
        "robotwin2_clean",
        dict.fromkeys(("cam_high", "cam_left_wrist", "cam_right_wrist"), (480, 640)),
        14,
        14,
        15,
    ),
    "robocasa365": SimFrame(
        "robocasa365",
        dict.fromkeys(
            ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"), (256, 256)
        ),
        16,
        12,
        20,
    ),
}


@EnvConfig.register_subclass("reki")
@dataclass
class RekiEnv(EnvConfig):
    # libero (also LIBERO-Plus), robotwin or robocasa365
    benchmark: str = "libero"
    # rollout.py --task: LIBERO suites or sim-server tasks, comma-separated. Default: the full
    # grid (RoboTwin: the tasks of the --policy.path checkpoint)
    task: str | None = None
    # rollout.py --task_id, --perturbation_axis, --n_episodes (per task) and --seed
    task_ids: list[int] | None = None
    perturbation_axis: str | None = None
    n_episodes: int | None = None
    seed: int = 42
    sim_host: str = "127.0.0.1"
    sim_port: int = 5798
    # rollout.py --robotwin_seed_cache (default: $REKI_ROBOTWIN_SEED_CACHE, else auto)
    robotwin_seed_cache: str | None = None
    # the policy's actions per predict (default: the --policy.path checkpoint's): rollout.py ends
    # an episode at the step cap only at the end of a chunk
    n_action_steps: int | None = None

    def __post_init__(self) -> None:
        from lerobot.configs import parser

        # fail at the CLI parse: without it, lerobot-eval first builds 50 envs
        if parser.parse_arg("env.type") == "reki" and parser.parse_arg("eval.batch_size") != "1":
            raise ValueError("--env.type=reki runs one episode at a time: pass --eval.batch_size=1")
        frame = SIMS[self.benchmark]
        self.fps = frame.fps
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(frame.action_dim,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(frame.state_dim,)),
        }
        self.features_map = {ACTION: ACTION, "agent_pos": OBS_STATE}
        for cam, (h, w) in frame.images.items():
            self.features[f"pixels/{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(h, w, 3))
            self.features_map[f"pixels/{cam}"] = f"{OBS_IMAGES}.{cam}"

    @property
    def package_name(self) -> str:
        return __name__

    @property
    def gym_id(self) -> str:
        return GYM_ID

    @property
    def gym_kwargs(self) -> dict:
        from lerobot.configs import parser
        from lerobot.configs.policies import PreTrainedConfig

        policy_path = parser.get_path_arg("policy") or ""
        n_action_steps = self.n_action_steps
        if n_action_steps is None and policy_path:
            n_action_steps = PreTrainedConfig.from_pretrained(policy_path).n_action_steps
        return {
            "benchmark": self.benchmark,
            "tasks": [t for t in self.task.split(",") if t] if self.task else None,
            "task_ids": self.task_ids,
            "perturbation_axis": self.perturbation_axis,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "sim_host": self.sim_host,
            "sim_port": self.sim_port,
            "robotwin_seed_cache": self.robotwin_seed_cache
            or os.environ.get("REKI_ROBOTWIN_SEED_CACHE")
            or "auto",
            "n_action_steps": n_action_steps or 1,
            "policy_path": policy_path,
        }


def _adapter(
    benchmark: str,
    tasks: list[str] | None,
    task_ids: list[int] | None,
    perturbation_axis: str | None,
    sim_host: str,
    sim_port: int,
    robotwin_seed_cache: str,
    policy_path: str,
) -> Any:
    """The rollout adapter of ``benchmark`` (``scripts/eval/rollout.py``'s ``make_adapter``)."""
    from reki.eval.rollout import remote

    if benchmark == "libero":
        from reki.eval.rollout.libero import LiberoAdapter

        return LiberoAdapter(tasks, task_ids, perturbation_axis)
    if benchmark == "robotwin":
        seed_cache, required = remote.seed_cache_root(robotwin_seed_cache)
        return remote.RoboTwinAdapter(
            tasks,
            policy_path,
            sim_host,
            sim_port,
            seed_cache=seed_cache,
            seed_cache_required=required,
        )
    return remote.RoboCasa365Adapter(tasks, policy_path, sim_host, sim_port)


def libero_dataset_frame(raw: dict[str, Any]) -> dict[str, Any]:
    """A LeRobot ``LiberoEnv`` observation as a frame of the LIBERO training dataset: images
    rotated 180°, state ``[eef_pos, axis-angle, gripper_qpos(2)]`` (``LiberoProcessorStep``)."""
    from reki.eval.rollout.libero import quat_xyzw_to_axisangle

    rs = raw["robot_state"]
    state = np.concatenate(
        [
            np.asarray(rs["eef"]["pos"], dtype=np.float32),
            quat_xyzw_to_axisangle(np.asarray(rs["eef"]["quat"], dtype=np.float32)),
            np.asarray(rs["gripper"]["qpos"], dtype=np.float32),
        ]
    )
    pixels = {k: np.flip(np.asarray(v), axis=(0, 1)).copy() for k, v in raw["pixels"].items()}
    return {"pixels": pixels, "agent_pos": state}


def server_dataset_frame(raw: dict[str, Any], frame: SimFrame) -> dict[str, Any]:
    """A sim-server observation (``reki.eval.sim_protocol``) as a frame of the training dataset:
    the sliced server state goes back to its slots of the dataset state, zeros elsewhere."""
    from reki.data.robotics.registry import DATASETS

    sliced = np.asarray(raw["state"], dtype=np.float32)
    state = np.zeros(frame.state_dim, dtype=np.float32)
    sl = DATASETS[frame.dataset].state_slice
    for j, i in enumerate(sl if sl is not None else range(frame.state_dim)):
        if i is not None:
            state[i] = sliced[j]
    pixels = {k: np.asarray(raw["images"][k], dtype=np.uint8) for k in frame.images}
    return {"pixels": pixels, "agent_pos": state}


class RekiSimEnv(gym.Env):
    """All units of one rollout as one gym env: each seeded reset starts the next episode."""

    def __init__(
        self,
        benchmark: str,
        tasks: list[str] | None = None,
        task_ids: list[int] | None = None,
        perturbation_axis: str | None = None,
        n_episodes: int | None = None,
        seed: int = 42,
        sim_host: str = "127.0.0.1",
        sim_port: int = 5798,
        robotwin_seed_cache: str = "auto",
        n_action_steps: int = 1,
        policy_path: str = "",
    ) -> None:
        from reki.eval.protocols import N_EPISODES

        self.benchmark = benchmark
        self.frame = SIMS[benchmark]
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": self.frame.fps}
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        k: spaces.Box(0, 255, (h, w, 3), np.uint8)
                        for k, (h, w) in self.frame.images.items()
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, (self.frame.state_dim,), np.float32),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, (self.frame.action_dim,), np.float32)
        self.seed_base = seed
        self.n_action_steps = n_action_steps
        self.adapter = _adapter(
            benchmark,
            tasks,
            task_ids,
            perturbation_axis,
            sim_host,
            sim_port,
            robotwin_seed_cache,
            policy_path,
        )
        units = self.adapter.units()
        n = n_episodes or N_EPISODES["libero_plus" if perturbation_axis else benchmark]
        self._episodes = iter([(u, ep) for u in units for ep in range(n)])
        print(
            f"[reki env] {len(units)} tasks x {n} episodes: run with "
            f"--eval.n_episodes={len(units) * n} for the full grid"
        )
        self._session = ExitStack()
        self._session.enter_context(self.adapter.session())
        self._unit: Any = None
        self._env: Any = None
        self._rng = np.random.default_rng()
        self._obs: dict[str, Any] = {}
        self._autoreset = False
        self.task = ""
        self.task_description = ""
        self._max_episode_steps = 0

    def _observe(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self.benchmark == "libero":
            self.task_description = self._env.env.task_description
            return libero_dataset_frame(raw)
        self.task_description = raw["instruction"]
        return server_dataset_frame(raw, self.frame)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        # The vector env resets right after a terminal step; lerobot-eval then resets again to
        # start the next episode. Answer the first with the final observation: the rollout
        # resets once per episode (a RoboTwin reset can search seeds for minutes).
        if self._autoreset:
            self._autoreset = False
            return self._obs, {}
        unit, ep = next(self._episodes, (None, None))
        if unit is None:
            raise RuntimeError("every episode of the grid ran: lower --eval.n_episodes")
        if unit is not self._unit:
            self._close_unit_env()
            self._unit = unit
            self._env = self.adapter.make_env(unit)
            cap = self._env.max_episode_steps
            self._max_episode_steps = -(-cap // self.n_action_steps) * self.n_action_steps
            self._rng = np.random.default_rng(self.seed_base)
            self.task = unit.key
            print(f"[reki env] {unit.key} (cap={self._max_episode_steps})")
        ep_seed = int(self._rng.integers(0, 2**31 - 1))
        torch.manual_seed(ep_seed)
        self._obs = self._observe(self._env.reset(ep_seed, episode=ep))
        return self._obs, {"seed": ep_seed}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        raw, success, terminated, truncated = self._env.step(
            np.asarray(action, dtype=np.float32), need_obs=True, sim_state=False
        )
        self._obs = self._observe(raw)
        terminated = success or terminated
        self._autoreset = terminated or truncated
        return self._obs, float(success), terminated, truncated, {"is_success": success}

    def render(self) -> np.ndarray:
        return np.concatenate([self._obs["pixels"][k] for k in self.frame.images], axis=1)

    def _close_unit_env(self) -> None:
        if self.benchmark == "libero" and self._env is not None:
            self._env.env.close()

    def close(self) -> None:
        self._close_unit_env()
        self._session.close()


gym.register(
    id=GYM_ID, entry_point=f"{__name__}:RekiSimEnv", order_enforce=False, disable_env_checker=True
)
