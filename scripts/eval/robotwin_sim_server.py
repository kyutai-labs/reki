"""RoboTwin 2.0 (dual-arm ALOHA) sim server.

Runs inside the isolated `robotwin` micromamba env (SAPIEN 3.0.0b1 + CuRobo + pytorch3d; see
scripts/eval/SETUP.md). Holds one RoboTwin task env and serves make/reset/step/close over a
TCP socket (``reki/eval/sim_protocol.py``); scripts/eval/rollout.py drives it with the policy
in its own env. Needs a GPU for the SAPIEN Vulkan render and for CuRobo motion planning (the
expert-feasibility check at reset).

Follows the lifecycle of RoboTwin's script/eval_policy.py, without the policy:
  - make: build the task env from its task_config, embodiment and cameras.
  - reset: search for the first seed the expert solves (at most 90 per request; the client
    continues), then set it up for the rollout. A trusted reset (trust=True) skips the search
    for a seed that already passed and reuses its recorded instruction.
  - step: apply the 14-dim qpos action. need_obs=False skips the render (the largest per-step
    cost).

Protocol (each request gets one reply):
  {"cmd":"make","task":<task>}            -> {"max_episode_steps": int,
                                              "instruction_type": "unseen", "code": {code_md5,
                                              task_config, task_config_md5, asset_list_md5},
                                              "trusted_reset": bool}
  {"cmd":"reset","seed":int,              -> {"instruction":str, "images":{cam_high,
   "trust":bool (default False),              cam_left_wrist,cam_right_wrist}(uint8),
   "instruction":str (required with trust)}   "state":list[14], "seed_used":int,
                                              "sim_state":{"actors":{key:{"pose","name"}}}}
  {"cmd":"step","action":[14],"need_obs"} -> {"success","terminated","truncated"}
                                             (+images,state,instruction if need_obs)
  {"cmd":"close"}                         -> {"ok":True}
A step request without "need_obs" renders. A request that raises gets {"error": str,
"traceback": str} and the server keeps serving.

"code" identifies the scene code: code_md5 hashes every .py/.yml/.yaml/.json under envs/,
task_config/ and description/ (git-tracked or untracked, not ignored), asset_list_md5 the
list of files under assets/objects. Actor poses are [x, y, z, qw, qx, qy, qz]; a name that
occurs more than once in the scene is keyed "<name>#<per_scene_id>".

obs: RoboTwin head/left/right cameras -> canonical cam_high/cam_left_wrist/cam_right_wrist;
state = joint_action.vector = left(6 arm + 1 grip) + right(6 arm + 1 grip) = 14-dim.

Parity with the released robotwin2_clean checkpoints: the task_config yml must set
head_camera_type / wrist_camera_type to Large_D435 (640x480, the collection resolution;
upstream defaults to D435 320x240). The server draws the instruction from the unseen split,
as the RoboTwin leaderboard does.

Keep --host on loopback: unauthenticated pickle over a socket permits remote code execution.
Other nodes reach the server through an SSH tunnel, or the opt-in relay of docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import dis
import functools
import hashlib
import importlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(
    str(Path(__file__).resolve().parents[2] / "reki" / "eval")
)  # sim_protocol etc.: no reki deps
import sim_protocol  # type: ignore[import-not-found]  # noqa: E402

# Attributes that check_success reads and only play_once sets at the RoboTwin pin. Upstream
# e154702 (scripts/eval/robotwin_arm_tag_fix.patch) sets them in load_actors.
_PLAY_ONCE_ATTRS = {
    "open_laptop": ("arm_tag",),
    "place_object_scale": ("arm_tag",),
    "put_object_cabinet": ("arm_tag", "origin_z"),
}
# A slow expert run takes tens of seconds: 90 seeds fit in the client's reset timeout.
_MAX_REJECTED_SEEDS = 90
_CODE_SUFFIXES = (".py", ".yml", ".yaml", ".json")

_CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


@functools.lru_cache(maxsize=None)
def _asset_list_md5(repo: str) -> str:
    """md5 of `cd <repo> && find assets/objects -type f -printf '%P\\n' | LC_ALL=C sort`."""
    root = os.path.realpath(os.path.join(repo, "assets", "objects"))
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            if os.path.isfile(full) and not os.path.islink(full):
                paths.append(os.fsencode(os.path.relpath(full, root)))
    return hashlib.md5(b"".join(p + b"\n" for p in sorted(paths))).hexdigest()


def _pose7(pose: Any) -> list:
    return [*np.asarray(pose.p, np.float64).tolist(), *np.asarray(pose.q, np.float64).tolist()]


class RoboTwinSim:
    """Single-task RoboTwin env wrapper mirroring eval_policy.py (no policy).

    RoboTwin resolves ./task_config, ./assets and ./description relative to the cwd, so the
    constructor chdirs into the repo."""

    def __init__(self, repo: str, task_config: str) -> None:
        self.repo = os.path.abspath(repo)
        self.task_config = task_config
        os.chdir(self.repo)
        for p in (
            self.repo,
            os.path.join(self.repo, "policy"),
            os.path.join(self.repo, "description", "utils"),
        ):
            if p not in sys.path:
                sys.path.insert(0, p)
        from envs import CONFIGS_PATH  # type: ignore[import-not-found]  # noqa: E402
        from envs.utils.create_actor import (  # type: ignore[import-not-found]  # noqa: E402
            UnStableError,
        )
        from generate_episode_instructions import (  # type: ignore[import-not-found]  # noqa: E402
            generate_episode_descriptions,
        )

        self._CONFIGS_PATH = CONFIGS_PATH
        self._UnStableError = UnStableError
        self._gen_descr = generate_episode_descriptions
        self.env: Any = None
        self.args: dict = {}
        self.instruction = ""
        self.episode_info: Any = None
        # The scene of the last reset, closed by the next reset or make.
        self._scene_open = False
        self._n_closed = 0

    def _build_args(self, task_name: str) -> dict:
        """Replicates eval_policy.main()'s merge of task yml + embodiment + camera config."""
        cfg_path = os.path.join("task_config", f"{self.task_config}.yml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"task config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        args["task_name"] = task_name
        args["task_config"] = self.task_config
        args["ckpt_setting"] = "eval"
        args["policy_name"] = "remote"
        args["eval_mode"] = True
        args["eval_video_log"] = False

        embodiment_type = args.get("embodiment")
        if embodiment_type is None:
            raise KeyError(f"'embodiment' missing in {cfg_path}")

        with open(
            os.path.join(self._CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8"
        ) as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def robot_file(emb: str) -> str:
            rf = _embodiment_types[emb]["file_path"]
            if rf is None:
                raise ValueError(f"no embodiment file_path for {emb!r}")
            return rf

        def emb_config(robot_dir: str) -> dict:
            with open(os.path.join(robot_dir, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(
            os.path.join(self._CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8"
        ) as f:
            _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[head_camera_type]["h"]
        args["head_camera_w"] = _camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = robot_file(embodiment_type[0])
            args["right_robot_file"] = robot_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = robot_file(embodiment_type[0])
            args["right_robot_file"] = robot_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        args["left_embodiment_config"] = emb_config(args["left_robot_file"])
        args["right_embodiment_config"] = emb_config(args["right_robot_file"])
        return args

    def _step_lim(self, task_name: str) -> int:
        """Eval step limit from task_config/_eval_step_limit.yml.

        The env reads the same file under eval_mode; a task absent from it gets RoboTwin's
        1000 with a printed notice.
        """
        path = os.path.join("task_config", "_eval_step_limit.yml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if task_name not in data:
            print(f"[robotwin] {task_name} not in {path}; step_lim=1000", flush=True)
            return 1000
        return int(data[task_name])

    def make(self, task: str) -> dict:
        envs_module = importlib.import_module(f"envs.{task}")
        if not hasattr(envs_module, task):
            raise AttributeError(f"envs.{task} has no class {task!r}")
        if self._scene_open:
            self._close_quiet()
        self.task_name = task
        self.args = self._build_args(task)
        self.env = getattr(envs_module, task)()
        self.trusted_reset = self._trusted_reset_ok(envs_module, task)
        return {
            "max_episode_steps": self._step_lim(task),
            "instruction_type": "unseen",
            "code": self._code_fingerprint(),
            "trusted_reset": self.trusted_reset,
        }

    @staticmethod
    def _trusted_reset_ok(envs_module: Any, task: str) -> bool:
        """A trusted reset skips play_once: the task's check_success must not read state that
        only play_once sets."""
        if task not in _PLAY_ONCE_ATTRS:
            return True
        load_actors = getattr(envs_module, task).load_actors
        stored = {i.argval for i in dis.get_instructions(load_actors) if i.opname == "STORE_ATTR"}
        return set(_PLAY_ONCE_ATTRS[task]) <= stored

    def _code_fingerprint(self) -> dict:
        """The scene code currently on disk (task code is imported per make). A field that
        cannot be read carries its error string."""

        def field(fn: Any) -> str:
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                return f"error: {type(e).__name__}: {e}"

        def cfg_md5() -> str:
            with open(os.path.join("task_config", f"{self.task_config}.yml"), "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

        return {
            "code_md5": field(self._code_md5),
            "task_config": self.task_config,
            "task_config_md5": field(cfg_md5),
            "asset_list_md5": field(lambda: _asset_list_md5(self.repo)),
        }

    def _code_md5(self) -> str:
        git = "git -c core.quotepath=off ls-files -z --cached --others --exclude-standard --"
        listed = subprocess.run(
            [*git.split(), "envs", "task_config", "description"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout.split(b"\0")
        suffixes = tuple(s.encode() for s in _CODE_SUFFIXES)
        h = hashlib.md5()
        for rel in sorted({p for p in listed if p.endswith(suffixes)}):
            full = os.path.join(self.repo, os.fsdecode(rel))
            if not os.path.isfile(full):
                continue
            with open(full, "rb") as f:
                h.update(rel + b"\0" + hashlib.md5(f.read()).hexdigest().encode() + b"\n")
        return h.hexdigest()

    def _close_env(self) -> None:
        """close_env, freeing the render cache every ``clear_cache_freq`` closes, expert-check
        closes included. Without it every texture and mesh a scene loads stays on the GPU: a
        hard or clutter server fills it within hundreds of episodes ("cannot create buffer")."""
        self._n_closed += 1
        try:
            self.env.close_env(
                clear_cache=self._n_closed % self.args.get("clear_cache_freq", 1) == 0
            )
        finally:
            self._scene_open = False

    def _close_quiet(self) -> None:
        try:
            self._close_env()
        except Exception:
            pass

    def _reject(self, start_seed: int, rejected: int) -> int:
        rejected += 1
        if rejected >= _MAX_REJECTED_SEEDS:
            raise RuntimeError(
                f"seed search gave up: {rejected} seeds rejected for task "
                f"{self.task_name!r} starting at seed {start_seed}"
            )
        return rejected

    def _reset_trusted(self, seed: int, instruction: str | None) -> dict:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"reset(trust=True) needs the instruction recorded for seed {seed}")
        if not self.trusted_reset:
            raise RuntimeError(
                f"{self.task_name!r} sets {_PLAY_ONCE_ATTRS[self.task_name]} only in play_once: "
                "apply scripts/eval/robotwin_arm_tag_fix.patch to the RoboTwin tree"
            )
        if self._scene_open:
            self._close_quiet()
        self.args["render_freq"] = 0
        try:
            self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
        except self._UnStableError as e:
            self._close_quiet()
            raise RuntimeError(f"trusted seed {seed} is unstable for {self.task_name!r}") from e
        self._scene_open = True
        self.episode_info = None
        self.instruction = instruction
        self.env.set_instruction(instruction=instruction)
        return self._reset_payload(seed)

    def reset(self, seed: int, trust: bool = False, instruction: str | None = None) -> dict:
        if self.env is None:
            raise RuntimeError("reset before make")
        if trust:
            return self._reset_trusted(int(seed), instruction)
        if self._scene_open:
            self._close_quiet()
        start_seed = now_seed = int(seed)
        rejected = 0
        while True:
            while True:
                self.args["render_freq"] = 0
                try:
                    self.env.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **self.args)
                    self.episode_info = self.env.play_once()
                    self._close_env()
                except self._UnStableError as e:
                    print(f"[robotwin] seed {now_seed} unstable: {e}", flush=True)
                    self._close_quiet()
                    rejected = self._reject(start_seed, rejected)
                    now_seed += 1
                    continue
                except Exception as e:  # noqa: BLE001 — RoboTwin's intentional seed-skip
                    print(f"[robotwin] seed {now_seed} expert-check failed: {e}", flush=True)
                    self._close_quiet()
                    rejected = self._reject(start_seed, rejected)
                    now_seed += 1
                    continue
                if self.env.plan_success and self.env.check_success():
                    break
                rejected = self._reject(start_seed, rejected)
                now_seed += 1
            try:
                self.env.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **self.args)
            except self._UnStableError as e:
                print(
                    f"[robotwin] real-setup seed {now_seed} unstable: {e}; re-searching",
                    flush=True,
                )
                self._close_quiet()
                rejected = self._reject(start_seed, rejected)
                now_seed += 1
                continue
            self._scene_open = True
            break

        # The instruction pick draws from np.random and a trusted reset does not. Restore the
        # post-setup RNG state, so both paths start the rollout from it (randomized lighting
        # draws from it every frame).
        rng_after_setup = np.random.get_state()
        results = self._gen_descr(self.task_name, [self.episode_info["info"]], 100)
        if not results or not results[0].get("unseen"):
            raise RuntimeError(f"no unseen instructions generated for {self.task_name}")
        self.instruction = str(np.random.choice(results[0]["unseen"]))
        np.random.set_state(rng_after_setup)
        self.env.set_instruction(instruction=self.instruction)
        return self._reset_payload(now_seed)

    def _reset_payload(self, seed_used: int) -> dict:
        payload = self._obs()
        payload["seed_used"] = seed_used
        payload["sim_state"] = {"actors": self._actor_poses()}
        return payload

    def _actor_poses(self) -> dict:
        """{key: {"pose", "name"}} of every rigid actor. RoboTwin names entities by model, so a
        name shared by several entities (actors and articulation links) is keyed by its
        per-scene id on every occurrence."""
        scene = self.env.scene
        actors = list(scene.get_all_actors())
        named = [(a.get_name(), int(a.per_scene_id)) for a in actors]
        named += [
            (link.get_name(), int(link.entity.per_scene_id))
            for art in scene.get_all_articulations()
            for link in art.get_links()
        ]
        counts = Counter(name for name, _ in named)
        poses = {}
        for a in actors:
            name, sid = a.get_name(), int(a.per_scene_id)
            key = name if name and counts[name] == 1 else f"{name}#{sid}"
            poses[key] = {"pose": _pose7(a.get_pose()), "name": name}
        return poses

    def _obs(self) -> dict:
        obs = self.env.get_obs()
        view = obs["observation"]
        images = {}
        for rt_cam, canon in _CAM_MAP.items():
            if rt_cam not in view:
                raise KeyError(f"camera {rt_cam!r} missing from obs (have {list(view)})")
            images[canon] = np.asarray(view[rt_cam]["rgb"], dtype=np.uint8)
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        payload = {"images": images, "state": state.tolist(), "instruction": self.instruction}
        return payload

    def step(self, action: list, need_obs: bool) -> dict:
        if self.env is None:
            raise RuntimeError("step before make/reset")
        act = np.asarray(action, dtype=np.float64)
        if act.shape != (14,):
            raise ValueError(f"expected 14-dim aloha qpos action, got shape {act.shape}")
        self.env.take_action(act, action_type="qpos")
        success = bool(self.env.eval_success)
        truncated = bool(self.env.take_action_cnt >= self.env.step_lim)
        reply: dict[str, Any] = {"success": success, "terminated": success, "truncated": truncated}
        if need_obs:
            reply.update(self._obs())
        return reply

    def close(self) -> None:
        if self.env is not None:
            self._close_quiet()


def serve(host: str, port: int, port_file: str | None, sim: RoboTwinSim) -> None:
    """Serve successive clients, one at a time, until killed.

    Writes the port file only after bind and listen succeed: the launcher waits for this file.
    With --port 0, the file holds the ephemeral port. ``sim`` (CuRobo + SAPIEN) is expensive
    and make() changes its task, so it is kept across client disconnects and stray probes.
    Only the per-client socket is replaced."""

    def reset(req: dict) -> dict:
        trust = req.get("trust", False)
        if not isinstance(trust, bool):
            raise TypeError(f"reset 'trust' must be a bool, got {trust!r}")
        return sim.reset(req["seed"], trust, req.get("instruction"))

    handlers = {
        "make": lambda req: sim.make(req["task"]),
        "reset": reset,
        "step": lambda req: sim.step(req["action"], bool(req.get("need_obs", True))),
    }
    srv = sim_protocol.listen(host, port, port_file)
    while True:
        conn, _ = srv.accept()
        print("[robotwin_sim_server] client connected", flush=True)
        try:
            sim_protocol.serve(conn, handlers, "robotwin_sim_server")
        except OSError as e:  # closed, reset (a tunnel drop) or a non-protocol frame
            print(f"[robotwin_sim_server] client gone ({e!r}); awaiting next", flush=True)
        finally:
            conn.close()


def main() -> None:
    """Run RoboTwin's render check first (prints "Render Well" or "Render Error"). A failure
    there is not fatal: make reports the real error."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", required=True, help="path to the RoboTwin clone (with assets)")
    p.add_argument(
        "--task-config",
        default="demo_clean",
        help="task_config/<name>.yml: demo_clean (easy), demo_randomized (hard), "
        "demo_<axis> (Clean2Rand)",
    )
    sim_protocol.add_server_args(p)
    a = p.parse_args()

    os.chdir(os.path.abspath(a.repo))
    sys.path.insert(0, os.path.join(os.path.abspath(a.repo), "script"))
    try:
        from test_render import Sapien_TEST  # type: ignore[import-not-found]  # noqa: E402

        Sapien_TEST()
    except Exception as e:  # noqa: BLE001
        print(f"[robotwin_sim_server] render probe raised: {e}", flush=True)

    sim = RoboTwinSim(a.repo, a.task_config)
    serve(a.host, a.port, a.port_file, sim)


if __name__ == "__main__":
    main()
