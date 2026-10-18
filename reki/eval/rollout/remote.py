"""RoboTwin and RoboCasa365: the simulator runs in its own Python env as a sim
server (``scripts/eval/*_sim_server.py``) and the rollout drives it over one socket
(``reki.eval.sim_protocol``)."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import socket
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from reki.eval import sim_protocol
from reki.eval.protocols import socket_bench_tasks
from reki.eval.robocasa365_protocol import EVAL_SPLIT, SUITE_OF
from reki.eval.rollout.loop import Adapter, Unit
from reki.inference.features import EnvObs, unnormalize_action
from reki.policy.train_config import TrainConfig


def robocasa365_cameras(max_cameras: int | None) -> tuple[list[str], list[int]]:
    """(sim camera names, canonical camera indices) ranked and capped exactly as training."""
    from reki.data.robotics.cameras import CAMERA_RANKING, select_top_cameras
    from reki.data.robotics.registry import DATASETS

    cam_map = DATASETS["robocasa365"].cameras
    sel_keys = select_top_cameras(cam_map, max_cameras)
    cam_short = [k.rsplit(".", 1)[-1] for k in sel_keys]
    return cam_short, [CAMERA_RANKING.index(cam_map[k].rsplit(".", 1)[-1]) for k in sel_keys]


class RemoteEnv:
    """One task env of a sim server. ``timeouts``: seconds per command (None: the socket's)."""

    def __init__(
        self,
        conn: socket.socket,
        make: dict[str, Any],
        cameras: list[str] | None,
        timeouts: dict[str, float] | None = None,
    ) -> None:
        self.conn = conn
        self.cameras = cameras
        self.timeouts = timeouts
        self.make_reply = self._rpc({"cmd": "make", **make})
        self.max_episode_steps = int(self.make_reply["max_episode_steps"])

    def _rpc(self, msg: dict) -> dict:
        if self.timeouts is not None:
            self.conn.settimeout(self.timeouts[msg["cmd"]])
        return sim_protocol.rpc(self.conn, msg)

    def reset(self, seed: int, episode: int = 0) -> dict:
        return self._rpc({"cmd": "reset", "seed": seed})

    def step(self, action: np.ndarray, *, need_obs: bool, sim_state: bool) -> tuple:
        msg = {"cmd": "step", "action": np.asarray(action, np.float32).tolist()}
        reply = self._rpc({**msg, "need_obs": need_obs, "sim_state": sim_state})
        return reply, bool(reply["success"]), bool(reply["terminated"]), bool(reply["truncated"])

    def _images(self, raw: dict) -> list[np.ndarray]:
        assert self.cameras is not None, (
            "the adapter sets the cameras (RoboCasa365: in configure())"
        )
        return [raw["images"][c] for c in self.cameras]

    def observe(self, raw: dict) -> EnvObs:
        return EnvObs(
            images=[Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in self._images(raw)],
            state_raw=np.asarray(raw["state"], dtype=np.float32),
            task_text=raw["instruction"],
        )

    def frame(self, raw: dict) -> np.ndarray:
        return np.asarray(self._images(raw)[0], np.uint8)

    def traj_start(self, raw: dict) -> dict:
        return raw

    def traj_step(self, raw: dict) -> dict:
        return raw


# The server rejects at most 90 seeds per reset call; the search continues this many calls.
ROBOTWIN_MAX_SEARCH_CALLS = 10
_SEARCH_GAVE_UP = re.compile(r"seed search gave up: (\d+) seeds rejected .* starting at seed (\d+)")
# A trusted reset must rebuild the recorded scene: actor poses and state agree to this (m, rad).
_TRUST_ATOL = 1e-4


class RoboTwinSeedCache:
    """Accepted RoboTwin seeds, shared by every eval on the same scene code.

    A searched reset walks forward until the scripted expert solves the scene (seconds to an
    hour). The cache stores one search per (fingerprint, task, requested seed); later evals
    reset with ``trust=True`` on the recorded seed and instruction. The fingerprint is the
    ``code`` block of the make reply plus the instruction type. A trusted reset that builds a
    different scene raises. The first writer of a key wins (hard-linked pickle).
    """

    def __init__(self, root: Path, task: str, make_reply: dict) -> None:
        code = make_reply["code"]
        keys = ("code_md5", "task_config", "task_config_md5", "asset_list_md5")
        fingerprint = {"instruction_type": make_reply["instruction_type"]}
        fingerprint |= {k: code[k] for k in keys}
        if bad := {k: v for k, v in fingerprint.items() if str(v).startswith("error:")}:
            raise RuntimeError(f"RoboTwin seed cache: unreadable server fingerprint {bad}")
        self.fingerprint = fingerprint
        tag = hashlib.md5(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
        config_dir = root / f"{fingerprint['task_config']}_{fingerprint['instruction_type']}_{tag}"
        self.dir = config_dir / task
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (config_dir / "fingerprint.json").exists():
            _publish(config_dir / "fingerprint.json", json.dumps(fingerprint, indent=2).encode())

    def get(self, seed: int) -> dict | None:
        path = self.dir / f"{seed}.pkl"
        if not path.exists():
            return None
        entry = pickle.loads(path.read_bytes())
        if entry["fingerprint"] != self.fingerprint or entry["seed"] != seed:
            raise RuntimeError(f"RoboTwin seed cache entry {path} does not match its key")
        return entry

    def put(self, seed: int, reply: dict) -> None:
        entry = {
            "seed": seed,
            "fingerprint": self.fingerprint,
            "seed_used": int(reply["seed_used"]),
            "instruction": reply["instruction"],
            "actor_poses": _actor_poses(reply),
            "state": list(reply["state"]),
        }
        _publish(self.dir / f"{seed}.pkl", pickle.dumps(entry))

    @staticmethod
    def check(entry: dict, reply: dict) -> None:
        """Raise unless ``reply`` (a trusted reset) rebuilt the scene recorded in ``entry``."""
        got, want = _actor_poses(reply), entry["actor_poses"]
        where = f"trusted seed {entry['seed_used']}"
        if set(got) != set(want):
            raise RuntimeError(f"{where}: actors {sorted(got)}, cached {sorted(want)}")
        for key, pose in want.items():
            if not np.allclose(got[key], pose, atol=_TRUST_ATOL):
                raise RuntimeError(f"{where}: actor {key} at {got[key]}, cached {pose}")
        if not np.allclose(reply["state"], entry["state"], atol=_TRUST_ATOL):
            raise RuntimeError(f"{where}: state {reply['state']}, cached {entry['state']}")


def _actor_poses(reply: dict) -> dict[str, list[float]]:
    if "sim_state" not in reply:
        raise KeyError("the RoboTwin seed cache needs the actor poses of the reset reply")
    return {k: a["pose"] for k, a in reply["sim_state"]["actors"].items()}


def _publish(final: Path, data: bytes) -> None:
    """Write ``final`` whole or not at all; when it already exists, keep it."""
    fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=f".{final.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            os.link(tmp, final)
        except FileExistsError:
            pass
    finally:
        os.unlink(tmp)


def seed_cache_root(arg: str) -> tuple[Path | None, bool]:
    """(cache root, required) of ``--robotwin_seed_cache``: ``none``, ``auto`` or a directory."""
    if arg == "none":
        return None, False
    if arg == "auto":
        return Path("~/.cache/reki/robotwin_seed_cache").expanduser(), False
    return Path(arg).expanduser(), True


class RoboTwinEnv(RemoteEnv):
    """A RoboTwin task env. With ``seed_cache``, a cached seed gives a trusted reset on its
    accepted seed. An unknown seed starts a search, and the cache stores the result."""

    def __init__(
        self,
        conn: socket.socket,
        task: str,
        cameras: list[str] | None,
        timeouts: dict[str, float] | None,
        seed_cache: Path | None,
        seed_cache_required: bool = True,
    ) -> None:
        super().__init__(conn, {"task": task}, cameras, timeouts)
        self.cache = None
        if seed_cache is None:
            return
        if not self.make_reply["trusted_reset"]:
            print(
                f"[robotwin] {task}: no seed cache: apply scripts/eval/robotwin_arm_tag_fix.patch "
                "to the RoboTwin tree"
            )
            return
        try:
            self.cache = RoboTwinSeedCache(seed_cache, task, self.make_reply)
        except (OSError, RuntimeError) as e:
            if seed_cache_required:
                raise
            print(f"[robotwin] {task}: no seed cache: {e}")

    def search(self, seed: int) -> dict:
        """The server's seed search, continued past its per-call limit of rejected seeds."""
        start = seed
        for _ in range(ROBOTWIN_MAX_SEARCH_CALLS):
            try:
                return self._rpc({"cmd": "reset", "seed": start})
            except sim_protocol.SimServerError as e:
                m = _SEARCH_GAVE_UP.search(e.error)
                if m is None or int(m[2]) != start:
                    raise
                start += int(m[1])
                print(f"  [reset] seed {seed}: {m[1]} more rejected, continuing at {start}")
        raise RuntimeError(f"RoboTwin seed search from {seed} rejected every seed up to {start}")

    def reset(self, seed: int, episode: int = 0) -> dict:
        entry = self.cache.get(seed) if self.cache else None
        if entry is None:
            reply = self.search(seed)
            if self.cache:
                self.cache.put(seed, reply)
            return reply
        reply = self._rpc(
            {
                "cmd": "reset",
                "seed": entry["seed_used"],
                "trust": True,
                "instruction": entry["instruction"],
            }
        )
        RoboTwinSeedCache.check(entry, reply)
        return reply


class RemoteAdapter(Adapter):
    """A socket benchmark: one connection for the run, one ``make`` per task."""

    env_type = ""
    connect_timeout = 1800.0
    timeouts: dict[str, float] | None = None
    cameras: list[str] | None = None

    def __init__(self, tasks: str | list[str] | None, ckpt_dir: str, host: str, port: int) -> None:
        self.tasks = socket_bench_tasks(self.env_type, tasks, ckpt_dir)
        self.host, self.port = host, port
        self.conn: socket.socket | None = None

    def make_msg(self, task: str) -> dict[str, Any]:
        raise NotImplementedError

    def units(self) -> list[Unit]:
        return [Unit(task, task, task, {"task": task}) for task in self.tasks]

    @contextmanager
    def session(self) -> Iterator[None]:
        conn = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        print(f"[{self.tag}] connected to the sim server {self.host}:{self.port}")
        self.conn = conn
        try:
            yield
        finally:
            try:
                sim_protocol.rpc(conn, {"cmd": "close"})
            except Exception:
                pass
            conn.close()

    def make_env(self, unit: Unit) -> RemoteEnv:
        assert self.conn is not None, "make_env outside session()"
        return RemoteEnv(self.conn, self.make_msg(unit.task), self.cameras, self.timeouts)

    def decode(self, action_norm: torch.Tensor, stats: dict) -> np.ndarray:
        return unnormalize_action(action_norm, stats)


class RoboTwinAdapter(RemoteAdapter):
    """RoboTwin 2.0 ALOHA tasks.

    ``seed_cache`` is the ``RoboTwinSeedCache`` root, or None to search on every reset; it is used
    only when the server reports ``trusted_reset``, and without ``seed_cache_required`` an
    unusable cache falls back to the search."""

    benchmark = env_type = tag = "robotwin"
    per_task_stats = True
    connect_timeout = 300.0
    timeouts = {"make": 600.0, "reset": 3600.0, "step": 300.0, "close": 300.0}
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    n_cameras = 3
    image_hw = (480, 640)  # Large_D435, the resolution of the training data
    skip_obs_within_chunk = True

    def __init__(
        self,
        tasks: str | list[str] | None,
        ckpt_dir: str,
        host: str,
        port: int,
        seed_cache: Path | None = None,
        seed_cache_required: bool = True,
    ) -> None:
        super().__init__(tasks, ckpt_dir, host, port)
        self.seed_cache = seed_cache
        self.seed_cache_required = seed_cache_required

    def make_env(self, unit: Unit) -> RoboTwinEnv:
        assert self.conn is not None, "make_env outside session()"
        return RoboTwinEnv(
            self.conn,
            unit.task,
            self.cameras,
            self.timeouts,
            self.seed_cache,
            self.seed_cache_required,
        )

    def prompt_fps(self, stats_fps: float) -> int:
        return int(stats_fps)

    def env_fps(self, stats_fps: float) -> int:
        return int(stats_fps)


class RoboCasa365Adapter(RemoteAdapter):
    """The official RoboCasa365 multitask protocol (``reki.eval.robocasa365_protocol``): the 50
    target tasks in the pretrain kitchen scenes, with collection-wide stats (the unseen tasks
    have no dataset of their own)."""

    benchmark = env_type = tag = "robocasa365"
    skip_obs_within_chunk = True

    def configure(self, train_cfg: TrainConfig) -> None:
        self.cameras, self.camera_indices = robocasa365_cameras(train_cfg.max_cameras)
        self.n_cameras = len(self.cameras)
        print(f"[robocasa365] cameras {self.cameras} -> canonical indices {self.camera_indices}")

    def units(self) -> list[Unit]:
        if unknown := [t for t in self.tasks if t not in SUITE_OF]:
            raise KeyError(f"{unknown} are not RoboCasa365 target tasks (robocasa365_protocol.py)")
        return [
            Unit(t, t, t, {"task": t, "suite": SUITE_OF[t], "split": EVAL_SPLIT})
            for t in self.tasks
        ]

    def make_msg(self, task: str) -> dict[str, Any]:
        return {"env_name": f"robocasa/{task}", "split": EVAL_SPLIT}
