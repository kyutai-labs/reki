"""RoboCasa365 sim server (runs in the robocasa365 micromamba env).

Holds a PandaOmron RoboCasaGymEnv and serves make/reset/step/close over a localhost socket
(``reki/eval/sim_protocol.py``); scripts/eval/rollout.py drives it with the policy in its own env
(this env pins numpy 2.2.5, mujoco 3.3.1, and robosuite @ 5ce6643 in
``scripts/eval/install.sh``). Serves one client, then exits.

Protocol (length-prefixed pickle, each request → one reply):
  {"cmd":"make","env_name":"robocasa/<Task>","split":str|None}
                                               -> {"max_episode_steps":int}
     max_episode_steps is the task horizon of the robocasa v1.0.1 registry (450-4350 steps);
     the client enforces it (robosuite's own horizon is off).
     split: kitchen scene and object split: "pretrain" (the official multitask-track eval
     scenes), "target", "all", or None (the robocasa default layouts).
  {"cmd":"reset","seed":int}                   -> obs payload + {articulation_joints}
  {"cmd":"step","action":list[12],"need_obs":bool,"sim_state":bool}
                                               -> {success,terminated,truncated}
                                                  (+ obs payload if need_obs, default True)
                                                  (+ sim_state, see mujoco_sim_state.py)
  {"cmd":"close"}                              -> {"ok":True}  (and exits)

obs payload: {"images": {cam_name: uint8[256,256,3] for 3 cams}, "state": float32[16]
(dataset order), "instruction": str}.

need_obs=False skips the camera renders of the step (most of its time). robosuite keeps sampling
the camera observables on their own clock; the gated sensors return the last frame instead of
rendering, so the rendered frames, the physics and the rng are those of a full step.

Parity with the training dataset (the GR00T LeRobot export):
  * State 16 = base_position(3) + base_rotation(4,quat) + ee_pos_rel(3) +
    ee_rot_rel(4,quat) + gripper_qpos(2), in this order (modality.json, and the
    state_slice of DATASETS["robocasa365"]).
  * Action 12 (policy output, LeRobot order) = base_motion(4) + control_mode(1) +
    ee_pos(3,Δ) + ee_rot(3,Δ) + gripper_close(1). The server maps these slices itself;
    robocasa's convert_action expects the HDF5 order ([ee, grip, base, mode]).
  * Cameras: robot0_agentview_left / _right / eye_in_hand, 256². The server returns them by
    name; the rollout orders them as training does (reki.eval.rollout.remote.robocasa365_cameras).
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import robocasa  # type: ignore[import]  # noqa: F401  (registers robocasa/<Task> gym ids)
from robocasa.utils.dataset_registry_utils import get_task_horizon  # type: ignore[import]
from robosuite.utils.observables import sensor  # type: ignore[import]

sys.path.append(
    str(Path(__file__).resolve().parents[2] / "reki" / "eval")
)  # sim_protocol etc.: no reki deps
import sim_protocol  # type: ignore[import-not-found]  # noqa: E402
from mujoco_sim_state import SceneIndex, snapshot  # type: ignore[import-not-found]  # noqa: E402

_STATE_ORDER = [
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
]
_CAM_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]


def _state16(obs: dict) -> np.ndarray:
    parts = []
    for key, dim in _STATE_ORDER:
        v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        assert v.shape[0] == dim, f"{key}: expected {dim}, got {v.shape[0]}"
        parts.append(v)
    return np.concatenate(parts)


def _action_dict(vec12: np.ndarray) -> dict:
    """Policy flat-12 (LeRobot order) → robosuite action.* dict (bypasses convert_action)."""
    a = np.asarray(vec12, dtype=np.float32)
    assert a.shape[0] == 12, f"action must be 12-dim, got {a.shape[0]}"
    return {
        "action.base_motion": a[0:4],
        "action.control_mode": a[4:5],
        "action.end_effector_position": a[5:8],
        "action.end_effector_rotation": a[8:11],
        "action.gripper_close": a[11:12],
    }


def _obs_payload(obs: dict) -> dict:
    return {
        "images": {k.split(".", 1)[1]: np.asarray(obs[k], dtype=np.uint8) for k in _CAM_KEYS},
        "state": _state16(obs),
        "instruction": str(obs.get("annotation.human.task_description", "")),
    }


class Server:
    """The RoboCasa365 environment and request handlers for one client."""

    def __init__(self) -> None:
        self.env: Any = None
        self.scene: SceneIndex | None = None
        self.cameras: list[str] = []
        self.render = True
        self.n_renders = 0

    def _gated(self, render_cam: Any) -> Any:
        last = None

        @sensor(modality=render_cam.__modality__)
        def cam(obs_cache: dict) -> Any:
            nonlocal last
            if self.render or last is None:
                last = render_cam(obs_cache)
                self.n_renders += 1
            return last

        cam.gated = True  # pyright: ignore[reportFunctionMemberAccess]
        return cam

    def _gate_cameras(self) -> list[str]:
        """Gate the camera sensors on ``self.render``. A hard reset installs new sensors."""
        rs = self.env.unwrapped.env
        names = [n for n, o in rs._observables.items() if o.modality == "image"]
        for name in names:
            render_cam = rs._observables[name]._sensor
            if not getattr(render_cam, "gated", False):
                rs.modify_observable(name, "sensor", self._gated(render_cam))
        return names

    def make(self, env_name: str, split: str | None = None) -> dict:
        self.env = gym.make(env_name, disable_env_checker=True, split=split)
        horizon = int(get_task_horizon(env_name.split("/")[-1]))
        return {"max_episode_steps": horizon}

    def reset(self, seed: int | None) -> dict:
        obs, _ = self.env.reset(seed=seed)
        self.scene = scene = SceneIndex.build(self.env.unwrapped.env.sim.model._model)
        self.cameras = self._gate_cameras()
        return {**_obs_payload(obs), "articulation_joints": scene.art_joints}

    def step(self, action: list, need_obs: bool, want_sim_state: bool) -> dict:
        self.render, self.n_renders = need_obs, 0
        try:
            obs, _reward, done, truncated, info = self.env.step(_action_dict(np.asarray(action)))
        finally:
            self.render = True
        payload = {
            "success": bool(info.get("success", False)),
            "terminated": bool(done),
            "truncated": bool(truncated),
        }
        if need_obs:
            # The frames must come from this step, not from an earlier gated one.
            assert self.n_renders == len(self.cameras), f"{self.n_renders} camera renders"
            payload.update(_obs_payload(obs))
        if want_sim_state:
            assert self.scene is not None, "step before reset"
            rs = self.env.unwrapped.env
            payload["sim_state"] = snapshot(
                rs.sim.model._model, rs.sim.data._data, self.scene, rs.timestep
            )
        return payload


def _serve(conn: socket.socket) -> None:
    srv = Server()
    sim_protocol.serve(
        conn,
        {
            "make": lambda msg: srv.make(msg["env_name"], msg.get("split")),
            "reset": lambda msg: srv.reset(msg.get("seed")),
            "step": lambda msg: srv.step(
                msg["action"], msg.get("need_obs", True), msg["sim_state"]
            ),
        },
        "robocasa365_sim_server",
    )
    if srv.env is not None:
        srv.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sim_protocol.add_server_args(parser)
    args = parser.parse_args()

    sock = sim_protocol.listen(args.host, args.port, args.port_file)
    conn, _ = sock.accept()
    try:
        _serve(conn)
    finally:
        conn.close()
        sock.close()


if __name__ == "__main__":
    main()
