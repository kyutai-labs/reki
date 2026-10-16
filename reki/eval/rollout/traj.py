"""The ``--save_traj`` trace of one episode, saved as ``traj/epNNN.npz``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class RolloutTraj:
    """One episode's trace.

    Per predict: the env step, the input state, the full predicted chunk (normalized), the
    register-ring depth before the predict. Per executed step: the
    action sent to the sim and, on the MuJoCo benches (LIBERO(-Plus), RoboCasa365), the scene
    state of ``reki.eval.mujoco_sim_state`` (body poses, articulation qpos, contact forces).
    RoboTwin records no scene state."""

    _warned_no_sim_state = False

    def __init__(self, reset_obs: dict) -> None:
        self.reset_extras = {
            k: reset_obs[k] for k in ("seed_used", "instruction") if k in reset_obs
        }
        self.art_joints = reset_obs.get("articulation_joints")
        self.p_step: list[int] = []
        self.p_state: list[np.ndarray] = []
        self.p_chunk: list[np.ndarray] = []
        self.p_ring: list[int] = []
        self.s_action: list[np.ndarray] = []
        self.s_sim: list[dict | None] = []

    def predict(
        self,
        *,
        step: int,
        state: np.ndarray,
        chunk: torch.Tensor,
        ring_depth: int,
    ) -> None:
        """Record one policy prediction and its register depth."""
        self.p_step.append(step)
        self.p_state.append(np.asarray(state, np.float32))
        self.p_chunk.append(chunk.to(torch.float32).cpu().numpy())
        self.p_ring.append(ring_depth)

    def step(self, action: np.ndarray, reply: dict) -> None:
        """Record one executed action and the simulator state of the reply, if any."""
        self.s_action.append(np.asarray(action, np.float32))
        sim_state = reply.get("sim_state")
        if sim_state is None and not RolloutTraj._warned_no_sim_state:
            RolloutTraj._warned_no_sim_state = True
            print(
                "[traj] no 'sim_state' block from the sim: actor poses and contacts will "
                "be absent from traj/*.npz.",
                flush=True,
            )
        self.s_sim.append(sim_state)

    def _sim_arrays(self) -> dict[str, np.ndarray]:
        """The per-step sim states as arrays. The ragged contact lists become a dense
        [steps, pairs] matrix (force or impulse) over all pairs of the episode. Loading needs
        no pickle."""
        states = [st for st in self.s_sim if st is not None]
        if not states:
            return {}
        if len(states) != len(self.s_sim):
            raise ValueError("sim_state on some steps but not others — mixed sim-server?")
        names = list(states[0]["actors"])
        arts = list(states[0]["articulations"])
        pairs = sorted({(a, b) for st in states for a, b, _ in st["contacts"]})
        pair_col = {pair: i for i, pair in enumerate(pairs)}
        pose = np.empty((len(states), len(names), 7), np.float32)
        art_root = np.empty((len(states), len(arts), 7), np.float32)
        art_qpos = {
            a: np.empty((len(states), len(states[0]["articulations"][a]["qpos"])), np.float32)
            for a in arts
        }
        impulse = np.zeros((len(states), len(pairs)), np.float32)
        sim_step = np.empty(len(states), np.int32)
        for i, st in enumerate(states):
            if list(st["actors"]) != names or list(st["articulations"]) != arts:
                raise ValueError(f"sim body set changed mid-episode: {list(st['actors'])}")
            pose[i] = np.asarray([st["actors"][n]["pose"] for n in names], np.float32)
            art_root[i] = np.asarray([st["articulations"][a]["root"] for a in arts], np.float32)
            for a in arts:
                art_qpos[a][i] = np.asarray(st["articulations"][a]["qpos"], np.float32)
            for a, b, imp in st["contacts"]:
                impulse[i, pair_col[(a, b)]] = imp
            sim_step[i] = st["t"]
        out: dict[str, np.ndarray] = {
            "actor_names": np.asarray(names),
            "actor_pose": pose,
            "actor_dynamic": np.asarray([states[0]["actors"][n]["dynamic"] for n in names]),
            "articulation_names": np.asarray(arts),
            "articulation_root": art_root,
            "contact_pairs": np.asarray([f"{a}|{b}" for a, b in pairs]),
            f"contact_{states[0].get('contact_quantity', 'impulse')}": impulse,
            "sim_step": sim_step,
            **{f"articulation_qpos__{a}": q for a, q in art_qpos.items()},
            **{
                f"articulation_joints__{a}": np.asarray(js)
                for a, js in (self.art_joints or {}).items()
            },
        }
        return out

    def save(self, path: Path, **scalars: Any) -> None:
        """Save the trace and ``scalars`` as a compressed ``.npz``."""
        arrays: dict[str, Any] = {
            "predict_step": np.asarray(self.p_step, np.int32),
            "predict_state": np.stack(self.p_state),
            "predict_chunk": np.stack(self.p_chunk),
            "predict_ring_depth": np.asarray(self.p_ring, np.int32),
            "exec_action": np.stack(self.s_action),
            **self._sim_arrays(),
            **{k: np.asarray(v) for k, v in {**scalars, **self.reset_extras}.items()},
        }
        np.savez_compressed(path, **arrays)
