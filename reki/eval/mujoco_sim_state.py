"""MuJoCo scene state as the per-step ``sim_state`` dict that ``RolloutTraj`` records.

The LIBERO rollout imports this module in-process. scripts/eval/robocasa365_sim_server.py runs
in its own micromamba env and imports it by name, as it does ``sim_protocol``. So this module
imports only numpy and mujoco. ``reki.eval.rollout.traj.RolloutTraj`` turns the per-step dicts
into ``traj/epNNN.npz``.

  actors        every free-joint body (the movable objects), world pose [x y z qw qx qy qz]
  articulations every kinematic tree rooted at a world child that has a non-free joint
                (the robot, drawers, doors, knobs, buttons): root pose + joint qpos
  contacts      (body_a, body_b, normal force N) summed over the pair's contact points at
                the last physics substep
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco  # type: ignore[import]
import numpy as np

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_QPOS_WIDTH = {_FREE: 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}


def _body_name(m: mujoco.MjModel, b: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


@dataclass
class SceneIndex:
    """Cached MuJoCo body, articulation, and joint indices for scene snapshots."""

    actor_ids: list[int]
    actor_names: list[str]
    art_ids: list[int]
    art_names: list[str]
    art_joints: dict[str, list[str]]
    art_qpos_idx: dict[str, np.ndarray]

    @classmethod
    def build(cls, m: mujoco.MjModel) -> "SceneIndex":
        """A scene index built from a MuJoCo model."""
        actor_ids = sorted({int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == _FREE})
        trees: dict[int, list[int]] = {}
        for j in range(m.njnt):
            if m.jnt_type[j] != _FREE:
                trees.setdefault(int(m.body_rootid[m.jnt_bodyid[j]]), []).append(j)
        art_ids = sorted(trees)
        art_names = [_body_name(m, r) for r in art_ids]
        art_joints, art_qpos_idx = {}, {}
        for r, name in zip(art_ids, art_names):
            js = trees[r]
            art_joints[name] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in js]
            art_qpos_idx[name] = np.concatenate(
                [
                    np.arange(
                        m.jnt_qposadr[j], m.jnt_qposadr[j] + _QPOS_WIDTH.get(int(m.jnt_type[j]), 1)
                    )
                    for j in js
                ]
            )
        return cls(
            actor_ids=actor_ids,
            actor_names=[_body_name(m, b) for b in actor_ids],
            art_ids=art_ids,
            art_names=art_names,
            art_joints=art_joints,
            art_qpos_idx=art_qpos_idx,
        )


def _pose(d: mujoco.MjData, b: int) -> list[float]:
    return [*d.xpos[b].tolist(), *d.xquat[b].tolist()]


def snapshot(m: mujoco.MjModel, d: mujoco.MjData, idx: SceneIndex, t: int) -> dict:
    """Actor poses, articulation state, and contact forces captured at step ``t``."""
    pair_force: dict[tuple[str, str], float] = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        a, b = sorted(
            (_body_name(m, m.geom_bodyid[c.geom1]), _body_name(m, m.geom_bodyid[c.geom2]))
        )
        pair_force[(a, b)] = pair_force.get((a, b), 0.0) + float(f6[0])
    return {
        "t": int(t),
        "actors": {
            n: {"pose": _pose(d, b), "dynamic": True}
            for n, b in zip(idx.actor_names, idx.actor_ids)
        },
        "articulations": {
            n: {"root": _pose(d, r), "qpos": d.qpos[idx.art_qpos_idx[n]].tolist()}
            for n, r in zip(idx.art_names, idx.art_ids)
        },
        "contacts": [(a, b, f) for (a, b), f in pair_force.items()],
        "contact_quantity": "force",
    }
