"""The simulation benchmarks: the training dataset and prompt fields of each, and their task
lists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    """Eval-time constants of one benchmark: the training dataset it maps to and the prompt
    fields. ``camera_keys`` is the canonical camera order the policy expects."""

    dataset: str
    embodiment: str
    robot_type: str
    control_mode: str
    camera_keys: tuple[str, ...]
    fps: int | None
    state_layout: str


BENCHMARKS: dict[str, Benchmark] = {
    "libero": Benchmark(
        dataset="libero",
        embodiment="single_arm_ee_euler",
        robot_type="franka",
        control_mode="delta end-effector",
        camera_keys=("camera_top", "camera_wrist_left"),
        fps=10,
        state_layout="[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]",
    ),
    "robotwin": Benchmark(
        dataset="robotwin2_clean",
        embodiment="aloha",
        robot_type="aloha",
        control_mode="absolute joint",
        camera_keys=("camera_top", "camera_wrist_left", "camera_wrist_right"),
        fps=None,
        state_layout="[left_arm(6), left_gripper, right_arm(6), right_gripper] joint positions",
    ),
    "robocasa365": Benchmark(
        dataset="robocasa365",
        embodiment="panda_omron",
        robot_type="PandaOmron",
        control_mode="delta end-effector",
        camera_keys=(),
        fps=20,
        state_layout="the RoboCasa365 dataset 16-dim state",
    ),
}


def benchmark_tasks(benchmark: str) -> list[tuple[str, Path]]:
    """(task name, leaf dir) of a per-task benchmark's eval dataset; empty otherwise."""
    from reki.data.robotics.lerobot_io import find_collection_leaves

    if benchmark == "robotwin":
        return [(p.name, p) for p in find_collection_leaves(BENCHMARKS[benchmark].dataset)]
    return []
