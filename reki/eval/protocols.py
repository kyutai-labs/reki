"""The released eval protocols: the unit grid of each benchmark (what ``scripts/eval/rollout.py``
runs by default), the episode counts, and the result dirs ``launchers/eval_protocols.py`` writes
and ``reki.eval.results`` reads."""

from __future__ import annotations

import json
from pathlib import Path

from reki.eval.robocasa365_protocol import TARGET_SUITES as ROBOCASA365_SUITES

LIBERO_SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
LIBERO_TASK_IDS = tuple(range(10))
LIBERO_PLUS_AXES = (
    "background_textures",
    "camera_viewpoints",
    "language_instructions",
    "light_conditions",
    "objects_layout",
    "robot_initial_states",
    "sensor_noise",
)
# LIBERO-Plus tasks per suite, one count per axis of LIBERO_PLUS_AXES (task_classification.json
# of sylvestf/LIBERO-plus @ 4976dc3; 10,030 in total).
LIBERO_PLUS_TASKS = {
    "libero_object": (248, 396, 354, 297, 403, 398, 422),
    "libero_spatial": (258, 376, 390, 292, 385, 350, 351),
    "libero_goal": (281, 408, 410, 279, 425, 409, 379),
    "libero_10": (289, 419, 383, 274, 312, 393, 449),
}
ROBOTWIN_TASK_CONFIG = {
    "easy": "demo_clean",
    "hard": "demo_randomized",
    "background": "demo_background",
    "light": "demo_light",
    "clutter": "demo_clutter",
    "height": "demo_height",
}
# Clean2Rand: demo_clean.yml with ONE axis set to its demo_randomized value.
CLEAN2RAND_KEYS = {
    "background": {"random_background": "true", "clean_background_rate": "0.02"},
    "light": {"random_light": "true", "crazy_random_light_rate": "0.02"},
    # RoboTwin skips clutter when rand() < clean_background_rate; demo_clean's 1 skips it always.
    "clutter": {"cluttered_table": "true", "clean_background_rate": "0.02"},
    "height": {"random_table_height": "0.03"},
}


# Eval task -> the dirs its jobs write under <out_dir>, and the protocol episode count.
EVAL_BENCH_DIRS = {
    "libero": ("libero",),
    "libero_plus": ("libero_plus",),
    "robotwin": ("robotwin_easy", "robotwin_hard"),
    "robotwin_clean2rand": tuple(f"robotwin_{a}" for a in CLEAN2RAND_KEYS),
    "robocasa365": ("robocasa365",),
}
# The dirs of the Clean2Rand number: Easy, the 4 randomizations, Hard.
CLEAN2RAND_SCENE_DIRS = tuple(f"robotwin_{s}" for s in ("easy", *CLEAN2RAND_KEYS, "hard"))
N_EPISODES = {
    "libero": 50,
    "libero_plus": 1,
    "robotwin": 20,
    "robotwin_clean2rand": 20,
    "robocasa365": 50,
}

# Eval task -> the `benchmark` of the export spec (scripts/export_checkpoint.py --benchmark).
TASK_BENCHMARK = {
    "libero": "libero",
    "libero_plus": "libero",
    "robotwin": "robotwin",
    "robotwin_clean2rand": "robotwin",
    "robocasa365": "robocasa365",
}


def libero_plus_category(axis: str) -> str:
    """The task_classification.json category of a LIBERO_PLUS_AXES axis."""
    return axis.replace("_", " ").title()


def libero_plus_group(suite: str, axis: str) -> str:
    """The result-group path for a LIBERO-Plus suite and axis."""
    return f"libero_plus_{suite.removeprefix('libero_')}/{axis}"


# Bench dir -> result group -> the number of units of a full protocol run.
GROUP_SIZES = {
    "libero": {s: len(LIBERO_TASK_IDS) for s in LIBERO_SUITES},
    "libero_plus": {
        libero_plus_group(s, a): n
        for s, counts in LIBERO_PLUS_TASKS.items()
        for a, n in zip(LIBERO_PLUS_AXES, counts, strict=True)
    },
}


# Bench dir -> the result groups (dirs under it, or RoboCasa365 suites) of a full protocol run.
EXPECTED_GROUPS = {
    "libero": LIBERO_SUITES,
    "libero_plus": tuple(libero_plus_group(s, a) for s in LIBERO_SUITES for a in LIBERO_PLUS_AXES),
    "robocasa365": tuple(sorted(ROBOCASA365_SUITES)),
}


def socket_bench_tasks(env_type: str, task: str | list[str] | None, ckpt_dir: str) -> list[str]:
    """The task list of a socket bench. The default is the full grid of the bench: the ``tasks``
    of an exported checkpoint, else the dataset leaves (RoboCasa365: the 50 target tasks)."""
    if task is not None:
        return [task] if isinstance(task, str) else list(task)
    # = reki.inference.spec.INFERENCE_SPEC_FILE (that module imports torch, so it is not imported).
    spec_p = Path(ckpt_dir) / "reki_inference.json"
    if env_type == "robotwin" and not Path(ckpt_dir).exists():
        if ckpt_dir.startswith(("/", ".")) or ckpt_dir.count("/") != 1:
            raise FileNotFoundError(f"{ckpt_dir}: no such checkpoint directory")
        from huggingface_hub import hf_hub_download

        spec_p = Path(hf_hub_download(ckpt_dir, "reki_inference.json"))  # an HF model id
    if env_type == "robotwin" and spec_p.exists():
        return list(json.loads(spec_p.read_text())["tasks"])
    if env_type == "robocasa365":
        from reki.eval.robocasa365_protocol import TARGET50

        return list(TARGET50)
    if env_type == "robotwin":
        from reki.inference.benchmarks import benchmark_tasks

        return [name for name, _ in benchmark_tasks(env_type)]
    raise ValueError(f"no task grid known for env_type {env_type!r}")
