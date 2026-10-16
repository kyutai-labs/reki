"""Finetuning of kyutai/reki-2b on LIBERO and RoboTwin clean: configs/sft.yaml plus the fields
below, on 8 GPUs.

    pixi run python launchers/sft.py table
    pixi run python launchers/sft.py select 0 submit
    pixi run python launchers/sft.py select 0 eval --step 15000 submit
    pixi run python launchers/sft.py select 0 results
    pixi run python launchers/sft.py select 1 eval --task robotwin_clean2rand submit

Environment: see launchers/common.py.
"""

PRETRAINED = "kyutai/reki-2b"

ALOHA = {"aloha": 14, "aloha_mirrored": 14}

BENCHMARKS: dict[str, dict[str, object]] = {
    "libero": {
        "dataset": "libero",
        "steps": 15_000,
        "warmup_steps": 1_000,
        "mirror_prob": 0.0,
        "state_mlp_embodiments": {"single_arm_ee_euler": 8},
        "aug_random_crop": 0.95,
        "aug_random_scale": True,
        "pin_overhead_camera": False,
        "gradient_accumulation_steps": 1,
    },
    "robotwin_clean": {
        "dataset": "robotwin2_clean",
        "steps": 20_000,
        "warmup_steps": 1_000,
        "backbone_lr_scale": 0.2,
        "mirror_prob": 0.5,
        "state_mlp_embodiments": ALOHA,
        "gradient_accumulation_steps": 1,
    },
}

EVAL_TASK = {
    "libero": "libero",
    "robotwin_clean": "robotwin",
}


def overrides(benchmark: str) -> dict[str, object]:
    """TrainConfig fields this launcher passes on top of configs/sft.yaml."""
    return {"pretrained_path": PRETRAINED, **BENCHMARKS[benchmark]}


if __name__ == "__main__":
    from common import OUTPUT_DIR, RekiGrid, as_cli, grid_kwargs, runs_dir, train_base_cmd

    tag = "sft"
    config = "configs/sft.yaml"
    grid = RekiGrid(
        base_name=tag,
        base_cmd=train_base_cmd(config),
        log_dir=runs_dir(tag),
        **grid_kwargs(total_gpus=8),  # type: ignore[arg-type]
        setup={name: as_cli(overrides(name)) for name in BENCHMARKS},
        output_dir=OUTPUT_DIR,
    )
    grid.default_eval_tasks = EVAL_TASK
    grid.config = config
    grid.fire()
