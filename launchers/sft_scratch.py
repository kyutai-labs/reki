"""Finetuning of Qwen/Qwen3.5-2B without pretraining on LIBERO and RoboTwin clean: configs/sft.yaml
plus the fields below, with a 5k warmup, on 8 GPUs.

    pixi run python launchers/sft_scratch.py table
    pixi run python launchers/sft_scratch.py select 0 submit
    pixi run python launchers/sft_scratch.py select 0 eval --task libero_plus submit

Environment: see launchers/common.py.
"""

from sft import BENCHMARKS, EVAL_TASK

STEPS = {"libero": 30_000, "robotwin_clean": 40_000}


def overrides(benchmark: str) -> dict[str, object]:
    """TrainConfig fields this launcher passes on top of configs/sft.yaml."""
    return {**BENCHMARKS[benchmark], "steps": STEPS[benchmark], "warmup_steps": 5_000}


if __name__ == "__main__":
    from common import OUTPUT_DIR, RekiGrid, as_cli, grid_kwargs, runs_dir, train_base_cmd

    tag = "sft_scratch"
    config = "configs/sft.yaml"
    grid = RekiGrid(
        base_name=tag,
        base_cmd=train_base_cmd(config),
        log_dir=runs_dir(tag),
        **grid_kwargs(total_gpus=8),  # type: ignore[arg-type]
        setup={name: as_cli(overrides(name)) for name in STEPS},
        output_dir=OUTPUT_DIR,
    )
    grid.default_eval_tasks = {name: EVAL_TASK[name] for name in STEPS}
    grid.config = config
    grid.fire()
