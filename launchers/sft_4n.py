"""Finetuning of kyutai/reki-2b on RoboCasa365: configs/sft.yaml plus the fields below, on
32 GPUs (4 nodes x 8).

    pixi run python launchers/sft_4n.py table
    pixi run python launchers/sft_4n.py select 0 submit
    pixi run python launchers/sft_4n.py select 0 eval submit

Environment: see launchers/common.py.
"""

from sft import PRETRAINED

BENCHMARKS: dict[str, dict[str, object]] = {
    "robocasa365": {
        "dataset": "robocasa365",
        "steps": 25_000,
        "warmup_steps": 1_000,
        "mirror_prob": 0.5,
        "state_mlp_embodiments": {"panda_omron": 16, "panda_omron_mirrored": 16},
        "gradient_accumulation_steps": 1,
    },
}

EVAL_TASK = {"robocasa365": "robocasa365"}


def overrides(benchmark: str) -> dict[str, object]:
    """TrainConfig fields this launcher passes on top of configs/sft.yaml."""
    return {"pretrained_path": PRETRAINED, **BENCHMARKS[benchmark]}


if __name__ == "__main__":
    from common import OUTPUT_DIR, RekiGrid, as_cli, grid_kwargs, runs_dir, train_base_cmd

    tag = "sft_4n"
    config = "configs/sft.yaml"
    grid = RekiGrid(
        base_name=tag,
        base_cmd=train_base_cmd(config),
        log_dir=runs_dir(tag),
        **grid_kwargs(total_gpus=32),  # type: ignore[arg-type]
        setup={name: as_cli(overrides(name)) for name in BENCHMARKS},
        output_dir=OUTPUT_DIR,
    )
    grid.default_eval_tasks = EVAL_TASK
    grid.config = config
    grid.fire()
