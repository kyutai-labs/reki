"""Finetuning of kyutai/reki-2b on RoboTwin clean + randomized: configs/sft.yaml plus the fields
below, on 24 GPUs (3 nodes x 8).

    pixi run python launchers/sft_3n.py table
    pixi run python launchers/sft_3n.py select 0 submit
    pixi run python launchers/sft_3n.py select 0 eval submit

Environment: see launchers/common.py.
"""

from sft import ALOHA, PRETRAINED

BENCHMARKS: dict[str, dict[str, object]] = {
    "robotwin_cleanrand": {
        "dataset": "robotwin2_clean,robotwin2_randomized",
        "steps": 25_000,
        "warmup_steps": 1_000,
        "mirror_prob": 0.5,
        "state_mlp_embodiments": ALOHA,
        "gradient_accumulation_steps": 1,
    },
}

EVAL_TASK = {"robotwin_cleanrand": "robotwin"}


def overrides(benchmark: str) -> dict[str, object]:
    """TrainConfig fields this launcher passes on top of configs/sft.yaml."""
    return {"pretrained_path": PRETRAINED, **BENCHMARKS[benchmark]}


if __name__ == "__main__":
    from common import OUTPUT_DIR, RekiGrid, as_cli, grid_kwargs, runs_dir, train_base_cmd

    tag = "sft_3n"
    config = "configs/sft.yaml"
    grid = RekiGrid(
        base_name=tag,
        base_cmd=train_base_cmd(config),
        log_dir=runs_dir(tag),
        **grid_kwargs(total_gpus=24),  # type: ignore[arg-type]
        setup={name: as_cli(overrides(name)) for name in BENCHMARKS},
        output_dir=OUTPUT_DIR,
    )
    grid.default_eval_tasks = EVAL_TASK
    grid.config = config
    grid.fire()
