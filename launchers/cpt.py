"""Pretraining of REKI from Qwen/Qwen3.5-2B on the full mixture: configs/cpt.yaml, 4 nodes x 8 GPUs.

    pixi run python launchers/cpt.py table
    pixi run python launchers/cpt.py select 0 submit

Environment: see launchers/common.py.
"""

if __name__ == "__main__":
    from common import OUTPUT_DIR, RekiGrid, grid_kwargs, runs_dir, train_base_cmd

    tag = "cpt"
    grid = RekiGrid(
        base_name=tag,
        base_cmd=train_base_cmd("configs/cpt.yaml"),
        log_dir=runs_dir(tag),
        **grid_kwargs(total_gpus=32),  # type: ignore[arg-type]
        output_dir=OUTPUT_DIR,
    )
    grid.config = "configs/cpt.yaml"
    grid.fire()
