# Launchers

Slurm job grids for the finetuning runs and the evaluations, built on
[launcheon](https://github.com/kyutai-labs/launcheon) (pixi `default` environment). Run them from
the repository root: `pixi run python launchers/<file>.py <verbs>`.

| File | Contents |
|---|---|
| `sft.py` | the LIBERO and RoboTwin clean runs (8 GPUs) |
| `sft_3n.py`, `sft_4n.py` | the RoboTwin clean + randomized run (24 GPUs) and the RoboCasa365 run (32 GPUs) |
| `sft_scratch.py` | LIBERO and RoboTwin clean from `Qwen/Qwen3.5-2B`, without pretraining |
| `eval.py` | evaluation of the released models (`MODELS`: Hub ids or local exports) |
| `eval_protocols.py` | the `scripts/eval/rollout.py` commands of each benchmark protocol; its CLI prints them without Slurm |
| `common.py` | `RekiGrid` with the `eval` and `results` verbs, the cluster settings, the run directories |

## Verbs

The verbs chain, and `select <i>` picks row `i` of `table`.
[docs/training.md](../docs/training.md#slurm) and
[docs/evaluation.md](../docs/evaluation.md#evaluate-your-checkpoints) give the commands and options.

## Environment

A run lives in `$REKI_RUNS_DIR/<launcher>/<hash>/`, with training under `train/` and results under
`eval/`. The Slurm scripts and logs of the eval jobs are under `$REKI_RUNS_DIR/<launcher>_eval/`.
`eval.py` writes to `$REKI_RUNS_DIR/models/<hash>/`. The cluster settings are `REKI_*` variables
and `SBATCH_PARTITION` in `.env` ([`.env.example`](../.env.example),
[docs/training.md](../docs/training.md#slurm)).
