#!/bin/bash
# qk* quantile stats (augment_quantiles.py). Loads every state/action row: needs RAM, not the login node.
# LEAVES is a dir of v3.0 leaves, sharded over the array (--shard i/N).
#
# Submit from the repo root (env is exported to the job):
#   LEAVES=<dir of v3.0 leaves> sbatch --array=0-15 scripts/data/sbatch/augment_quantiles.sbatch
#SBATCH --job-name=augment_quantiles
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --array=0-0
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${LEAVES:?dir of v3.0 leaves}"
pixi run --frozen python scripts/data/augment_quantiles.py --leaves "$LEAVES" \
  --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
