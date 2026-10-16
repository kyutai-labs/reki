#!/bin/bash
# RoboTwin 2.0 v2.1 -> v3.0 in place for one collection (stock converter + CHW->HWC fix + qk*),
# 8 CPU shards. Idempotent.
#
# Submit from the repo root (env is exported to the job):
#   COLLECTION=robotwin2_randomized sbatch scripts/data/sbatch/prepare_robotwin.sbatch
# When all shards finish, run once without --shard (global qk01/qk99; done leaves are skipped):
#   pixi run --frozen python scripts/data/prepare_robotwin.py --collections $COLLECTION
#SBATCH --job-name=prepare_robotwin
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${COLLECTION:?robotwin2_clean or robotwin2_randomized}"
pixi run --frozen python scripts/data/prepare_robotwin.py --collections "$COLLECTION" --shard "$SLURM_ARRAY_TASK_ID/8"
