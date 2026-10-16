#!/bin/bash
# Euler-stats sidecar (meta/norm_stats_euler.json, or norm_stats_euler_roll.json with a shifted
# state roll, e.g. libero) for the axis-angle datasets in $DATASETS.
#
# Submit from the repo root (env is exported to the job):
#   DATASETS="libero" sbatch scripts/data/sbatch/recompute_euler_stats.sbatch
#SBATCH --job-name=recompute_euler_stats
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${DATASETS:?space-separated dataset names}"
read -r -a NAMES <<< "$DATASETS"
pixi run --frozen python scripts/data/recompute_euler_stats.py --datasets "${NAMES[@]}" --overwrite
