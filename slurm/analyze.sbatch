#!/bin/bash
#SBATCH --account=def-roymay
#SBATCH --job-name=prcert-analysis
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
# sbatch --export=ALL,PLAN=plan_full.json,RESULTS=$SCRATCH/prcert-results slurm/analyze.sbatch
set -euo pipefail
module load StdEnv/2023 python/3.11
source "${ENV_DIR:-$HOME/envs/prcert}/bin/activate"
cd "$SLURM_SUBMIT_DIR"
python -m analysis.aggregate --plan "$PLAN" --results "$RESULTS" --out tables
python -m analysis.figures --tables tables --out figures
