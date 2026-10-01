#!/bin/bash
#SBATCH --account=def-roymay
#SBATCH --job-name=prcert
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --output=logs/%x_%A_%a.out
# Submitted by slurm/submit.sh, which sets --array and exports PLAN, RESULTS, OFFSET.
# Each array element runs one task (about one core-hour); finished tasks are skipped, so
# resubmitting the same array after a failure or timeout only reruns what is missing.
set -euo pipefail
module load StdEnv/2023 python/3.11
source "${ENV_DIR:-$HOME/envs/prcert}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -m experiments.run_task --plan "$PLAN" --results "$RESULTS" --array-offset "$OFFSET"
