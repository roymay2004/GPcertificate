#!/bin/bash
# Submit every task of a plan as job arrays of at most CHUNK elements.
#   bash slurm/submit.sh plan_full.json $SCRATCH/prcert-results
# Optional: MAXRUN (concurrent array elements per chunk, default 400), CHUNK (default 1000).
set -euo pipefail
PLAN=${1:?plan file}
RESULTS=${2:?results directory}
CHUNK=${CHUNK:-1000}
MAXRUN=${MAXRUN:-400}
module load StdEnv/2023 python/3.11
source "${ENV_DIR:-$HOME/envs/prcert}/bin/activate"
NT=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$PLAN")
mkdir -p logs "$RESULTS"
echo "submitting $NT tasks from $PLAN -> $RESULTS"
for ((OFF=0; OFF<NT; OFF+=CHUNK)); do
  N=$(( NT - OFF < CHUNK ? NT - OFF : CHUNK ))
  sbatch --array=0-$((N - 1))%${MAXRUN} \
         --export=ALL,PLAN="$PLAN",RESULTS="$RESULTS",OFFSET="$OFF" slurm/array.sbatch
done
