#!/bin/bash
# End-to-end check on a laptop or login node (about 3 minutes on 2 cores).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tests.test_core
python -m experiments.plan --scale smoke --out plan_smoke.json
python -m experiments.run_task --plan plan_smoke.json --results results_smoke --all --workers "${WORKERS:-2}"
python -m analysis.aggregate --plan plan_smoke.json --results results_smoke --out tables_smoke
python -m analysis.figures --tables tables_smoke --out figures_smoke --h-csweep 0.05
