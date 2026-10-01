#!/bin/bash
# One-time environment setup on Narval (Digital Research Alliance of Canada).
# Uses the Alliance wheelhouse (--no-index); no internet access is needed on compute nodes.
set -euo pipefail
module load StdEnv/2023 python/3.11
ENV_DIR=${ENV_DIR:-$HOME/envs/prcert}
virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy pandas matplotlib
python -c "import numpy, scipy, pandas, matplotlib; print('environment ready:', numpy.__version__, scipy.__version__)"
