#!/bin/bash -l
# First-time setup only; run from setup_assets.sbatch on an allocated GPU.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
export CONDA_ENV="${CONDA_ENV:-/gpfs/work/aac/zixinfeng23/.conda/envs/fypemotion}"
if [[ -z "${SLURM_JOB_ID:-}" || "$(uname -s)" != Linux ]]; then
    echo "Use sbatch slurm/setup_assets.sbatch on HPC." >&2
    exit 1
fi
if [[ "$CONDA_ENV" != /* || "${CONDA_ENV##*/}" != fypemotion ]]; then
    echo "Only the dedicated absolute fypemotion environment may be used." >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p environment logs
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
if [[ -f "$CONDA_ENV/.fypemotion-setup-complete" ]]; then
    source hpc/activate_env.sh
    "$PYTHON" -m pip check
    "$PYTHON" -m scripts.check_environment
    echo "Existing FYP environment verified; no packages reinstalled."
    exit 0
fi
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
    # -p fixes the exact shared-GPFS prefix even if the school's envs_dirs differs.
    conda create -p "$CONDA_ENV" python=3.10 pip -y --override-channels -c defaults
fi
source hpc/activate_env.sh
"$PYTHON" -m pip install pip==25.0.1
"$PYTHON" -m pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install -r requirements-nlp.txt
"$PYTHON" -m pip check
"$PYTHON" -m scripts.check_environment --save environment/torch-info.json | tee environment/torch-info.txt
"$PYTHON" --version > environment/python-version.txt
"$PYTHON" -m pip freeze > environment/pip-freeze.txt
"$PYTHON" -m unittest discover -s tests -v
touch "$CONDA_ENV/.fypemotion-setup-complete"
echo "FYP environment setup and GPU check passed."
