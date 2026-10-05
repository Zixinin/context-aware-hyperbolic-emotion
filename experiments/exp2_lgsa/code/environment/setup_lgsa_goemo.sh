#!/bin/bash -l
# HPC GPU allocation only. Has NOT yet been run on HPC.
set -euo pipefail
PROJECT_ROOT=/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
LGSA_ENV=/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo
if [[ "$(uname -s)" != Linux || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit slurm/lgsa_environment.sbatch via slurm/submit_checked.sh on HPC." >&2
    exit 1
fi
cd "$PROJECT_ROOT"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
if [[ -e "$LGSA_ENV" && ! -f "$LGSA_ENV/.lgsa-owned" ]]; then
    echo "Existing unmarked lgsa_goemo environment: inspect before changing it." >&2
    exit 1
fi
if [[ ! -d "$LGSA_ENV" ]]; then
    conda create -p "$LGSA_ENV" python=3.10 pip -y --override-channels -c defaults
    touch "$LGSA_ENV/.lgsa-owned"
fi
set +u
conda activate "$LGSA_ENV"
set -u
export PATH="$CONDA_PREFIX/bin:$PATH"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
mkdir -p "$HF_HOME" environment
PYTHON=/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python
if [[ "$CONDA_PREFIX" != "$LGSA_ENV" ]]; then exit 1; fi
if [[ ! -f "$LGSA_ENV/.lgsa-setup-complete" ]]; then
    "$PYTHON" -m pip install pip==25.0.1
    "$PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    "$PYTHON" -m pip install -r environment/requirements-candidate.txt
fi
"$PYTHON" -m pip check
nvidia-smi
"$PYTHON" --version
"$PYTHON" environment/check_gpu.py
# Only freeze after the actual CUDA/PyG test passed.
"$PYTHON" -m pip freeze > "environment/requirements-lock-${SLURM_JOB_ID}.txt"
cp "environment/requirements-lock-${SLURM_JOB_ID}.txt" environment/requirements-lock.txt
touch "$LGSA_ENV/.lgsa-setup-complete"
