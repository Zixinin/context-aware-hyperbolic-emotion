#!/bin/bash
# Source this file: source hpc/activate_env.sh
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
export CONDA_ENV="${CONDA_ENV:-/gpfs/work/aac/zixinfeng23/.conda/envs/fypemotion}"
export CUDA_MODULE="cuda/11.8.0-gcc-8.5.0-d7ndetl"
if [[ "$CONDA_ENV" != /* || "${CONDA_ENV##*/}" != "fypemotion" ]]; then
    echo "CONDA_ENV must be an absolute path to the dedicated fypemotion environment." >&2
    return 1
fi
fyp_restore_nounset=0
case $- in *u*) fyp_restore_nounset=1 ;; esac
set +u
module load "$CUDA_MODULE" || return 1
source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
conda activate "$CONDA_ENV" || return 1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHON="$CONDA_ENV/bin/python"
export HF_HOME="${HF_HOME:-/gpfs/work/aac/zixinfeng23/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
hash -r
if [[ "$fyp_restore_nounset" == 1 ]]; then set -u; fi
unset fyp_restore_nounset
if [[ "$CONDA_PREFIX" != "$CONDA_ENV" || ! -x "$PYTHON" ]]; then
    echo "Conda activation failed." >&2
    return 1
fi
