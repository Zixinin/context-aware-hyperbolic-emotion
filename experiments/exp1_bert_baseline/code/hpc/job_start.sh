#!/bin/bash
# Shared startup for smoke and full GPU batch jobs.
if [[ -z "${SLURM_JOB_ID:-}" || "$(uname -s)" != Linux ]]; then
    echo "Submit training with sbatch; do not run the .sbatch file using bash." >&2
    exit 1
fi
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
cd "$PROJECT_ROOT"
source hpc/activate_env.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
date -Is
hostname
echo "Slurm Job ID: $SLURM_JOB_ID"
nvidia-smi
"$PYTHON" --version
"$PYTHON" -m scripts.check_environment
test -s assets/manifest.json
