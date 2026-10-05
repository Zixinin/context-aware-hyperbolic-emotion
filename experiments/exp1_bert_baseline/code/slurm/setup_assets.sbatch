#!/bin/bash -l
#SBATCH --job-name=goemo_setup
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-setup-%j.out
#SBATCH --error=logs/slurm-setup-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file using sbatch}"
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
cd "$PROJECT_ROOT"
date -Is
hostname
nvidia-smi
bash hpc/setup_environment.sh
source hpc/activate_env.sh
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
"$PYTHON" -u -m scripts.prepare_assets --data-source "${DATA_SOURCE:-huggingface}"
"$PYTHON" -u -m scripts.analyze_dataset
echo "SETUP_AND_ASSETS_PASS"
