#!/bin/bash -l
#SBATCH --job-name=goemo_smoke
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-smoke-%j.out
#SBATCH --error=logs/slurm-smoke-%j.err
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
source "$PROJECT_ROOT/hpc/job_start.sh"
"$PYTHON" -u -m src.train --mode smoke
date -Is
