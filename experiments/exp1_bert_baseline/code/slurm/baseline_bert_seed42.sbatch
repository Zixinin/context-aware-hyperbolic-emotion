#!/bin/bash -l
#SBATCH --job-name=goemo_bert42
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-bert42-%j.out
#SBATCH --error=logs/slurm-bert42-%j.err
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
source "$PROJECT_ROOT/hpc/job_start.sh"
: "${SMOKE_RUN:?Use bash hpc/submit_baseline.sh SMOKE_JOB_ID after smoke completes}"
"$PYTHON" -u -m src.train --mode full
date -Is
