#!/bin/bash -l
#SBATCH --job-name=lgsa_env
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
# Recipient supplied by submit_checked.sh as sbatch --mail-user=...
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
: "${LGSA_EMAIL:?Use slurm/submit_checked.sh and set a real LGSA_EMAIL}"
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
bash environment/setup_lgsa_goemo.sh
date --iso-8601=seconds
