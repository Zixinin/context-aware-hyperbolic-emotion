#!/bin/bash -l
#SBATCH --job-name=lgsa_wos_frozen
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
set -u
date --iso-8601=seconds
hostname
nvidia-smi
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u scripts/run_frozen_smoke.py
date --iso-8601=seconds
