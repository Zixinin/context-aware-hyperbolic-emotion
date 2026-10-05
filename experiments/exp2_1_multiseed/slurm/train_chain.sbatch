#!/bin/bash -l
#SBATCH --job-name=fyp21_train
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
test -n "${SLURM_JOB_ID:-}"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/gpfs/work/aac/zixinfeng23/.cache/huggingface
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_MODULE=cuda/11.8.0-gcc-8.5.0-d7ndetl
set +u
module load "$CUDA_MODULE"
set -u
hostname
date --iso-8601=seconds
nvidia-smi
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u chain.py
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u finalize.py --train-job "$SLURM_JOB_ID"
date --iso-8601=seconds
echo GPU_TRAINING_AND_REPORT_PASS_AWAITING_SLURM_COMPLETION
