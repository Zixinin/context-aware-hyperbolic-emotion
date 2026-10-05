#!/bin/bash -l
#SBATCH --job-name=fyp21_report
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=cpu8358
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/FYP-Experiment-2.1-MultiSeed
test -n "${SLURM_JOB_ID:-}"
: "${FYP21_TRAIN_JOB:?Missing upstream job ID}"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u finalize.py --train-job "$FYP21_TRAIN_JOB"
echo EXPERIMENT_2_1_REPORT_GENERATED_STOP
