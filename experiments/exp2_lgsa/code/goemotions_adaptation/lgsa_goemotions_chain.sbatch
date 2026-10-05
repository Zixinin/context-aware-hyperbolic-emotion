#!/bin/bash -l
#SBATCH --job-name=lgsa_goemo_s42
#SBATCH --account=hengyanliu_fyp
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Zixin.Feng23@student.xjtlu.edu.cn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set +u
module load cuda/11.8.0-gcc-8.5.0-d7ndetl
set -u
LGSA_PY=/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python
# Slurm signals batch shell; forward advance-timeout signal to its Python child.
LGSA_CHILD=''
trap 'if [[ -n "$LGSA_CHILD" ]]; then kill -USR1 "$LGSA_CHILD" 2>/dev/null || true; wait "$LGSA_CHILD" || true; fi; exit 1' USR1
run_phase() {
    "$LGSA_PY" -u goemotions_adaptation/run.py "$@" &
    LGSA_CHILD=$!
    wait "$LGSA_CHILD"
    LGSA_CHILD=''
}
date --iso-8601=seconds
hostname
nvidia-smi
"$LGSA_PY" -u goemotions_adaptation/preflight.py
"$LGSA_PY" -u goemotions_adaptation/upstream_evidence.py --job 2985243
if [[ -n "${LGSA_RESUME_RUN:-}" ]]; then
    # This path must be an unfinished full run; run.py checks checkpoint lineage.
    : "${LGSA_SMOKE_RUN:?Set original passed GoEmotions smoke directory for a resume}"
    "$LGSA_PY" -u goemotions_adaptation/accept_smoke.py "$LGSA_SMOKE_RUN"
    run_phase --phase full --smoke-run "$LGSA_SMOKE_RUN" --resume-run "$LGSA_RESUME_RUN"
else
    LGSA_SMOKE_RUN="$PWD/outputs/lgsa_goemotions/smoke/$SLURM_JOB_ID"
    run_phase --phase smoke
    "$LGSA_PY" -u goemotions_adaptation/accept_smoke.py "$LGSA_SMOKE_RUN"
    # Separate Python process; no model/optimizer object survives from smoke.
    run_phase --phase full --smoke-run "$LGSA_SMOKE_RUN"
fi
date --iso-8601=seconds
echo GOEMOTIONS_CHAIN_PASS
