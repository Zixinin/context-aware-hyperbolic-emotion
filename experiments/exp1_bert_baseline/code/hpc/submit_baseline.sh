#!/bin/bash -l
set -euo pipefail
if [[ $# != 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash hpc/submit_baseline.sh SMOKE_JOB_ID" >&2
    exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/work/aac/zixinfeng23/fyp/goemotions}"
cd "$PROJECT_ROOT"
source hpc/activate_env.sh
export SMOKE_RUN="$PROJECT_ROOT/outputs/smoke/seed42/job$1"
"$PYTHON" -m scripts.verify_run "$SMOKE_RUN" --gate
if [[ -e outputs/bert_baseline/seed42 ]]; then
    echo "outputs/bert_baseline/seed42 already exists. Inspect/preserve that run before resubmission." >&2
    exit 1
fi
mkdir -p logs
sbatch --export=ALL slurm/baseline_bert_seed42.sbatch
