#!/bin/bash
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
: "${LGSA_EMAIL:?Set LGSA_EMAIL to your real university email before submission}"
if [[ ! "$LGSA_EMAIL" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
    echo "Invalid email or unfilled placeholder; nothing submitted." >&2
    exit 1
fi
if [[ "${1:-}" != environment ]]; then
    echo "Only environment setup is enabled. Official/model smoke and full run remain gated." >&2
    exit 1
fi
export LGSA_EMAIL
mkdir -p logs
sbatch --mail-user="$LGSA_EMAIL" slurm/lgsa_environment.sbatch
