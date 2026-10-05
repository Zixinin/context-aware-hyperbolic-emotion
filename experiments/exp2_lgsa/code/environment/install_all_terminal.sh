#!/bin/bash
# Run after install_all_cpu.py has been placed in environment/ on HPC.
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_TIMEOUT=120 PIP_RETRIES=10 PIP_RESUME_RETRIES=20
export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=2
export MPLBACKEND=Agg
mkdir -p environment logs downloads/wheels
/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python -u \
    environment/install_all_cpu.py 2>&1 | tee "logs/install-env-$(date +%Y%m%d-%H%M%S).log"
