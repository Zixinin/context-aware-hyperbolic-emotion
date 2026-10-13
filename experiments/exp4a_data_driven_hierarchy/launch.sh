#!/usr/bin/env bash
# bash, not zsh: PIPESTATUS captures the Python process status.
set -uo pipefail
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion || exit 1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/logs/exp4a_matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
log="logs/exp4a_data_driven_hierarchy.log"
if [[ -e "$log" ]]; then
    printf 'Refusing to overwrite existing log: %s\n' "$log"
    exit 1
fi
.venv-analysis/bin/python -u experiments/exp4a_data_driven_hierarchy/run_exp4a.py 2>&1 | tee "$log"
pipeline_codes=("${PIPESTATUS[@]}")
python_code="${pipeline_codes[0]}"
tee_code="${pipeline_codes[1]}"
printf 'EXP4A_PYTHON_EXIT_CODE=%s\nEXP4A_TEE_EXIT_CODE=%s\n' "$python_code" "$tee_code" | tee -a "$log"
if [[ "$python_code" -ne 0 ]]; then
    exit "$python_code"
fi
if [[ "$tee_code" -ne 0 ]]; then
    curl --fail --silent --show-error --max-time 10 -d "❌ FYP Exp4A log capture failed; Python exit=0, tee exit=$tee_code" https://ntfy.sh/zixin-fyp-4090-1119 || true
    exit "$tee_code"
fi
exit 0
