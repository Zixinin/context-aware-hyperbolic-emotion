#!/usr/bin/env bash
# Explicit smoke / train / verify; no SSH, dependency installation, or A0 writes.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: expected D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6c1_roberta27_hypaux" || exit 2
phase=${1:?Use smoke or train or verify}
shift
case "$phase" in smoke|train|verify) ;; *) exit 2 ;; esac
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || { echo 'STOP: frozen A0 Python missing'; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME

if test "$phase" = verify; then
    test "$#" -eq 1 || exit 2
    export CUDA_VISIBLE_DEVICES=''
    exec "$env_python" "$here/a1run.py" verify --run-root "$1"
fi
base="$repo/results/exp6c1_roberta27_hypaux"
test ! -L "$base" || exit 2
mkdir -p "$base"
exec 9>"$base/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: A1 phase is active'; exit 2; }
legacy_lock=/home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
test -f "$legacy_lock" || { echo 'STOP: expected established experiment lock missing'; exit 2; }
exec 8<"$legacy_lock"
flock -n 8 || { echo 'STOP: another HypEmo phase is active'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute already active'; exit 2; }
if test "$phase" = smoke; then
    test "$#" -eq 0 || exit 2
    root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
else
    test "$#" -eq 1 || exit 2
    root=$(realpath "$1")
    test "$(dirname "$root")" = "$base" || exit 2
    test -f "$root/smoke/SMOKE_VERIFIED.json" || exit 2
    for name in FAILURE.json FORMAL_TRAINING_CLAIM.json FORMAL_TEST_CLAIM.json SUCCESS.json train.log; do
        test ! -e "$root/$name" || { echo "STOP: prior attempt or failure: $name"; exit 2; }
    done
fi
log="$root/${phase}.log"
test ! -e "$log" || exit 2
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
        --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED (experiment exit retained)'
    fi
    return 0
}
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$root/${phase}_exit_code.txt"
    if (( rc != 0 )); then
        notify "Exp6C-1 $phase BLOCKED exit=$rc results=$root. No retry or extra seed. Return FAILURE.json and log."
    elif test "$phase" = smoke; then
        notify "Exp6C-1 actual GPU smoke VERIFIED. A0 frozen; formal seed42 not started. Review actual loss ratios/gradients. results=$root"
    else
        notify "Exp6C-1 seed42 COMPLETE VERIFIED; 4epochs, full validation reload, one Test and matched A0 comparison. results=$root. Experiment stopped."
    fi
    echo "PHASE=$phase EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$log") 2>&1
echo "RUN_ROOT=$root"
notify "Exp6C-1 $phase started; RoBERTa27 ordinary BCE + 0.1 FormulaA auxiliary. A0 read-only. results=$root"
cd "$repo"
"$env_python" "$here/a1run.py" "$phase" --run-root "$root"
