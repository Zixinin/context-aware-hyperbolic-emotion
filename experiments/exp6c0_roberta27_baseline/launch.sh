#!/usr/bin/env bash
# Each phase is explicit. Smoke never auto-starts formal training.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: expected D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || { echo 'STOP: expected zixinfeng'; exit 2; }
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6c0_roberta27_baseline" || { echo 'STOP: wrong package path'; exit 2; }
phase=${1:?Use smoke or train or verify}
shift
case "$phase" in smoke|train|verify) ;; *) echo 'Invalid phase'; exit 2 ;; esac
env_python=${EXP6C0_PYTHON:-"$repo/.venv-bert/bin/python"}
test -x "$env_python" || { echo 'STOP: existing environment missing; return inspection output'; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME

if test "$phase" = verify; then
    root=${1:?Provide the exact run directory}
    test "$#" -eq 1 || exit 2
    export CUDA_VISIBLE_DEVICES=''
    exec "$env_python" "$here/run.py" verify --run-root "$root"
fi

base="$repo/results/exp6c0_roberta27_baseline"
test ! -L "$base" || { echo 'STOP: result root is symlink'; exit 2; }
mkdir -p "$base"
exec 9>"$base/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: another Exp6C-0 phase is running'; exit 2; }
# Reuse the established upstream read lock if present; never create/modify it.
legacy_lock=/home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
if test -f "$legacy_lock"; then
    exec 8<"$legacy_lock"
    flock -n 8 || { echo 'STOP: another HypEmo phase is running'; exit 2; }
fi
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute is already active; wait for it to finish'; exit 2; }
if test "$phase" = smoke; then
    root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
    log="$root/smoke.log"
else
    root=${1:?Provide the smoke-verified run directory}
    shift
    test "$#" -eq 0 || exit 2
    test "$(dirname -- "$(realpath "$root")")" = "$base" || { echo 'STOP: wrong run root'; exit 2; }
    test -f "$root/smoke/SMOKE_VERIFIED.json" || { echo 'STOP: smoke proof missing'; exit 2; }
    test ! -e "$root/FORMAL_TRAINING_CLAIM.json" || { echo 'STOP: formal run already attempted; no automatic rerun'; exit 2; }
    test ! -e "$root/SUCCESS.json" || exit 2
    log="$root/train.log"
fi
test ! -e "$log" || { echo 'STOP: refusing to overwrite phase log'; exit 2; }
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"

notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 15 \
        --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED (original experiment exit code retained)'
    fi
    return 0
}
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$root/${phase}_exit_code.txt"
    if (( rc == 0 )); then
        if test "$phase" = train; then
            notify "Exp6C-0 seed42 COMPLETE VERIFIED; 4 epochs, full validation reload, one formal test; results=$root. No extra seeds or A1/A2."
        else
            notify "Exp6C-0 GPU smoke VERIFIED; formal training not started. Return smoke report. results=$root"
        fi
    else
        notify "Exp6C-0 $phase BLOCKED exit=$rc; results=$root. No automatic retry. Return phase log and FAILURE.json."
    fi
    echo "PHASE=$phase EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$log") 2>&1
echo "RUN_ROOT=$root"
notify "Exp6C-0 $phase started; RoBERTa27 BCE seed42; results=$root"
cd "$repo"
if test "$phase" = smoke; then
    "$env_python" "$here/run.py" smoke --run-root "$root" "$@"
else
    "$env_python" "$here/run.py" train --run-root "$root"
fi
