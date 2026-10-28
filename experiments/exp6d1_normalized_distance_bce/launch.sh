#!/usr/bin/env bash
# User runs on Ubuntu. Smoke gate -> ONE fresh seed42 N1 pilot -> stop.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: expected Ubuntu D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6d1_normalized_distance_bce" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    export CUDA_VISIBLE_DEVICES=''
    exec "$env_python" -X utf8 "$here/nrun.py" verify --run-root "$2"
fi
test "$#" -eq 0 || exit 2
base="$repo/results/exp6d1_normalized_distance_bce"
test ! -L "$base" || exit 2
mkdir -p "$base"
root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
phase=preflight
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED (experiment exit code retained)'
    fi
    return 0
}
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$root/exit_code.txt"
    if ((rc==0)); then
        notify "Exp6D-1 N1 seed42 COMPLETE VERIFIED: 4epochs, Validation reload, NO NEW TEST. Results printed in terminal. $root"
        printf '%s\n' '==================================================' 'EXP6D1_N1_SEED42_VALIDATION_PILOT_COMPLETE' 'Task: RoBERTa27 Multi-label' 'Control: Exp6C-1S' 'Distance weighting: N1 normalized' 'Hyperbolic input scale: 0.03' 'Auxiliary coefficient: 0.1' 'Seed: 42' 'Checkpoint: Validation Macro-F1' 'Formal threshold: 0.30' 'New Test inference: NONE' '=================================================='
    else
        notify "Exp6D-1 STOPPED phase=$phase exit=$rc. No retry/new seed/Test. Paste terminal error. $root"
        echo 'EXP6D1_FAILED_STOPPED_NO_AUTOMATIC_RETRY'
    fi
    echo "EXP6D1_EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee "$root/run.log") 2>&1
echo "RUN_ROOT=$root"
notify "Exp6D-1 started: N1 smoke gate then ONE fresh seed42, 4epochs, s=.03/gamma=.1, Validation only. $root"
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: A1/GPU phase busy'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || { echo 'STOP: HypEmo phase busy'; exit 2; }
exec 7>"$base/.n1_phase.lock"
flock -n 7 || { echo 'STOP: another N1 phase active'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute already active'; exit 2; }
for claim in "$base"/run_*/FORMAL_TRAINING_CLAIM.json; do
    if test -f "$claim"; then echo "STOP: one N1 formal training attempt already exists: $claim"; exit 2; fi
done
printf '%s\n' "$root" > "$base/LATEST_EXP6D1_RUN.txt"
cd "$repo"
phase=smoke
if timeout --signal=TERM --kill-after=30s 90m "$env_python" -X utf8 -u "$here/nrun.py" smoke --run-root "$root"; then
    printf '%s\n' 0 > "$root/smoke_exit_code.txt"
else
    rc=$?;printf '%s\n' "$rc" > "$root/smoke_exit_code.txt";exit "$rc"
fi
notify "Exp6D-1 SMOKE PASS: original Exp6D loss/module gradient replay and fresh initialization passed. Starting authorized seed42 N1 only. $root"
phase=train
if timeout --signal=TERM --kill-after=30s 8h "$env_python" -X utf8 -u "$here/nrun.py" train --run-root "$root"; then
    printf '%s\n' 0 > "$root/train_exit_code.txt"
else
    rc=$?;printf '%s\n' "$rc" > "$root/train_exit_code.txt";exit "$rc"
fi
phase=final_verification
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/nrun.py" verify --run-root "$root"
echo "TERMINAL_RESULTS_COMMAND: cat $root/EXP6D1_TRAINING_SUMMARY.md"
