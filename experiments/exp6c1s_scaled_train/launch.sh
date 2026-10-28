#!/usr/bin/env bash
# The user runs this on Ubuntu. Automatic smoke gate, then only s=.03 seed42.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: expected D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6c1s_scaled_train" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    export CUDA_VISIBLE_DEVICES=''
    exec "$env_python" -X utf8 "$here/srun.py" verify --run-root "$2"
fi
test "$#" -eq 0 || { echo 'Use launch.sh, or launch.sh verify RUN_ROOT'; exit 2; }
base="$repo/results/exp6c1s_scaled_train"
test ! -L "$base" || exit 2
mkdir -p "$base"
root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
phase=preflight
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 15 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED (pilot exit code retained)'
    fi
    return 0
}
download_command() {
    echo "MAC_DOWNLOAD_COMMAND: scp ubuntu4090:$root/review/$1 /Users/fengzixin/Desktop/PSP303-FYP/交付/Exp6C1S_Scaled_Train_20261022/"
}
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$root/exit_code.txt"
    if (( rc != 0 )); then
        printf '%s\n' "$rc" > "$root/${phase}_exit_code.txt"
        if CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/export_review.py" --root "$root" --failed; then
            download_command Exp6C1S_Train_failure_review.tar.gz
        fi
        notify "Exp6C-1S-Train STOPPED phase=$phase exit=$rc. No retry/new seed/Test. Return failure bundle. results=$root"
    else
        notify "Exp6C-1S-Train seed42 COMPLETE VERIFIED: s=.03 gamma=.1 4epochs Validation reload; NO NEW TEST. Pilot stopped. results=$root"
        printf '%s\n' '=================================================' 'EXP6C1S_SCALED_SEED42_PILOT_COMPLETE' 'Backbone: roberta-base' 'Task: 27-label multi-label' 'Hyperbolic input scale: 0.03' 'Auxiliary coefficient: 0.1' 'Checkpoint: Validation Macro-F1' 'Formal threshold: 0.30' 'New Test inference: NONE' '================================================='
    fi
    echo "EXP6C1S_TRAIN_EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee "$root/run.log") 2>&1
echo "RUN_ROOT=$root"
notify "Exp6C-1S-Train started: smoke gate then s=.03 seed42 only, 4epochs, Validation only, no Test. results=$root"
# Read-only descriptors preserve old lock files; hold throughout the pilot.
test -f "$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock" || exit 2
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: A1/GPU phase active'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || { echo 'STOP: HypEmo phase active'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute already active'; exit 2; }
for claim in "$base"/run_*/FORMAL_TRAINING_CLAIM.json; do
    test ! -e "$claim" || { echo "STOP: an earlier seed42 formal attempt exists: $claim. Review it; no second run."; exit 2; }
done
if test -n "${EXP6C1S_DELIVERY_ARCHIVE:-}"; then cp "$EXP6C1S_DELIVERY_ARCHIVE" "$root/package.tar.gz"; fi
cd "$repo"
phase=smoke
timeout --signal=TERM --kill-after=30s 60m "$env_python" -X utf8 "$here/srun.py" smoke --run-root "$root" 2>&1 | tee "$root/smoke.log"
printf '0\n' > "$root/smoke_exit_code.txt"
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/srun.py" verify-smoke --run-root "$root"
notify "Exp6C-1S-Train smoke PASSED; starting the authorized fresh seed42 s=.03 four-epoch Validation-only pilot. results=$root"
phase=train
timeout --signal=TERM --kill-after=30s 8h "$env_python" -X utf8 "$here/srun.py" train --run-root "$root" 2>&1 | tee "$root/train.log"
printf '0\n' > "$root/train_exit_code.txt"
phase=verify
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/srun.py" verify --run-root "$root" 2>&1 | tee "$root/verify.log"
printf '0\n' > "$root/verify_exit_code.txt"
printf '0\n' > "$root/exit_code.txt"
phase=export
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/export_review.py" --root "$root"
download_command Exp6C1S_Train_review.tar.gz
