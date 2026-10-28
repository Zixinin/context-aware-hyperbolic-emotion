#!/usr/bin/env bash
# ONE real four-epoch calibrated A1-G-C pilot after preflight/smoke. No new Test.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: use Ubuntu D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6e2_calibrated_geodesic_train" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    CUDA_VISIBLE_DEVICES='' exec "$env_python" -X utf8 "$here/crun.py" verify --root "$2"
fi
test "$#" -eq 0 || exit 2
base="$repo/results/exp6e2_calibrated_geodesic_train"
test ! -L "$base" || exit 2
mkdir -p "$base"
for claim in "$base"/run_*/PLAN.json; do
    if test -f "$claim"; then echo "STOP: existing pilot plan: $claim; verify original run, no automatic retry"; exit 2; fi
done
root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
phase=preflight
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED; real task exit code retained'
    fi
    return 0
}
finish() {
    rc=$?;trap - EXIT
    printf '%s\n' "$rc" > "$root/exit_code.txt"
    if ((rc==0)); then
        notify "Exp6E-2 A1-G-C TRAINING COMPLETE VERIFIED: seed42, 4 epochs/7648 steps, gamma=0.02091538317504222 s=1; Validation reload passed; NO NEW TEST. $root"
        echo 'EXP6E2_A1GC_FOUR_EPOCH_TRAINING_COMPLETE'
    else
        notify "Exp6E-2 A1-G-C STOPPED phase=$phase exit=$rc. Preserve records; no gamma changes, retry, new seed or Test. $root"
        echo 'EXP6E2_A1GC_STOPPED_REVIEW_REQUIRED'
    fi
    echo "EXP6E2_EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee "$root/run.log") 2>&1
echo "RUN_ROOT=$root"
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: FYP GPU phase busy'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || { echo 'STOP: HypEmo GPU phase busy'; exit 2; }
exec 6<"$repo/results/exp6e_geodesic_alignment/.gpu_phase.lock"
flock -n 6 || { echo 'STOP: original Exp6E GPU phase busy'; exit 2; }
exec 7>"$base/.gpu_phase.lock"
flock -n 7 || { echo 'STOP: calibrated pilot already running'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute active'; exit 2; }
printf '%s\n' "$root" > "$base/LATEST_EXP6E2_RUN.txt"
notify "Exp6E-2 A1-G-C pipeline started: preflight/smoke then REAL seed42 FOUR-EPOCH training; fixed gamma=0.02091538317504222 s=1; Validation only. $root"
cd "$repo"
phase=prepare_frozen_inputs
timeout --signal=TERM --kill-after=30s 90m "$env_python" -X utf8 -u "$here/crun.py" prepare --root "$root"
phase=actual_fresh_replay_and_discarded_smoke
timeout --signal=TERM --kill-after=30s 4h "$env_python" -X utf8 -u "$here/crun.py" smoke --root "$root"
notify "Exp6E-2 A1-G-C smoke PASSED. Starting REAL fresh seed42 training now: 4epochs/7648 steps; gamma fixed; no Test. $root"
phase=FORMAL_FOUR_EPOCH_TRAINING
timeout --signal=TERM --kill-after=30s 10h "$env_python" -X utf8 -u "$here/crun.py" train --root "$root"
phase=saved_artifact_verification
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/crun.py" verify --root "$root"
echo "TERMINAL_RESULTS_COMMAND: cat $root/GEODESIC_ALIGNMENT_SUMMARY_CN.md"
