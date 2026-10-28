#!/usr/bin/env bash
# Six authorized fresh cells, serial. No seed42 retraining or new Test inference.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || exit 2
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6d2_paired_multiseed_v1_1" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
base="$repo/results/exp6d2_paired_multiseed"
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    CUDA_VISIBLE_DEVICES='' exec "$env_python" -X utf8 "$here/psuite.py" verify --suite "$2"
fi
test ! -L "$base" || exit 2
mkdir -p "$base"
if test "${1:-}" = resume; then
    test "$#" -eq 2 || exit 2
    root="$2"
    test -f "$root/SUITE_PLAN.json" && test ! -f "$root/SUCCESS.json" || exit 2
    need_prepare=0
else
    test "$#" -eq 0 || exit 2
    for claim in "$base"/run_*/SUITE_PLAN.json; do
        if test -f "$claim"; then echo "STOP: suite already exists: $claim; inspect/verify or explicitly resume it"; exit 2; fi
    done
    root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
    need_prepare=1
fi
phase=preflight
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED (real suite exit code retained)'
    fi
    return 0
}
finish() {
    rc=$?;trap - EXIT
    printf '%s\n' "$rc" > "$root/exit_code.txt"
    if ((rc==0)); then
        notify "Exp6D-2 COMPLETE VERIFIED: 6 new cells +3 frozen seed42; paired42/123/2026; Validation only; NO NEW TEST. $root"
        printf '%s\n' 'EXP6D2_PAIRED_THREE_SEED_VALIDATION_COMPLETE' 'Models: A0 / A1-S / N1' 'Seeds: 42 reused, 123 / 2026 newly trained' 'New independent training runs: 6' 'Threshold: 0.30; epochs: 4; selection: Validation Macro-F1' 'New Test inference: NONE; Euclidean control: NOT STARTED'
    else
        notify "Exp6D-2 STOPPED phase=$phase exit=$rc. Completed cells preserved. No automatic retry/new seed/Test. $root"
        echo 'EXP6D2_STOPPED_INCOMPLETE_NO_AUTOMATIC_RETRY'
    fi
    echo "EXP6D2_EXIT_CODE=$rc SUITE_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$root/suite.log") 2>&1
echo "SUITE_ROOT=$root"
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: FYP/GPU phase busy'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || { echo 'STOP: HypEmo phase busy'; exit 2; }
exec 7>"$base/.suite_phase.lock"
flock -n 7 || { echo 'STOP: another D2 suite is active'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute already active'; exit 2; }
printf '%s\n' "$root" > "$base/LATEST_EXP6D2_SUITE.txt"
notify "Exp6D-2 v1.1 started/resumed: preserve failed v1 prepare; six serial A0/A1S/N1 cells at123/2026; reuse frozen42; no new Test. $root"
cd "$repo"
if ((need_prepare==1)); then
    phase=prepare
    timeout --signal=TERM --kill-after=30s 90m "$env_python" -X utf8 -u "$here/psuite.py" prepare --suite "$root"
fi
if ! test -f "$root/SMOKE_SUCCESS.json"; then
    phase=all_six_smokes
    for seed in 123 2026; do
        export PYTHONHASHSEED="$seed"
        timeout --signal=TERM --kill-after=30s 3h "$env_python" -X utf8 -u "$here/psuite.py" smoke --suite "$root" --seed "$seed"
    done
fi
notify "Exp6D-2 all six model/seed smokes passed; paired initialization and data-order gates passed. Beginning/continuing six authorized serial cells. $root"
for seed in 123 2026; do
    export PYTHONHASHSEED="$seed"
    for variant in A0 A1S N1; do
        phase="seed${seed}_${variant}"
        notify "Exp6D-2 cell $phase starting/validating completed seal. Fresh pretrained if unstarted; no checkpoint continuation. $root"
        if timeout --signal=TERM --kill-after=30s 8h "$env_python" -X utf8 -u "$here/psuite.py" train-one --suite "$root" --seed "$seed" --variant "$variant"; then
            printf '%s\n' 0 > "$root/${phase}_exit_code.txt"
        else
            rc=$?;printf '%s\n' "$rc" > "$root/${phase}_exit_code.txt";exit "$rc"
        fi
        notify "Exp6D-2 cell $phase COMPLETE VERIFIED. Validation reload passed; no new Test. $root"
    done
done
phase=paired_summary
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 -u "$here/psuite.py" summarize --suite "$root"
phase=final_verification
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/psuite.py" verify --suite "$root"
echo "TERMINAL_RESULTS_COMMAND: cat $root/EXP6D2_SUMMARY_CN.md"
