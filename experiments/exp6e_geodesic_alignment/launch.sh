#!/usr/bin/env bash
# User-operated Ubuntu pipeline: audit -> numerical smoke -> ONE fresh A1-G.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: use Ubuntu D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6e_geodesic_alignment" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify || test "${1:-}" = inspect; then
    test "$#" -eq 2 || exit 2
    CUDA_VISIBLE_DEVICES='' exec "$env_python" -X utf8 "$here/grun.py" "$1" --root "$2"
fi
test "$#" -eq 0 || exit 2
base="$repo/results/exp6e_geodesic_alignment"
test ! -L "$base" || exit 2
mkdir -p "$base"
for claim in "$base"/run_*/PLAN.json; do
    if test -f "$claim"; then echo "STOP: existing audit/pilot plan: $claim; inspect it, do not start another pilot"; exit 2; fi
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
        notify "Exp6E A1-G COMPLETE VERIFIED: one seed42 27-label geodesic-alignment pilot;4epochs;Validation only;NO NEW TEST. $root"
        echo 'GEODESIC_ALIGNMENT_SEED42_VALIDATION_PILOT_COMPLETE'
    elif ((rc==3)); then
        notify "Exp6E DIAGNOSTICS COMPLETE / SCALE REVIEW REQUIRED: no optimizer steps, formal training or new Test. Original weight .1 retained. $root"
        echo 'GEODESIC_NO_UPDATE_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED'
        cat "$root/SCALE_REVIEW_STOP.md"
        echo "TERMINAL_DIAGNOSTICS_COMMAND: bash $here/launch.sh inspect $root"
    else
        notify "Exp6E STOPPED phase=$phase exit=$rc. Preserve records; no automatic gamma/scale changes, retry, extra seed or Test. $root"
        echo 'GEODESIC_ALIGNMENT_STOPPED_REVIEW_REQUIRED'
        if test -f "$root/SCALE_REVIEW_STOP.md"; then cat "$root/SCALE_REVIEW_STOP.md"; fi
    fi
    echo "EXP6E_EXIT_CODE=$rc RUN_ROOT=$root"
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
exec 7>"$base/.gpu_phase.lock"
flock -n 7 || { echo 'STOP: geodesic pilot already running'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute active'; exit 2; }
printf '%s\n' "$root" > "$base/LATEST_EXP6E_RUN.txt"
notify "Exp6E started: original unscaled A1 control, replace alignment only; audit then severe numerical gates then ONE seed42 pilot. $root"
cd "$repo"
phase=phase1_audit
timeout --signal=TERM --kill-after=30s 90m "$env_python" -X utf8 -u "$here/grun.py" audit --root "$root"
phase=phase2_smoke_no_formal_training
timeout --signal=TERM --kill-after=30s 4h "$env_python" -X utf8 -u "$here/grun.py" smoke --root "$root"
notify "Exp6E audit and fresh/A1-best train-only loss/gradient gates PASS. Starting the authorized fresh seed42 A1-G pilot;gamma .1,scale1 unchanged. $root"
phase=phase3_one_seed42_training
timeout --signal=TERM --kill-after=30s 8h "$env_python" -X utf8 -u "$here/grun.py" train --root "$root"
phase=phase4_saved_result_verification
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/grun.py" verify --root "$root"
echo "TERMINAL_RESULTS_COMMAND: cat $root/GEODESIC_ALIGNMENT_SUMMARY_CN.md"
