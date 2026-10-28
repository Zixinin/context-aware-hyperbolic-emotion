#!/usr/bin/env bash
# Calibration only: no GPU, model execution or training phase exists here.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: run on Ubuntu D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6e1_gradient_calibration" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || { echo 'Existing .venv-bert required; no installation'; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    exec "$env_python" -X utf8 "$here/calibrate.py" verify --root "$2"
fi
test "$#" -eq 0 || exit 2
base="$repo/results/exp6e1_gradient_calibration"
test ! -L "$base" || exit 2
mkdir -p "$base"
exec 7>"$base/.calibration.lock"
flock -n 7 || { echo 'STOP: another calibration is active'; exit 2; }
for claim in "$base"/run_*/PLAN.json; do
    if test -f "$claim"; then echo "STOP: existing calibration: $claim; verify it, no automatic retry"; exit 2; fi
done
root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
phase=calibration_preflight
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED; real calibration exit code retained'
    fi
    return 0
}
finish() {
    rc=$?;trap - EXIT
    printf '%s\n' "$rc" > "$root/exit_code.txt"
    if ((rc==0)); then
        notify "Exp6E-1 CALIBRATION COMPLETE: one frozen gamma proposal from fresh train-only real gradients. NO TRAINING, NO VALIDATION/TEST inference. Return report for review. $root"
        echo 'EXP6E1_CALIBRATION_COMPLETE_NO_TRAINING'
    else
        notify "Exp6E-1 CALIBRATION STOPPED phase=$phase exit=$rc. Records preserved; no automatic retry or training. $root"
        echo 'EXP6E1_CALIBRATION_STOPPED'
    fi
    echo "EXP6E1_EXIT_CODE=$rc RUN_ROOT=$root"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee "$root/run.log") 2>&1
printf '%s\n' "$root" > "$base/LATEST_EXP6E1_RUN.txt"
echo "RUN_ROOT=$root"
notify "Exp6E-1 started: CPU read-only calibration of sealed fresh-init gradients; train-only 10 cohorts. No model, optimizer, training, Validation or Test inference. $root"
cd "$repo"
phase=readonly_fresh_gradient_calibration
timeout --signal=TERM --kill-after=15s 20m "$env_python" -X utf8 -u "$here/calibrate.py" run --root "$root"
phase=saved_artifact_verification
"$env_python" -X utf8 "$here/calibrate.py" verify --root "$root"
echo "TERMINAL_RESULTS_COMMAND: bash $here/launch.sh verify $root"
