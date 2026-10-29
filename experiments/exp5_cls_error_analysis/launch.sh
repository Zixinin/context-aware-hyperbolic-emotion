#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="$repo_dir/.venv-bert/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
cd "$repo_dir"
test "$(hostname)" = "D3020-A3"
test -x "$python_bin"
phase="${1:-}"
test "$#" -eq 1
notify_script="$repo_dir/experiments/exp5_label_aware_euclidean/notify.py"
notify() {
  "$python_bin" "$notify_script" --message "$1" --out "$2" || true
}
if [ "$phase" = "notify-test" ]; then
  "$python_bin" "$notify_script" --message "Exp5 CLS CPU-only error-analysis notification test" \
    --out /tmp/exp5_cls_error_analysis_ntfy_test.json
  "$python_bin" -c 'import json; p="/tmp/exp5_cls_error_analysis_ntfy_test.json"; assert json.load(open(p))["delivered"], "Notification test was not delivered"'
  exit 0
fi
if [ "$phase" != "analyze" ]; then
  printf 'Usage: bash launch.sh notify-test|analyze\n'
  exit 2
fi
run_dir="$repo_dir/results/exp5_cls_error_analysis/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test ! -e "$run_dir"
mkdir -p "$run_dir"
on_failure() {
  diagnosis_exit="$?"
  trap - ERR
  notify "Exp5 CLS error analysis FAILED exit=$diagnosis_exit; originals retained; $run_dir" "$run_dir/notification_failure.json"
  printf 'EXP5_CLS_ERROR_ANALYSIS_FAILED exit=%s\nEXP5_CLS_ANALYSIS_DIR=%s\n' "$diagnosis_exit" "$run_dir"
  exit "$diagnosis_exit"
}
trap on_failure ERR
notify "Exp5 CLS CPU-only error analysis START; no model load/training/tuning" "$run_dir/notification_start.json"
(
  cd "$script_dir"
  sha256sum -c PACKAGE_MANIFEST.sha256
) 2>&1 | tee "$run_dir/package_check.log"
"$python_bin" "$script_dir/test_analysis.py" 2>&1 | tee "$run_dir/unit_tests.log"
# analyze.py requires its output directly under the independent diagnostics root.
# Keep launch logs in the separate run_* wrapper. Never create files in old runs.
analysis_dir="$repo_dir/results/exp5_cls_error_analysis/run_$(date -u +%Y%m%dT%H%M%SZ)_$$_analysis"
"$python_bin" "$script_dir/analyze.py" --repo "$repo_dir" --out "$analysis_dir" \
  2>&1 | tee "$run_dir/analysis.log"
results_archive="$run_dir/EXP5_CLS_ERROR_ANALYSIS_results.tar.gz"
test ! -e "$results_archive"
tar -czf "$results_archive" -C "$analysis_dir" .
notify "Exp5 CLS error analysis COMPLETE VERIFIED; $analysis_dir" "$run_dir/notification_success.json"
trap - ERR
cat "$analysis_dir/EXP5_CLS_ERROR_ANALYSIS.md"
printf '\nEXP5_CLS_ERROR_ANALYSIS_LAUNCH_COMPLETE\nEXP5_CLS_ANALYSIS_DIR=%s\nEXP5_CLS_LOG_DIR=%s\nEXP5_CLS_RESULTS_ARCHIVE=%s\n' "$analysis_dir" "$run_dir" "$results_archive"
