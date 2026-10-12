#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
test "$(hostname)" = D3020-A3
python_bin="$repo_dir/.venv-lgsa/bin/python"
test -x "$python_bin"
export PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
phase="${1:-smoke}"
if [ "$phase" = smoke ]; then
  run_dir="$repo_dir/results/exp4b/run_$(date -u +%Y%m%dT%H%M%SZ)"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
elif [ "$phase" = full ]; then
  test "$#" -eq 2
  run_dir="$(realpath -- "$2")"
  case "$run_dir" in "$repo_dir"/results/exp4b/run_*) ;; *) exit 2 ;; esac
  test -f "$run_dir/smoke/SUCCESS.json"
  test ! -e "$run_dir/seed42"
  test ! -e "$run_dir/full.log"
else
  printf 'Usage: bash launch.sh smoke | bash launch.sh full /absolute/verified/run_directory\n'
  exit 2
fi
printf 'EXP4B_RUN_DIR=%s\n' "$run_dir"
"$python_bin" "$script_dir/notify.py" --message "🚀 FYP Exp4B $phase started: full hierarchy, text50, final28 labels" --out "$run_dir/notification_start_$phase.json" || true
on_failure() {
  stage_exit="$?"
  trap - ERR
  "$python_bin" "$script_dir/notify.py" --message "❌ FYP Exp4B $phase failed, exit=$stage_exit; inspect $run_dir" --out "$run_dir/notification_failure_$phase.json" || true
  printf 'EXP4B_STAGE_FAILED exit=%s; retained: %s\n' "$stage_exit" "$run_dir"
  exit "$stage_exit"
}
trap on_failure ERR
if [ "$phase" = smoke ]; then
  "$python_bin" "$script_dir/unit_tests.py" --repo "$repo_dir" \
    --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
    --out "$run_dir/unit_test_report.json" 2>&1 | tee "$run_dir/unit_tests.log"
  "$python_bin" "$script_dir/train.py" --phase audit --out "$run_dir/audit" \
    --unit-proof "$run_dir/unit_test_report.json" 2>&1 | tee "$run_dir/audit.log"
  "$python_bin" "$script_dir/train.py" --phase smoke --out "$run_dir/smoke" \
    --unit-proof "$run_dir/unit_test_report.json" --audit-run "$run_dir/audit" \
    2>&1 | tee "$run_dir/smoke_test.log"
  "$python_bin" "$script_dir/train.py" --verify-run "$run_dir/smoke"
  marker=EXP4B_SMOKE_BATCH_COMPLETE_VERIFIED
else
  "$python_bin" "$script_dir/train.py" --phase full --out "$run_dir/seed42" \
    --unit-proof "$run_dir/unit_test_report.json" --audit-run "$run_dir/audit" \
    --smoke-run "$run_dir/smoke" 2>&1 | tee "$run_dir/full.log"
  "$python_bin" "$script_dir/train.py" --verify-run "$run_dir/seed42"
  marker=EXP4B_SEED42_PILOT_COMPLETE_VERIFIED
fi
"$python_bin" "$script_dir/notify.py" --message "✅ FYP $marker; $run_dir" --out "$run_dir/notification_success_$phase.json" || true
printf '%s\nEXP4B_RUN_DIR=%s\n' "$marker" "$run_dir"
