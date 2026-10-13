#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
test "$(hostname)" = D3020-A3
python_bin="$repo_dir/.venv-lgsa/bin/python"
test -x "$python_bin"
mode="${1:-}"
test "$mode" = run -o "$mode" = prepared -o "$mode" = resume
test "$#" -eq 2
work="$(realpath -- "$2")"
case "$work" in "$repo_dir"/results/exp4b/multiseed_*) ;; *) exit 2 ;; esac
test -d "$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
printf 'EXP4B_MULTI_WORK=%s\n' "$work"
send_notice() {
  "$python_bin" "$script_dir/notify_retry.py" --message "$1" --out "$work/$2" || true
}
on_failure() {
  stage_exit="$?"
  trap - ERR
  send_notice "❌ FYP Exp4B multi-seed stopped, exit=$stage_exit; retained $work" notification_failure.json
  printf 'EXP4B_MULTI_STOPPED exit=%s; no outputs deleted: %s\n' "$stage_exit" "$work"
  exit "$stage_exit"
}
trap on_failure ERR
send_notice "🚀 FYP Exp4B multi-seed started: reuse42, serial123/2026, read-only diagnostics" notification_start.json
if [ "$mode" = run ]; then
  test ! -e "$work/unit_test_report.json"
  "$python_bin" "$repo_dir/experiments/exp4b_data_driven_hierarchy/unit_tests.py" --repo "$repo_dir" \
    --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
    --out "$work/unit_test_report.json" 2>&1 | tee "$work/pilot_unit_tests.log"
  "$python_bin" "$script_dir/unit_tests.py" --repo "$repo_dir" \
    --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
    --out "$work/extension_unit_report.json" 2>&1 | tee "$work/extension_unit_tests.log"
  "$python_bin" "$script_dir/preflight_extension.py" --work "$work" --out "$work/preflight.json" \
    2>&1 | tee "$work/preflight.log"
else
  test -f "$work/unit_test_report.json"
  test -f "$work/extension_unit_report.json"
  "$python_bin" "$script_dir/preflight_extension.py" --work "$work"
fi
for seed in 123 2026; do
  export PYTHONHASHSEED="$seed"
  seed_dir="$work/seed$seed"
  mkdir -p "$seed_dir"
  send_notice "🚀 FYP Exp4B seed$seed audit/smoke/full stages" "notification_seed${seed}_start.json"
  for phase in audit smoke full; do
    stage="$seed_dir/$phase"
    if [ -f "$stage/SUCCESS.json" ]; then
      "$python_bin" "$script_dir/seed_extension.py" --seed "$seed" \
        --extension-unit-proof "$work/extension_unit_report.json" --verify-run "$stage"
      printf 'SKIP_COMPLETED seed=%s phase=%s\n' "$seed" "$phase"
      continue
    fi
    test ! -e "$stage"
    test ! -e "$seed_dir/$phase.log"
    extra=()
    if [ "$phase" != audit ]; then extra+=(--audit-run "$seed_dir/audit"); fi
    if [ "$phase" = full ]; then extra+=(--smoke-run "$seed_dir/smoke"); fi
    "$python_bin" "$script_dir/seed_extension.py" --seed "$seed" --phase "$phase" --out "$stage" \
      --unit-proof "$work/unit_test_report.json" --extension-unit-proof "$work/extension_unit_report.json" \
      "${extra[@]}" 2>&1 | tee "$seed_dir/$phase.log"
    "$python_bin" "$script_dir/seed_extension.py" --seed "$seed" \
      --extension-unit-proof "$work/extension_unit_report.json" --verify-run "$stage"
  done
  send_notice "✅ FYP Exp4B seed$seed full training and paired comparison verified" "notification_seed${seed}_success.json"
done
export PYTHONHASHSEED=42
if [ -f "$work/diagnostics/SUCCESS.json" ]; then
  "$python_bin" "$repo_dir/experiments/exp4b_data_driven_hierarchy/train.py" --verify-run "$work/diagnostics"
else
  test ! -e "$work/diagnostics"
  test ! -e "$work/diagnostics.log"
  "$python_bin" "$script_dir/diagnostics.py" --work "$work" --out "$work/diagnostics" \
    2>&1 | tee "$work/diagnostics.log"
fi
"$python_bin" "$repo_dir/experiments/exp4b_data_driven_hierarchy/train.py" --verify-run "$work/diagnostics"
send_notice "✅ EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE; $work/diagnostics" notification_success.json
printf 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE\nEXP4B_MULTI_WORK=%s\n' "$work"
