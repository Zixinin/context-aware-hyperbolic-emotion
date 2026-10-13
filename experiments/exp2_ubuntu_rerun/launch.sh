#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
test "$(hostname)" = D3020-A3
test "$#" -eq 2
mode="$1"
test "$mode" = prepared -o "$mode" = resume
work="$(realpath -- "$2")"
case "$work" in "$repo_dir"/results/exp2_ubuntu/run_*) ;; *) exit 2 ;; esac
test -d "$work"
python_bin="$repo_dir/.venv-lgsa/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
send_notice() {
  "$python_bin" experiments/exp4b_multiseed_diagnostics/notify_retry.py --message "$1" --out "$work/$2" || true
}
on_failure() {
  stage_exit="$?"
  trap - ERR
  send_notice "❌ FYP Exp2 Ubuntu rerun stopped exit=$stage_exit; retained $work" notification_failure.json
  printf 'EXP2_UBUNTU_STOPPED exit=%s; retained %s\n' "$stage_exit" "$work"
  exit "$stage_exit"
}
trap on_failure ERR
printf 'EXP2_UBUNTU_WORK=%s\n' "$work"
test -f "$work/unit_test_report.json"
test -f "$work/preflight.json"
send_notice '🚀 FYP original Exp2 Ubuntu rerun started: serial42/123/2026, old tree, then same-host Exp4B comparison' notification_start.json
for seed in 42 123 2026; do
  export PYTHONHASHSEED="$seed"
  gates="$work/gates/seed$seed"
  mkdir -p "$gates"
  send_notice "🚀 FYP original Exp2 Ubuntu seed$seed audit/smoke/full" "notification_seed${seed}_start.json"
  for phase in audit smoke full; do
    stage="$gates/$phase"
    if [ "$phase" = full ]; then stage="$work/baselines/seed$seed"; fi
    if [ -f "$stage/SUCCESS.json" ]; then
      "$python_bin" "$script_dir/train_ubuntu.py" --seed "$seed" --verify-run "$stage"
      printf 'SKIP_COMPLETED seed=%s phase=%s\n' "$seed" "$phase"
      continue
    fi
    test ! -e "$stage"
    test ! -e "$gates/$phase.log"
    extra=()
    if [ "$phase" != audit ]; then extra+=(--audit-run "$gates/audit"); fi
    if [ "$phase" = full ]; then extra+=(--smoke-run "$gates/smoke"); fi
    "$python_bin" "$script_dir/train_ubuntu.py" --seed "$seed" --phase "$phase" --out "$stage" \
      --unit-proof "$work/unit_test_report.json" "${extra[@]}" 2>&1 | tee "$gates/$phase.log"
    "$python_bin" "$script_dir/train_ubuntu.py" --seed "$seed" --verify-run "$stage"
  done
  send_notice "✅ Original Exp2 Ubuntu seed$seed full training and evaluation verified" "notification_seed${seed}_success.json"
done
export PYTHONHASHSEED=42
if [ -f "$work/comparison/SUCCESS.json" ]; then
  "$python_bin" "$script_dir/train_ubuntu.py" --verify-run "$work/comparison"
else
  test ! -e "$work/comparison"
  "$python_bin" "$script_dir/compare_ubuntu.py" --work "$work" --out "$work/comparison" 2>&1 | tee "$work/comparison.log"
fi
"$python_bin" "$script_dir/train_ubuntu.py" --verify-run "$work/comparison"
send_notice "✅ EXP2_UBUNTU_RERUN_AND_EXP4B_COMPARISON_COMPLETE; $work/comparison" notification_success.json
printf 'EXP2_UBUNTU_RERUN_AND_EXP4B_COMPARISON_COMPLETE\nEXP2_UBUNTU_WORK=%s\n' "$work"
