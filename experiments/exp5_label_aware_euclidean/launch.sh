#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="$repo_dir/.venv-bert/bin/python"

cd "$repo_dir"
test "$(hostname)" = "D3020-A3"
test -x "$python_bin"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

phase="${1:-}"

if [ "$phase" = "notify-test" ]; then
  "$python_bin" "$script_dir/notify.py" \
    --message "Exp5A notification test from D3020-A3" \
    --out "/tmp/exp5a_ntfy_test.json"
  exit 0
fi

if [ "$phase" = "audit" ]; then
  test "$#" -eq 1
  run_dir="$repo_dir/results/exp5a/run_$(date -u +%Y%m%dT%H%M%SZ)"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
  "$python_bin" "$script_dir/unit_tests.py" --out "$run_dir/unit_test_report.json" \
    2>&1 | tee "$run_dir/unit_tests.log"
  "$python_bin" "$script_dir/preflight.py" \
    --unit-proof "$run_dir/unit_test_report.json" \
    --out "$run_dir/preflight.json" \
    2>&1 | tee "$run_dir/preflight.log"
  printf 'EXP5A_AUDIT_COMPLETE_VERIFIED\nEXP5A_RUN_DIR=%s\n' "$run_dir"
  exit 0
fi

if [ "$phase" != "smoke" ] && [ "$phase" != "full" ]; then
  printf 'Usage:\n  bash launch.sh notify-test\n  bash launch.sh audit\n  bash launch.sh smoke /absolute/run_dir\n  bash launch.sh full /absolute/run_dir\n'
  exit 2
fi

test "$#" -eq 2
run_dir="$(realpath -- "$2")"
case "$run_dir" in
  "$repo_dir"/results/exp5a/run_*) ;;
  *) printf 'Refusing unexpected run directory: %s\n' "$run_dir"; exit 2 ;;
esac
test -f "$run_dir/unit_test_report.json"
test -f "$run_dir/preflight.json"

if [ "$phase" = "smoke" ]; then
  output_dir="$run_dir/smoke"
  test ! -e "$output_dir"
else
  output_dir="$run_dir/seed42"
  test -f "$run_dir/smoke/SUCCESS.json"
  test ! -e "$output_dir"
fi

"$python_bin" "$script_dir/notify.py" \
  --message "Exp5A $phase started on D3020-A3: seed42 label-aware Euclidean baseline" \
  --out "$run_dir/notification_start_$phase.json" || true

on_failure() {
  experiment_exit="$?"
  trap - ERR
  "$python_bin" "$script_dir/notify.py" \
    --message "Exp5A $phase failed, exit=$experiment_exit; inspect $run_dir" \
    --out "$run_dir/notification_failure_$phase.json" || true
  printf 'EXP5A_%s_FAILED exit=%s; retained=%s\n' "$phase" "$experiment_exit" "$run_dir"
  exit "$experiment_exit"
}
trap on_failure ERR

if [ "$phase" = "smoke" ]; then
  "$python_bin" "$script_dir/train.py" \
    --phase smoke --out "$output_dir" --preflight "$run_dir/preflight.json" \
    2>&1 | tee "$run_dir/smoke.log"
else
  "$python_bin" "$script_dir/train.py" \
    --phase full --out "$output_dir" --preflight "$run_dir/preflight.json" \
    --smoke-run "$run_dir/smoke" \
    2>&1 | tee "$run_dir/full.log"
fi

"$python_bin" "$script_dir/verify.py" "$output_dir"

if [ "$phase" = "full" ]; then
  metric_text="$($python_bin - "$output_dir/test_fixed_0.30_metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"Micro-F1={m['micro_f1']:.4f}, Macro-F1={m['macro_f1']:.4f}")
PY
)"
else
  metric_text="engineering smoke only; not a scientific result"
fi

"$python_bin" "$script_dir/notify.py" \
  --message "Exp5A $phase completed: $metric_text; inspect $run_dir" \
  --out "$run_dir/notification_success_$phase.json" || true

trap - ERR
printf 'EXP5A_%s_COMPLETE_VERIFIED\nEXP5A_RUN_DIR=%s\n' "${phase^^}" "$run_dir"
