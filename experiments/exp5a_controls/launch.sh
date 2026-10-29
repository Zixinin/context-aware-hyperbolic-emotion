#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="$repo_dir/.venv-bert/bin/python"
notify_script="$repo_dir/experiments/exp5_label_aware_euclidean/notify.py"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
cd "$repo_dir"
test "$(hostname)" = "D3020-A3"
test -x "$python_bin"
phase="${1:-}"

if [ "$phase" = "notify-test" ]; then
  test "$#" -eq 1
  "$python_bin" "$notify_script" --message "Exp5A-Control notification test from D3020-A3" \
    --out /tmp/exp5a_control_ntfy_test.json
  exit 0
fi

case "$phase" in
  audit)
    test "$#" -eq 1
    run_dir="$repo_dir/results/exp5a_controls/run_$(date -u +%Y%m%dT%H%M%SZ)"
    test ! -e "$run_dir"
    mkdir -p "$run_dir"
    ;;
  smoke|full|verify)
    test "$#" -eq 2
    run_dir="$(realpath -- "$2")"
    case "$run_dir" in
      "$repo_dir"/results/exp5a_controls/run_*) ;;
      *) printf 'Refusing unexpected result root: %s\n' "$run_dir"; exit 2 ;;
    esac
    test "$(dirname -- "$run_dir")" = "$repo_dir/results/exp5a_controls"
    test -f "$run_dir/preflight.json"
    ;;
  *) printf 'Usage: bash launch.sh notify-test|audit\n       bash launch.sh smoke|full|verify /absolute/run_root\n'; exit 2 ;;
esac

notify() {
  "$python_bin" "$notify_script" --message "$1" --out "$2" || true
}
on_failure() {
  experiment_exit="$?"
  trap - ERR
  notify "Exp5A-Control $phase failed exit=$experiment_exit; retained=$run_dir" "$run_dir/notification_failure_$phase.json"
  printf 'EXP5A_CONTROL_FAILED phase=%s exit=%s retained=%s\n' "$phase" "$experiment_exit" "$run_dir"
  exit "$experiment_exit"
}
trap on_failure ERR

if [ "$phase" != "verify" ]; then
  notify "Exp5A-Control $phase started: CLS vs mean, seed42 only; C frozen" "$run_dir/notification_start_$phase.json"
fi

if [ "$phase" = "audit" ]; then
  "$python_bin" "$script_dir/control_tests.py" --out "$run_dir/unit_test_report.json" 2>&1 | tee "$run_dir/unit_tests.log"
  "$python_bin" "$script_dir/control_train.py" --help > "$run_dir/train_help.txt"
  "$python_bin" "$script_dir/control_audit.py" --unit-proof "$run_dir/unit_test_report.json" \
    --out "$run_dir/preflight.json" 2>&1 | tee "$run_dir/preflight.log"
elif [ "$phase" = "verify" ]; then
  "$python_bin" "$script_dir/control_compare.py" "$run_dir" --verify-only
else
  if [ "$phase" = "full" ]; then
    # Validate BOTH smoke runs before any full training starts.
    "$python_bin" "$script_dir/control_verify.py" "$run_dir/smoke_cls"
    "$python_bin" "$script_dir/control_verify.py" "$run_dir/smoke_mean"
  fi
  # Fresh raw-weight/environment/C audit; compare without changing frozen proof.
  "$python_bin" "$script_dir/control_audit.py" --unit-proof "$run_dir/unit_test_report.json" \
    --out "$run_dir/preflight.json" --verify-only 2>&1 | tee "$run_dir/recheck_$phase.log"
  for representation in cls mean; do
    if [ "$phase" = "smoke" ]; then
      model_dir="$run_dir/smoke_$representation"
    else
      model_dir="$run_dir/${representation}_seed42"
    fi
    if [ -e "$model_dir" ]; then
      # Completed verified models may be skipped on rerun; partial runs are never overwritten.
      "$python_bin" "$script_dir/control_verify.py" "$model_dir"
      printf 'Reusing verified %s\n' "$model_dir"
    else
      notify "Exp5A-Control $phase $representation seed42 started" "$run_dir/notification_start_${phase}_${representation}.json"
      "$python_bin" "$script_dir/control_train.py" --phase "$phase" --representation "$representation" \
        --run-root "$run_dir" 2>&1 | tee "$run_dir/${phase}_${representation}.log"
      "$python_bin" "$script_dir/control_verify.py" "$model_dir"
      notify "Exp5A-Control $phase $representation seed42 verified" "$run_dir/notification_success_${phase}_${representation}.json"
    fi
  done
  if [ "$phase" = "full" ]; then
    if [ -f "$run_dir/CONTROL_SUCCESS.json" ]; then
      "$python_bin" "$script_dir/control_compare.py" "$run_dir" --verify-only
    else
      "$python_bin" "$script_dir/control_compare.py" "$run_dir" 2>&1 | tee "$run_dir/comparison.log"
    fi
  fi
fi

if [ "$phase" != "verify" ]; then
  notify "Exp5A-Control $phase complete verified; inspect $run_dir" "$run_dir/notification_success_$phase.json"
fi
trap - ERR
printf 'EXP5A_CONTROL_%s_COMPLETE_VERIFIED\nEXP5A_CONTROL_RUN_DIR=%s\n' "${phase^^}" "$run_dir"
