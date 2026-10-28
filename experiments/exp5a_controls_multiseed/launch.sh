#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python_bin="$repo_dir/.venv-bert/bin/python"
notify_script="$repo_dir/experiments/exp5_label_aware_euclidean/notify.py"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
cd "$repo_dir"
test "$(hostname)" = "D3020-A3"
test -x "$python_bin"
phase="${1:-}"

if [ "$phase" = "notify-test" ]; then
  test "$#" -eq 1
  "$python_bin" "$notify_script" --message "Exp5A CLS-Mean multi-seed notification test: new seeds123/2026" \
    --out /tmp/exp5a_multiseed_ntfy_test.json
  exit 0
fi

case "$phase" in
  audit)
    test "$#" -eq 1
    run_dir="$repo_dir/results/exp5a_controls_multiseed/run_$(date -u +%Y%m%dT%H%M%SZ)"
    test ! -e "$run_dir"
    mkdir -p "$run_dir"
    ;;
  chain|verify)
    test "$#" -eq 2
    run_dir="$(realpath -- "$2")"
    case "$run_dir" in
      "$repo_dir"/results/exp5a_controls_multiseed/run_*) ;;
      *) printf 'Unexpected results root: %s\n' "$run_dir"; exit 2 ;;
    esac
    test "$(dirname -- "$run_dir")" = "$repo_dir/results/exp5a_controls_multiseed"
    test -f "$run_dir/preflight.json"
    ;;
  *) printf 'Usage: bash launch.sh notify-test|audit\n       bash launch.sh chain|verify /absolute/run_root\n'; exit 2 ;;
esac

notify() {
  "$python_bin" "$notify_script" --message "$1" --out "$2" || true
}
on_failure() {
  experiment_exit="$?"
  trap - ERR
  notify "Exp5A CLS-Mean multi-seed $phase FAILED exit=$experiment_exit; retain $run_dir" "$run_dir/notification_failure_$phase.json"
  printf 'EXP5A_MULTISEED_FAILED exit=%s retained=%s\n' "$experiment_exit" "$run_dir"
  exit "$experiment_exit"
}
trap on_failure ERR
if [ "$phase" != "verify" ]; then
  notify "Exp5A CLS-Mean multi-seed $phase START; only new seeds123/2026; seed42 frozen" "$run_dir/notification_start_$phase.json"
fi

if [ "$phase" = "audit" ]; then
  export PYTHONHASHSEED=42
  "$python_bin" "$script_dir/multi_tests.py" --out "$run_dir/unit_test_report.json" 2>&1 | tee "$run_dir/unit_tests.log"
  "$python_bin" "$script_dir/multi_train.py" --help > "$run_dir/train_help.txt"
  "$python_bin" "$script_dir/multi_audit.py" --unit-proof "$run_dir/unit_test_report.json" \
    --out "$run_dir/preflight.json" 2>&1 | tee "$run_dir/preflight.log"
elif [ "$phase" = "verify" ]; then
  "$python_bin" "$script_dir/multi_compare.py" "$run_dir" --verify-only
else
  "$python_bin" "$script_dir/multi_audit.py" --unit-proof "$run_dir/unit_test_report.json" \
    --out "$run_dir/preflight.json" --verify-only 2>&1 | tee "$run_dir/recheck_chain.log"
  # ALL four smoke gates complete before ANY expensive full run.
  for mode in smoke full; do
    if [ "$mode" = "full" ]; then
      # Re-audit actual raw tensors/assets after smoke, before any formal model.
      "$python_bin" "$script_dir/multi_audit.py" --unit-proof "$run_dir/unit_test_report.json" \
        --out "$run_dir/preflight.json" --verify-only 2>&1 | tee "$run_dir/recheck_full.log"
    fi
    for seed in 123 2026; do
      export PYTHONHASHSEED="$seed"
      for representation in cls mean; do
        if [ "$mode" = "smoke" ]; then
          model_dir="$run_dir/smoke_${representation}_seed${seed}"
        else
          model_dir="$run_dir/${representation}_seed${seed}"
        fi
        if [ -e "$model_dir" ]; then
          "$python_bin" "$script_dir/multi_verify.py" "$model_dir"
          printf 'Reusing verified %s\n' "$model_dir"
        else
          notify "Exp5A $mode $representation seed$seed START" "$run_dir/notification_start_${mode}_${representation}_${seed}.json"
          "$python_bin" "$script_dir/multi_train.py" --phase "$mode" --representation "$representation" \
            --seed "$seed" --run-root "$run_dir" 2>&1 | tee "$run_dir/${mode}_${representation}_seed${seed}.log"
          "$python_bin" "$script_dir/multi_verify.py" "$model_dir"
          notify "Exp5A $mode $representation seed$seed VERIFIED" "$run_dir/notification_success_${mode}_${representation}_${seed}.json"
        fi
      done
    done
  done
  if [ -f "$run_dir/MULTISEED_SUCCESS.json" ]; then
    "$python_bin" "$script_dir/multi_compare.py" "$run_dir" --verify-only
  else
    "$python_bin" "$script_dir/multi_compare.py" "$run_dir" 2>&1 | tee "$run_dir/comparison.log"
  fi
fi

if [ "$phase" != "verify" ]; then
  notify "Exp5A CLS-Mean multi-seed $phase COMPLETE VERIFIED; inspect $run_dir" "$run_dir/notification_success_$phase.json"
fi
trap - ERR
printf 'EXP5A_MULTISEED_%s_COMPLETE_VERIFIED\nEXP5A_MULTISEED_RUN_DIR=%s\n' "${phase^^}" "$run_dir"
