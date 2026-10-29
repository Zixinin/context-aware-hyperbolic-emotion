#!/usr/bin/env bash
# User-authorized: ONE coordinate pilot now, then complete verified export for Mac.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || { echo 'STOP: run on Ubuntu D3020-A3'; exit 2; }
test "$(id -un)" = zixinfeng || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$here" = "$repo/experiments/exp6e3_mac_transfer_v1_1" || exit 2
env_python="$repo/.venv-bert/bin/python"
test -x "$env_python" || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 GIT_OPTIONAL_LOCKS=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
if test "${1:-}" = verify; then
    test "$#" -eq 2 || exit 2
    CUDA_VISIBLE_DEVICES='' exec "$env_python" -X utf8 "$here/mrun.py" verify --root "$2"
fi
test "$#" -eq 0 || exit 2
base="$repo/results/exp6e3_matched_gamma_coordinate"
test ! -L "$base" || exit 2
mkdir -p "$base"
for claim in "$base"/run_*/PLAN.json; do
    if test -f "$claim"; then echo "STOP: existing experiment: $claim; verify it; no automatic retraining"; exit 2; fi
done
root=$(mktemp -d "$base/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
records_root=/home/zixinfeng/Documents/FYP_migration_records
mkdir -p "$records_root"
recordbase=$(mktemp -d "$records_root/session_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")
session=$(basename "$recordbase")
pre="$records_root/${session}_pre"
post="$records_root/${session}_post"
phase=preflight
science_complete=0
export_complete=0
exec 3>&1
notify() {
    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null; then
        echo 'NOTIFICATION_DELIVERY_FAILED; actual task exit code retained'
    fi
    return 0
}
finish() {
    rc=$?;trap - EXIT
    printf '%s\n' "$rc" > "$recordbase/pipeline_exit_code.txt"
    if test ! -f "$root/exit_code.txt"; then printf '%s\n' "$rc" > "$root/exit_code.txt"; fi
    if ((rc==0)); then
        notify "Exp6E-3 COMPLETE: seed42 4epochs verified, coordinate gamma=0.02091538317504222; result export ready. External disk NOT ATTEMPTED; Mac transfer PENDING; NO TEST / NO GIT PUSH. $root"
    else
        notify "Exp6E-3 STOPPED phase=$phase exit=$rc science_complete=$science_complete result_export_complete=$export_complete. Preserve all artifacts. No retry/new seed/Test/push. $recordbase"
        echo "STOPPED_PHASE=$phase SCIENCE_COMPLETE=$science_complete RESULT_EXPORT_COMPLETE=$export_complete" >&3
        if test "$phase" = mandatory_prebackup; then
            tail -n 60 "$recordbase/pre_console.log" >&3
            findmnt --raw --output TARGET,SOURCE,FSTYPE,OPTIONS >&3 || true
        fi
    fi
    echo "EXP6E3_PIPELINE_EXIT_CODE=$rc RUN_ROOT=$root SESSION_RECORDS=$recordbase" >&3
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkfifo "$recordbase/science_console.pipe"
tee "$root/run.log" < "$recordbase/science_console.pipe" >&3 &
tee_pid=$!
exec > "$recordbase/science_console.pipe" 2>&1
echo "RUN_ROOT=$root SESSION_RECORDS=$recordbase"
export XDG_CACHE_HOME="$root/runtime_cache" HF_HOME="$root/runtime_cache/huggingface"
export TORCH_HOME="$root/runtime_cache/torch" CUDA_CACHE_PATH="$root/runtime_cache/cuda"
mkdir -p "$CUDA_CACHE_PATH"
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: FYP GPU phase busy'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || { echo 'STOP: HypEmo GPU phase busy'; exit 2; }
exec 7>"$base/.gpu_phase.lock"
flock -n 7 || { echo 'STOP: coordinate pilot already running'; exit 2; }
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$compute_pids" || { echo 'STOP: GPU compute active'; exit 2; }
printf '%s\n' "$root" > "$base/LATEST_EXP6E3_RUN.txt"
printf '%s\n' "$recordbase" > "$base/LATEST_EXP6E3_SESSION.txt"
notify "Exp6E-3 started under NEW human authorization: no external disk; real seed42 four-epoch training after parity/smoke, then Mac result export. Mac backup PENDING. No Test or Git push. $root"
cd "$repo"
phase=record_user_authorized_transfer_policy
"$env_python" -X utf8 - "$root" "$here" <<'PY_POLICY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from transfer_policy import POLICY,check_policy
root=Path(sys.argv[1])
(root/'USER_AUTHORIZED_TRANSFER_POLICY.json').write_text(json.dumps(POLICY,indent=2)+'\n',encoding='utf-8')
print(json.dumps(check_policy(root),indent=2))
PY_POLICY
phase=prepare_frozen_inputs
timeout --signal=TERM --kill-after=30s 90m "$env_python" -X utf8 -u "$here/mrun.py" prepare --root "$root"
phase=actual_forward_gradient_parity_and_smoke
timeout --signal=TERM --kill-after=30s 4h "$env_python" -X utf8 -u "$here/mrun.py" smoke --root "$root"
notify "Exp6E-3 parity/smoke PASS. Starting REAL seed42 coordinate training: gamma fixed,4epochs/7648steps, no Test. $root"
phase=formal_four_epoch_training
timeout --signal=TERM --kill-after=30s 10h "$env_python" -X utf8 -u "$here/mrun.py" train --root "$root"
phase=science_saved_verification
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 "$here/mrun.py" verify --root "$root"
science_complete=1
printf '0\n' > "$root/exit_code.txt"
echo 'EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE'
echo "Science verified. Sealing source log, then exporting complete new results for Mac. SESSION=$recordbase"
# Quiescent source: do not write any result file after this point.
exec > "$recordbase/export_control.log" 2>&1
wait "$tee_pid"
phase=mac_result_archive_export
notify "Exp6E-3 REAL TRAINING AND VALIDATION COMPLETE. Building full checkpoint/results download package; Mac transfer still PENDING. $root"
timeout --signal=TERM --kill-after=30s 4h "$env_python" -X utf8 -u "$here/export_results.py" --root "$root" | tee "$recordbase/export_console.log" >&3
export_complete=1
phase=final_status
CUDA_VISIBLE_DEVICES='' "$env_python" -X utf8 - "$root" <<'PY_STATUS' >&3
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads((root/'validation_metrics.json').read_text())
print('================================================\nEXP6E3 FINAL STATUS\n')
print('Model=A1-C-Gamma Seed=42 Gamma=0.02091538317504222 Coordinate Dot Product; New Test=NONE')
for key in ['micro_f1','macro_f1','rare_f1','macro_average_precision']:print(key+'='+format(100*m[key],'.4f')+'%')
print('Science: PASS; source and checkpoint reload verified')
print('External drive: NOT_ATTEMPTED_USER_AUTHORIZED_DEFER')
print('Mac local: PENDING_USER_TRANSFER_AND_TARGET_SHA256_VERIFICATION')
print('GitHub: NOT_PUSHED; historical backup completeness: UNVERIFIED_IN_CURRENT_SESSION')
print('Source: '+str(root))
print('================================================')
PY_STATUS
echo "TERMINAL_RESULTS_COMMAND: bash $here/launch.sh verify $root; cat $root/EXP6E3_TRAINING_SUMMARY_CN.md" >&3
echo "MAC_DOWNLOAD: run Exp6E3_pull_results_mac_v1_1.sh ON YOUR MAC after pipeline completes" >&3
