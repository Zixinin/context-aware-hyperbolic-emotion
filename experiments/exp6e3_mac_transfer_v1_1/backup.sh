#!/usr/bin/env bash
# Repeatable incremental backup ONLY. Never train, delete, schedule or git push.
set -Eeuo pipefail
test "$(hostname)" = D3020-A3 || exit 2
repo=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_python="$repo/.venv-bert/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0
unset PYTHONPATH PYTHONHOME
exec 9<"$repo/results/exp6c1_roberta27_hypaux/.gpu_phase.lock"
flock -n 9 || { echo 'STOP: experiment phase active; back up only when quiescent'; exit 2; }
exec 8</home/zixinfeng/Documents/HypEmo-Reproduction/environment/phase2.lock
flock -n 8 || exit 2
mkdir -p /home/zixinfeng/Documents/FYP_migration_records
record=/home/zixinfeng/Documents/FYP_migration_records/backup_$(date -u +%Y%m%dT%H%M%SZ)_${RANDOM}_${RANDOM}
phase=pre
root="$repo"
if test -f "$repo/results/exp6e3_matched_gamma_coordinate/LATEST_EXP6E3_RUN.txt"; then
    root=$(cat "$repo/results/exp6e3_matched_gamma_coordinate/LATEST_EXP6E3_RUN.txt")
    if test -f "$root/SUCCESS.json"; then phase=post; fi
fi
notify() { curl --fail --silent --show-error --connect-timeout 5 --max-time 12 --data-binary "$1" https://ntfy.sh/zixin-fyp-4090-1119 >/dev/null || echo 'NOTIFICATION_DELIVERY_FAILED'; return 0; }
finish() { rc=$?;trap - EXIT;notify "FYP backup finished exit=$rc record=$record; Mac pending transfer; no training/push";echo "BACKUP_EXIT_CODE=$rc RECORD=$record";exit "$rc"; }
trap finish EXIT
notify "FYP incremental backup started: full repo/dependencies, no deletion. $record"
"$env_python" -X utf8 -u "$here/migrate.py" sync --phase "$phase" --run-root "$root" --record "$record"
