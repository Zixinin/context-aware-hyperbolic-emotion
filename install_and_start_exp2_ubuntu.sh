#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
cd "$repo_dir"
test "$(hostname)" = D3020-A3
test "$(pwd -P)" = "$repo_dir"
python_bin="$repo_dir/.venv-lgsa/bin/python"
code_dir="$repo_dir/experiments/exp2_ubuntu_rerun"
test -x "$python_bin"
command -v tmux >/dev/null
sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)"
while IFS= read -r session; do
  case "$session" in
    fyp_exp4b_*|fyp_exp2_ubuntu_*)
      printf 'STOP: existing tmux session %s; inspect first.\n' "$session"
      exit 1
      ;;
  esac
done <<< "$sessions"
if [ -e "$code_dir" ] || [ -e logs/exp2_ubuntu_rerun_workdir.txt ] || [ -e logs/exp2_ubuntu_rerun_launcher.log ]; then
  printf 'STOP: rerun source/launch record already exists; refusing overwrite.\n'
  exit 1
fi
test -z "$(git diff --cached --name-only)"
test -z "$(git status --porcelain --untracked-files=all -- experiments/exp4b_data_driven_hierarchy experiments/exp4b_multiseed_diagnostics)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42

"$python_bin" - <<'PY'
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
repo = Path.cwd().resolve()
archive = repo / 'Exp2_ubuntu_rerun.zip'
expected_sha = '04496c77cfdbb26e4026bae657e8c81119bfd7bb3761b17a877dc18e53144bd7'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_sha, 'ZIP hash mismatch'
target = repo / 'experiments/exp2_ubuntu_rerun'
assert not target.exists(), 'Refusing overwrite'
assert target.parent.resolve() == repo / 'experiments', 'Symlinked experiments directory rejected'
names = {'r2lib.py', 'train_ubuntu.py', 'compare_ubuntu.py', 'unit_tests_ubuntu.py',
    'launch.sh', 'README.md', 'protocol.json'}
wanted = {'experiments/exp2_ubuntu_rerun/' + n for n in names}
checked = {}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read('DELIVERY_MANIFEST.json'))
    assert manifest['format'] == 'EXP2_UBUNTU_RERUN_DELIVERY_V1'
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == wanted | {'DELIVERY_MANIFEST.json'}
    assert set(manifest['files']) == wanted
    for name, record in manifest['files'].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts[:2] == ('experiments', 'exp2_ubuntu_rerun')
        assert not stat.S_ISLNK(z.getinfo(name).external_attr >> 16)
        data = z.read(name)
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        checked[name] = data
target.mkdir(exist_ok=False)
for name, data in checked.items():
    destination = repo / name
    assert destination.resolve().is_relative_to(target.resolve())
    with destination.open('xb') as f: f.write(data)
print('EXP2_UBUNTU_CODE_INSTALLED_VERIFIED: seven new source files; no old code/results overwritten')
PY

mkdir -p results/exp2_ubuntu logs
work="$repo_dir/results/exp2_ubuntu/run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$work"
"$python_bin" - "$work" <<'PY'
from pathlib import Path
import sys
with Path('logs/exp2_ubuntu_rerun_workdir.txt').open('x') as f:
    f.write(sys.argv[1]+'\n')
PY
send_notice() {
  "$python_bin" experiments/exp4b_multiseed_diagnostics/notify_retry.py --message "$1" --out "$work/$2" || true
}
on_failure() {
  stage_exit="$?"
  trap - ERR
  send_notice "❌ Exp2 Ubuntu preparation failed exit=$stage_exit; no new GPU training; inspect $work" notification_preparation_failure.json
  printf 'PREPARATION_STOPPED exit=%s; retained %s\n' "$stage_exit" "$work"
  exit "$stage_exit"
}
trap on_failure ERR
send_notice '🔎 Original Exp2 Ubuntu rerun preparation: verify old protocol and completed Exp4B; no GPU training yet' notification_preparation_start.json
bash -n "$code_dir/launch.sh"
"$python_bin" experiments/exp4b_data_driven_hierarchy/unit_tests.py --repo "$repo_dir" \
  --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
  --out "$work/pilot_unit_report.json" 2>&1 | tee "$work/pilot_units.log"
"$python_bin" experiments/exp4b_multiseed_diagnostics/unit_tests.py --repo "$repo_dir" \
  --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
  --out "$work/multi_unit_report.json" 2>&1 | tee "$work/multi_units.log"
"$python_bin" "$code_dir/unit_tests_ubuntu.py" \
  --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
  --out "$work/unit_test_report.json" 2>&1 | tee "$work/rerun_units.log"
"$python_bin" "$code_dir/train_ubuntu.py" --preflight --out "$work/preflight.json" 2>&1 | tee "$work/preflight.log"

"$python_bin" - <<'PY'
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path('experiments/exp2_ubuntu_rerun').resolve()))
from r2lib import HERE, REPO, code_hashes, validate_sources
from seed_extension import check_committed
validate_sources(); check_committed()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only']).strip(), 'Other staged changes appeared'
files = sorted(str((HERE / n).relative_to(REPO)) for n in code_hashes())
assert len(files) == 7
subprocess.run(['git', 'add', '--', *files], check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().split('\0')[:-1]
assert set(staged) == set(files), 'Unexpected staged scope; no commit'
subprocess.run(['git', 'commit', '-m', 'Freeze unchanged original Exp2 Ubuntu paired-seed rerun', '--', *files], check=True)
assert not subprocess.check_output(['git', 'status', '--porcelain', '--', str(HERE.relative_to(REPO))]).strip()
validate_sources()
print('EXP2_UBUNTU_CODE_FROZEN', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
PY

tmux new-session -d -s fyp_exp2_ubuntu_rerun -c "$repo_dir" \
  "bash experiments/exp2_ubuntu_rerun/launch.sh prepared '$work' > logs/exp2_ubuntu_rerun_launcher.log 2>&1"
trap - ERR
printf 'STARTED_ONLY: original Exp2 seed42→123→2026 audit/smoke/full → same-host Exp4B comparison → STOP.\n'
printf 'WORK=%s\nLOG=%s/logs/exp2_ubuntu_rerun_launcher.log\n' "$work" "$repo_dir"
printf 'Old HPC and Exp4B results retained. Launch is NOT completion.\n'
