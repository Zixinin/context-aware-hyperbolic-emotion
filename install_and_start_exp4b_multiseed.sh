#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
cd "$repo_dir"
test "$(hostname)" = D3020-A3
test "$(pwd -P)" = "$repo_dir"
python_bin="$repo_dir/.venv-lgsa/bin/python"
code_dir="$repo_dir/experiments/exp4b_multiseed_diagnostics"
test -x "$python_bin"
command -v tmux >/dev/null
sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)"
while IFS= read -r session; do
  case "$session" in
    fyp_exp4b_*)
      printf 'STOP: existing tmux session %s; inspect it first.\n' "$session"
      exit 1
      ;;
  esac
done <<< "$sessions"
if [ -e "$code_dir" ] || [ -e logs/exp4b_multiseed_workdir.txt ] || [ -e logs/exp4b_multi_seed_launcher.log ]; then
  printf 'STOP: extension or launch record already exists; nothing overwritten.\n'
  exit 1
fi
if [ -n "$(git diff --cached --name-only)" ]; then
  printf 'STOP: Git index contains other staged changes; nothing staged or committed.\n'
  exit 1
fi
test -z "$(git status --porcelain --untracked-files=all -- experiments/exp4b_data_driven_hierarchy)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42

"$python_bin" - <<'PY'
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

repo = Path.cwd().resolve()
archive = repo / 'Exp4B_multi_seed_diagnostics.zip'
expected_sha = '2e287055ccdc15476b47650940d0d12b71ab3f1ea590a4d8df30d4616e2c1610'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_sha, 'Delivery ZIP hash mismatch'
target = repo / 'experiments/exp4b_multiseed_diagnostics'
assert not target.exists(), 'Refusing overwrite'
assert target.parent.resolve() == repo / 'experiments', 'Symlinked experiments directory rejected'
names = {'README.md', 'diagnostics.py', 'launch.sh', 'm4lib.py', 'notify_retry.py',
    'preflight_extension.py', 'protocol.json', 'seed_extension.py', 'unit_tests.py'}
wanted = {'experiments/exp4b_multiseed_diagnostics/' + n for n in names}
checked = {}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read('DELIVERY_MANIFEST.json'))
    assert manifest['format'] == 'EXP4B_MULTI_SEED_DELIVERY_V1'
    assert manifest['frozen_pilot_package_sha256'] == 'ac43c7531186856af2c7f5e409ac7735a68b9808d11d4b05187a3dc610beb1a5'
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == wanted | {'DELIVERY_MANIFEST.json'}
    assert set(manifest['files']) == wanted
    for name, record in manifest['files'].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts[:2] == ('experiments', 'exp4b_multiseed_diagnostics')
        assert not stat.S_ISLNK(z.getinfo(name).external_attr >> 16)
        data = z.read(name)
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
        checked[name] = data
target.mkdir(exist_ok=False)
for name, data in checked.items():
    destination = repo / name
    assert destination.resolve().is_relative_to(target.resolve())
    with destination.open('xb') as stream:
        stream.write(data)
print('EXP4B_MULTI_EXTENSION_INSTALLED_VERIFIED: nine new sources; pilot42 unchanged')
PY

mkdir -p results/exp4b logs
work="$repo_dir/results/exp4b/multiseed_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$work"
"$python_bin" - "$work" <<'PY'
from pathlib import Path
import sys
with Path('logs/exp4b_multiseed_workdir.txt').open('x') as f:
    f.write(sys.argv[1] + '\n')
PY
send_notice() {
  "$python_bin" "$code_dir/notify_retry.py" --message "$1" --out "$work/$2" || true
}
on_failure() {
  stage_exit="$?"
  trap - ERR
  send_notice "❌ Exp4B multi-seed preparation failed exit=$stage_exit; no GPU run authorized; inspect $work" notification_preparation_failure.json
  printf 'PREPARATION_STOPPED exit=%s; retained %s\n' "$stage_exit" "$work"
  exit "$stage_exit"
}
trap on_failure ERR
send_notice '🔎 Exp4B three-seed preflight: frozen42 and matched baselines; no GPU training yet' notification_preparation_start.json
bash -n "$code_dir/launch.sh"
"$python_bin" "$repo_dir/experiments/exp4b_data_driven_hierarchy/unit_tests.py" --repo "$repo_dir" \
  --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
  --out "$work/unit_test_report.json" 2>&1 | tee "$work/pilot_unit_tests.log"
"$python_bin" "$code_dir/unit_tests.py" --repo "$repo_dir" \
  --hierarchy "$repo_dir/outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json" \
  --out "$work/extension_unit_report.json" 2>&1 | tee "$work/extension_unit_tests.log"
"$python_bin" "$code_dir/preflight_extension.py" --work "$work" --out "$work/preflight.json" \
  2>&1 | tee "$work/preflight.log"

"$python_bin" - <<'PY'
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path('experiments/exp4b_multiseed_diagnostics').resolve()))
from m4lib import HERE, REPO, extension_hashes, validate_sources
from train import check_code_committed
validate_sources()
check_code_committed()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only']).strip(), 'Other staged changes appeared'
files = sorted(str((HERE / name).relative_to(REPO)) for name in extension_hashes())
assert len(files) == 9
subprocess.run(['git', 'add', '--', *files], check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().split('\0')[:-1]
assert set(staged) == set(files), 'Staged scope differs; do not commit'
subprocess.run(['git', 'commit', '-m', 'Freeze Exp4B three-seed read-only diagnostic extension', '--', *files], check=True)
assert not subprocess.check_output(['git', 'status', '--porcelain', '--', str(HERE.relative_to(REPO))]).strip()
validate_sources()
print('EXP4B_EXTENSION_CODE_FROZEN', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
PY

tmux new-session -d -s fyp_exp4b_multiseed -c "$repo_dir" \
  "bash experiments/exp4b_multiseed_diagnostics/launch.sh prepared '$work' > logs/exp4b_multi_seed_launcher.log 2>&1"
trap - ERR
printf 'STARTED_ONLY: seed123 audit/smoke/full → seed2026 audit/smoke/full → three-seed read-only diagnostics → STOP.\n'
printf 'WORK=%s\nLOG=%s/logs/exp4b_multi_seed_launcher.log\n' "$work" "$repo_dir"
printf 'Launch is not completion. Check SUCCESS.json and verified final marker in the log.\n'
