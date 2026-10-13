#!/usr/bin/env bash
set -euo pipefail
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
test "$(hostname)" = D3020-A3
test -x .venv-lgsa/bin/python
test ! -e results/exp4b/run_20261012T130406Z/seed42
test ! -e results/exp4b/run_20261012T130406Z/full.log
test ! -e logs/exp4b_seed42_launcher.log
for session in fyp_exp4b_smoke fyp_exp4b_smoke_parser_v1 fyp_exp4b_smoke_raw_v2 fyp_exp4b_seed42; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Exp4B session %s already exists; stopping without changes.\n' "$session"
    exit 1
  fi
done

.venv-lgsa/bin/python - <<'PY'
from pathlib import Path
import json
import subprocess
import sys

repo = Path.cwd().resolve()
relative = 'experiments/exp4b_data_driven_hierarchy'
sys.path.insert(0, str(repo / relative))
from train import verify_run
from b4lib import package_hashes, read, require, sha

run = repo / 'results/exp4b/run_20261012T130406Z'
audit = verify_run(run / 'audit', 'EXP4B_UBUNTU_PREFLIGHT_PASS')
smoke = verify_run(run / 'smoke', 'EXP4B_SMOKE_TEST_PASS')
current = package_hashes()
unit = read(run / 'unit_test_report.json')
require(unit['status'] == 'EXP4B_UNIT_TESTS_PASS' and unit['system'] == 'Linux'
    and unit['tests_run'] == 37 and unit['failures'] == unit['errors'] == 0, 'Need exact passed Ubuntu37 unit proof')
require(unit['package_sha256'] == current == smoke['identity']['package_sha256'], 'Code changed after smoke')
require(audit['identity'] == smoke['identity'], 'Audit/smoke identities differ')
require(sha(run / 'unit_test_report.json') == smoke['identity']['unit_report_sha256'], 'Unit proof changed')
raw = read(run / 'smoke/raw_initialization.json')
require(raw['tensors_compared'] == 197 and not raw['pooler_present']
    and len(raw['legacy_LayerNorm_aliases_verified']) == 50, 'Wrong raw initialization proof')
report = read(run / 'smoke/smoke_test_report.json')
require(all(report[k] == v for k, v in {
    'forward_backward_optimizer_steps': 4, 'final_classifier_output_dimension': 28,
    'internal_supervision_nodes': 56, 'num_depths': 9, 'text_window': 50,
    'total_input_length': 68, 'test_inference_performed': False,
    'finite_losses_and_gradients': True, 'checkpoint_reload_verified': True}.items()), 'Smoke contract differs')
print('PRE_FULL_GATES_REVERIFIED', json.dumps(report), flush=True)

# Do not commit, reset or unstage the user's unrelated work.
require(not subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']),
    'Other files already staged; stopping before git add. Send this output for a scoped commit procedure.')
expected = {relative + '/' + name for name in current}
require(len(expected) == 15, 'Unexpected Exp4B source-file set')
subprocess.run(['git', 'add', '--', *sorted(expected)], check=True)
staged = {s.decode() for s in subprocess.check_output(
    ['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0') if s}
require(staged <= expected, 'Unrelated staged files detected; no commit performed')
if staged:
    subprocess.run(['git', 'commit', '-m', 'Freeze validated Exp4B full hierarchy seed42 protocol'], check=True)
require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all', '--', relative]).strip(),
    'Exp4B directory not clean; do not start full training')
tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', relative]).decode().split('\0'))
require(expected <= tracked and package_hashes() == current, 'Source freeze incomplete or content changed')
print('EXP4B_CODE_FROZEN', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), flush=True)
PY

mkdir -p logs
tmux new-session -d -s fyp_exp4b_seed42 \
  -c /home/zixinfeng/Documents/context-aware-hyperbolic-emotion \
  'bash experiments/exp4b_data_driven_hierarchy/launch.sh full /home/zixinfeng/Documents/context-aware-hyperbolic-emotion/results/exp4b/run_20261012T130406Z > logs/exp4b_seed42_launcher.log 2>&1'
printf 'STARTED_ONLY: seed42 full training -> complete test evaluation -> paired Exp2 comparison -> artifact verification.\n'
printf 'Expected: 4 epochs, 10856 optimizer updates; only seed42.\n'
printf 'Log: logs/exp4b_seed42_launcher.log\n'
printf 'Wait for EXP4B_SEED42_PILOT_COMPLETE_VERIFIED; launch alone is not completion.\n'
