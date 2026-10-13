#!/usr/bin/env bash
set -euo pipefail
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
test "$(hostname)" = D3020-A3
test -x .venv-lgsa/bin/python
for session in fyp_exp4b_smoke fyp_exp4b_smoke_parser_v1 fyp_exp4b_smoke_raw_v2; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session %s already exists; refusing to modify running code.\n' "$session"
    exit 1
  fi
done
test ! -e logs/exp4b_smoke_raw_v2.log

.venv-lgsa/bin/python - <<'PY'
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import stat
import zipfile
from datetime import datetime, timezone

repo = Path.cwd().resolve()
archive = repo / 'Exp4B_raw_backbone_repair_v2.zip'
def sha(data):
    return hashlib.sha256(data).hexdigest()
assert sha(archive.read_bytes()) == '16e85520f71d2cd26bfad59e652f722676e2e136c5af63b5cdd125e53c0dcb19', 'Repair ZIP hash mismatch'
prefix = 'experiments/exp4b_data_driven_hierarchy/'
allowed = {prefix + name for name in ('preflight.py', 'unit_tests.py', 'README.md')}
backup = repo / 'results/exp4b/repairs/raw_backbone_v2'
assert not backup.exists(), 'Backup already exists; do not reinstall or overwrite'
payloads = {}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'Damaged repair archive'
    assert len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive entries'
    assert set(z.namelist()) == allowed | {'REPAIR_MANIFEST.json'}, 'Unexpected archive contents'
    manifest = json.loads(z.read('REPAIR_MANIFEST.json'))
    assert set(manifest['files']) == allowed
    assert manifest['expected_unit_tests'] == 37
    for name, record in manifest['files'].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert not stat.S_ISLNK(z.getinfo(name).external_attr >> 16)
        destination = repo / name
        assert destination.is_file() and not destination.is_symlink()
        assert destination.resolve().is_relative_to(repo / 'experiments/exp4b_data_driven_hierarchy')
        assert sha(destination.read_bytes()) == record['before_sha256'], 'Installed code differs from parser-v1: ' + name
        assert not destination.with_name(destination.name + '.raw-v2.tmp').exists()
        data = z.read(name)
        assert sha(data) == record['after_sha256'], 'Replacement file hash mismatch: ' + name
        payloads[name] = data

# Finish every old-file/archive check, then back up every file before replacing any.
backup.mkdir(parents=True, exist_ok=False)
for name in sorted(payloads):
    saved = backup / name
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / name, saved)
    assert sha(saved.read_bytes()) == manifest['files'][name]['before_sha256']
with (backup / 'REPAIR_MANIFEST.json').open('x', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
for name, data in payloads.items():
    destination = repo / name
    temporary = destination.with_name(destination.name + '.raw-v2.tmp')
    with temporary.open('xb') as f:
        f.write(data)
    temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
for name in payloads:
    destination = repo / name
    os.replace(destination.with_name(destination.name + '.raw-v2.tmp'), destination)
    assert sha(destination.read_bytes()) == manifest['files'][name]['after_sha256']
with (backup / 'REPAIR_INSTALLED.json').open('x', encoding='utf-8') as f:
    json.dump({'status': 'EXP4B_RAW_BACKBONE_CHECKER_REPAIR_INSTALLED',
        'installed_at': datetime.now(timezone.utc).isoformat(), 'files': manifest['files'],
        'datasets_models_losses_hierarchy_modified': False}, f, indent=2)
print('EXP4B_RAW_BACKBONE_CHECKER_REPAIR_INSTALLED')
print('Original parser-v1 files backed up:', backup)
print('Both previous failed runs retained; no model weights or optimizer steps reused.')
PY

bash -n experiments/exp4b_data_driven_hierarchy/launch.sh
mkdir -p logs
tmux new-session -d -s fyp_exp4b_smoke_raw_v2 \
  -c /home/zixinfeng/Documents/context-aware-hyperbolic-emotion \
  'bash experiments/exp4b_data_driven_hierarchy/launch.sh smoke > logs/exp4b_smoke_raw_v2.log 2>&1'
printf 'STARTED_ONLY: 37 tests -> complete audit -> real model raw/HGS checks -> 4 GPU updates -> artifact verification.\n'
printf 'Log: logs/exp4b_smoke_raw_v2.log\n'
printf 'This is NOT a success result; wait for EXP4B_SMOKE_BATCH_COMPLETE_VERIFIED.\n'
printf 'Seed42 full training has NOT been launched.\n'
