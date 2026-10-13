#!/usr/bin/env bash
set -euo pipefail
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
test "$(hostname)" = D3020-A3
test -x .venv-lgsa/bin/python
for session in fyp_exp4b_smoke fyp_exp4b_smoke_parser_v1; do
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session %s already exists; refusing to modify running code.\n' "$session"
    exit 1
  fi
done
test ! -e logs/exp4b_smoke_parser_v1.log

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
archive = repo / 'Exp4B_tsv_parser_repair_v1.zip'
def sha(data):
    return hashlib.sha256(data).hexdigest()
assert sha(archive.read_bytes()) == '22e62b4d1de8b89d9a59c84214fa9cc2eec1cccbfb4da9b19a1f07332b4c8160', 'Repair ZIP hash mismatch'
prefix = 'experiments/exp4b_data_driven_hierarchy/'
allowed = {prefix + name for name in ('preflight.py', 'unit_tests.py', 'README.md')}
backup = repo / 'results/exp4b/repairs/tsv_parser_v1'
assert not backup.exists(), 'Repair backup already exists; refusing repeat installation'
payloads = {}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'Damaged repair archive'
    assert len(z.namelist()) == len(set(z.namelist())), 'Duplicate ZIP entries'
    assert set(z.namelist()) == allowed | {'REPAIR_MANIFEST.json'}, 'Unexpected ZIP contents'
    manifest = json.loads(z.read('REPAIR_MANIFEST.json'))
    assert set(manifest['files']) == allowed
    for name, record in manifest['files'].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert not stat.S_ISLNK(z.getinfo(name).external_attr >> 16)
        destination = repo / name
        assert destination.is_file() and not destination.is_symlink()
        assert destination.resolve().is_relative_to(repo / 'experiments/exp4b_data_driven_hierarchy')
        assert sha(destination.read_bytes()) == record['before_sha256'], 'Installed file differs from original delivery: ' + name
        assert not destination.with_name(destination.name + '.tsv-parser-v1.tmp').exists()
        data = z.read(name)
        assert sha(data) == record['after_sha256'], 'Repair file hash mismatch: ' + name
        payloads[name] = data

# All archive and old-file checks finish before any source file is replaced.
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
    temporary = destination.with_name(destination.name + '.tsv-parser-v1.tmp')
    with temporary.open('xb') as f:
        f.write(data)
    temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
for name in payloads:
    destination = repo / name
    os.replace(destination.with_name(destination.name + '.tsv-parser-v1.tmp'), destination)
    assert sha(destination.read_bytes()) == manifest['files'][name]['after_sha256']
with (backup / 'REPAIR_INSTALLED.json').open('x', encoding='utf-8') as f:
    json.dump({'status': 'EXP4B_TSV_PARSER_REPAIR_INSTALLED',
        'installed_at': datetime.now(timezone.utc).isoformat(),
        'files': manifest['files'], 'datasets_models_losses_hierarchy_modified': False}, f, indent=2)
print('EXP4B_TSV_PARSER_REPAIR_INSTALLED')
print('Original files backed up:', backup)
print('Original failed run retained: results/exp4b/run_20261012T120712Z')
PY

bash -n experiments/exp4b_data_driven_hierarchy/launch.sh
mkdir -p logs
tmux new-session -d -s fyp_exp4b_smoke_parser_v1 \
  -c /home/zixinfeng/Documents/context-aware-hyperbolic-emotion \
  'bash experiments/exp4b_data_driven_hierarchy/launch.sh smoke > logs/exp4b_smoke_parser_v1.log 2>&1'
printf 'Fresh audit and GPU smoke started. Log: logs/exp4b_smoke_parser_v1.log\n'
printf 'No seed42 full training has been launched.\n'
