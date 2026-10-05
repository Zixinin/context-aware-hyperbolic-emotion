"""Small auditable helpers; no network or training on import."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
UPSTREAM = '4ea3dfb39dcee82654f443811471dd29be32e87e'
PATCH_COMMIT = '209f83762dae1b354ba40e68b6952482ecf96208'
UPSTREAM_JOB = '2985243'
MODEL_REPO = 'google-bert/bert-base-cased'
MODEL_REV = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
SIZES = {'train': 43410, 'validation': 5426, 'test': 5427}
NAMES = 'admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral'.split()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def source_record():
    require((PACKAGE / '.git').is_dir(), 'Adapter must have its own version-controlled repository')
    require(not git(PACKAGE, 'status', '--porcelain'), 'Adapter repository must be clean')
    files = git(PACKAGE, 'ls-files').splitlines()
    return {'adaptation_commit': git(PACKAGE, 'rev-parse', 'HEAD'),
            'files_sha256': {p: sha(PACKAGE / p) for p in files}}


def config():
    return read(PACKAGE / 'lgsa_goemotions_seed42.json')
