"""Shared orchestration/evidence helpers. No training or network side effects."""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PROJECTS = {'bert': BASE / 'goemotions', 'lgsa': BASE / 'LGSA-GoEmotions'}
ENV_BASE = Path('/gpfs/work/aac/zixinfeng23/.conda/envs')
PYTHONS = {'bert': ENV_BASE / 'fypemotion/bin/python', 'lgsa': ENV_BASE / 'lgsa_goemo/bin/python'}
SEEDS = (42, 123, 2026)
NAMES = 'admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral'.split()
RARE = [16, 21, 23, 19, 12, 24, 14]
SIZES = {'train': 43410, 'validation': 5426, 'test': 5427}
REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
PATCH = '209f83762dae1b354ba40e68b6952482ecf96208'
UPSTREAM = '4ea3dfb39dcee82654f443811471dd29be32e87e'
CANDIDATES = [round(i * .05, 2) for i in range(1, 19)]
METRICS = ['micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1', 'rare_f1', 'nonrare_f1']
OLD = {'bert': PROJECTS['bert'] / 'outputs/bert_baseline/seed42',
       'lgsa': PROJECTS['lgsa'] / 'outputs/lgsa_goemotions/seed42/2985311'}
MODEL_DIR = {'bert': 'bert_baseline', 'lgsa': 'lgsa_goemotions'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def objsha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def capture(args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, **kwargs).strip()


def csvwrite(path, rows):
    require(bool(rows), 'Refusing empty table: ' + str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def csvread(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def release():
    manifest = read(ROOT / 'RELEASE.json')
    for name, digest in manifest['files_sha256'].items():
        require(sha(ROOT / name) == digest, 'Release file changed: ' + name)
    return sha(ROOT / 'RELEASE.json')


def artifacts(path, success=None):
    path = Path(path)
    success = success or read(path / 'SUCCESS.json')
    hashes = success['artifacts_sha256']
    require(bool(hashes), 'Empty artifact manifest')
    for name, digest in hashes.items():
        target = path / name
        require(target.is_file() and sha(target) == digest, 'Artifact failed: ' + str(target))
    return hashes


def sacct(job):
    require(str(job).isdecimal(), 'Invalid job id')
    raw = capture(['sacct', '-S', '2026-09-01', '-j', str(job), '--parsable2', '--noheader',
                   '--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS,AllocTRES'])
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    lookup = {r[0]: r for r in rows}
    for key in (str(job), str(job) + '.batch'):
        require(key in lookup, 'Accounting unavailable (do not assume failure or rerun): ' + key)
        require(lookup[key][1] == 'COMPLETED' and lookup[key][2] == '0:0', 'Slurm not COMPLETED/0:0: ' + str(lookup[key]))
    for key, row in lookup.items():
        if key.startswith(str(job) + '.'):
            require(row[1] == 'COMPLETED' and row[2] == '0:0', 'Failed Slurm step: ' + str(row))
    return {'job_id': str(job), 'rows': rows, 'verified_at': now()}


def check_logs(paths):
    result = {}
    pattern = r'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|oom-kill|(?:^|\n).*\bKilled\b|DUE TO TIME LIMIT'
    for p in paths:
        p = Path(p)
        require(p.is_file(), 'Missing job log: ' + str(p))
        require(not re.search(pattern, p.read_text(errors='replace'), re.IGNORECASE), 'Error in log: ' + str(p))
        result[str(p)] = sha(p)
    return result


def config(model, seed, phase='full'):
    require(model in PROJECTS and seed in SEEDS, 'Unsupported model/seed')
    cfg = read(ROOT / 'frozen' / (model + '.json'))
    cfg['seed'] = seed
    if model == 'bert':
        cfg['mode'] = phase
        if phase == 'smoke':
            cfg['epochs'] = 1
    # LGSA's legacy experiment_name remains frozen; metadata supplies the actual seed.
    return cfg


def current_sources(model):
    root = PROJECTS[model] if model == 'bert' else PROJECTS[model] / 'goemotions_adaptation'
    expected = read(ROOT / 'frozen/expected_sources.json')[model]
    for name, digest in expected.items():
        require(sha(root / name) == digest, 'Original source drift; stop before training: ' + str(root / name))
    if model == 'lgsa':
        require(capture(['git', '-C', root, 'status', '--porcelain']) == '', 'Original adapter has uncommitted changes')
        for sub, commit in [('original_lgsa', UPSTREAM), ('working_lgsa', PATCH)]:
            path = PROJECTS[model] / sub
            require(capture(['git', '-C', path, 'rev-parse', 'HEAD']) == commit, 'LGSA commit mismatch: ' + sub)
            require(capture(['git', '-C', path, 'status', '--porcelain']) == '', 'Dirty LGSA tree: ' + sub)
    return expected


def runtime_environment():
    import importlib.metadata as im
    import platform
    import torch
    packages = {}
    for name in ['torch', 'transformers', 'datasets', 'numpy', 'torch-geometric', 'scikit-learn', 'huggingface-hub']:
        try:
            packages[name] = im.version(name)
        except im.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'python_executable': sys.executable,
            'hostname': platform.node(), 'packages': packages, 'cuda': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def expected_environment(model):
    current = runtime_environment()
    versions = {'bert': {'torch': '2.6.0+cu118', 'transformers': '4.51.3', 'datasets': '3.5.0'},
                'lgsa': {'torch': '2.5.1+cu118', 'transformers': '4.44.2', 'datasets': '2.21.0', 'torch-geometric': '2.6.1'}}[model]
    require(current['python'] == '3.10.21' and current['cuda'] == '11.8', 'Python/CUDA drift')
    require(Path(sys.executable).resolve() == PYTHONS[model].resolve(), 'Use dedicated ' + model + ' Python')
    for k, v in versions.items():
        require(current['packages'].get(k) == v, 'Environment mismatch: ' + k)
    frozen = (OLD[model] / ('pip-freeze.txt' if model == 'bert' else 'requirements-installed.txt')).read_text()
    actual = capture([sys.executable, '-m', 'pip', 'freeze'])
    require(sorted(frozen.splitlines()) == sorted(actual.splitlines()), 'Installed dependencies changed since seed42; do not upgrade/downgrade automatically')
    current['pip_freeze_sha256'] = hashlib.sha256((actual + '\n').encode()).hexdigest()
    return current


def load_audit():
    require(read(ROOT / 'RELEASE.json').get('development_only') is False, 'Development releases cannot run HPC experiments')
    a = read(ROOT / 'audit/SEED42_AUDIT_PASS.json')
    require(a['release_sha256'] == release(), 'Audit belongs to a different code release')
    require(a['status'] == 'SEED42_AUDIT_PASS', 'Seed42 audit not passed')
    return a
