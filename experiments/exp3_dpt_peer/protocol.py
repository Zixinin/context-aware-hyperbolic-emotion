"""Frozen protocol, source checks and portable path binding for Ubuntu."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
DPT_COMMIT = '43e2daf6578ed6df885329cf229c3f5b749e19e7'
RARE = [16, 21, 23, 19, 12, 24, 14]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def package_hashes():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh'} and '__pycache__' not in p.parts}


def source_audit(lgsa_root):
    root = Path(lgsa_root).resolve()
    adapter, model = root / 'goemotions_adaptation', root / 'working_lgsa'
    frozen = read(REPO / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
    records = {}
    for name, expected in frozen.items():
        path = adapter / name
        require(path.is_file() and sha(path) == expected, 'Experiment 2 adapter drift: ' + str(path))
        records['adapter/' + name] = expected
    model_expected = read(HERE / 'frozen_model_sources.json')
    for name, expected in model_expected.items():
        path = model / name
        require(path.is_file() and sha(path) == expected, 'LGSA model drift: ' + str(path))
        records['model/' + name] = expected
    cfg = read(adapter / 'lgsa_goemotions_seed42.json')
    require(cfg == read(REPO / 'experiments/exp2_1_multiseed/frozen/lgsa.json'), 'Exp2 config drift')
    require(read(adapter / 'hierarchy.json') == read(REPO / 'experiments/exp2_1_multiseed/frozen/hierarchy.json'),
            'Frozen hierarchy mismatch')
    return records


def bind_legacy(root):
    root = Path(root).resolve()
    adapter = root / 'goemotions_adaptation'
    require('run' not in sys.modules and 'common' not in sys.modules, 'Run launcher in a fresh Python process')
    sys.path.insert(0, str(adapter))
    import run as legacy
    require(Path(legacy.__file__).resolve() == adapter / 'run.py', 'Wrong legacy trainer import')
    return legacy


def assets(root, model_path=None):
    root = Path(root).resolve()
    directory = root / 'assets/goemotions-v0'
    manifest = read(directory / 'manifest.json')
    require(manifest['status'] == 'GOEMOTIONS_ASSETS_READY', 'Missing prepared Exp2 dataset assets')
    require(manifest['split_sizes'] == {'train': 43410, 'validation': 5426, 'test': 5427}, 'Split sizes changed')
    require(manifest['rare_ids'] == RARE, 'Frozen training-frequency rare labels changed')
    adapter = root / 'goemotions_adaptation'
    require(manifest['config_sha256'] == sha(adapter / 'lgsa_goemotions_seed42.json'), 'Manifest config mismatch')
    require(manifest['hierarchy_sha256'] == sha(adapter / 'hierarchy.json'), 'Manifest hierarchy mismatch')
    # Regression and smoke must not read test labels; its bytes are verified in full only.
    for name in ('train.jsonl', 'validation.jsonl'):
        require(sha(directory / name) == manifest['files_sha256'][name], 'Split hash mismatch: ' + name)
    record = manifest['model']
    require(record['repo_id'] == 'google-bert/bert-base-cased' and record['revision'] == REVISION, 'Wrong raw model pin')
    model = Path(model_path).resolve() if model_path else Path(record['path'])
    require(model.name == REVISION, 'Model path must be the original pinned snapshot, never a checkpoint')
    for name, expected in record['sha256'].items():
        require(sha(model / name) == expected, 'Raw model file changed: ' + name)
    # Relocate only the in-memory path. Never rewrite old manifests or assets.
    manifest['model'] = {**record, 'path': str(model)}
    return manifest


def verify_raw(model, manifest):
    import torch
    from safetensors import safe_open
    aliases = []
    count = 0
    with safe_open(str(Path(manifest['model']['path']) / 'model.safetensors'), framework='pt', device='cpu') as raw:
        keys = set(raw.keys())
        for name, value in model.bert.state_dict().items():
            candidates = ['bert.' + name, name]
            if name.endswith('.LayerNorm.weight'):
                old = name[:-len('weight')] + 'gamma'
                candidates += ['bert.' + old, old]
            elif name.endswith('.LayerNorm.bias'):
                old = name[:-len('bias')] + 'beta'
                candidates += ['bert.' + old, old]
            key = next((key for key in candidates if key in keys), None)
            require(key is not None, 'Missing raw backbone tensor: ' + name)
            reference = raw.get_tensor(key)
            require(value.dtype == reference.dtype and tuple(value.shape) == tuple(reference.shape)
                    and torch.equal(value.detach().cpu(), reference), 'Not fresh raw initialization: ' + name)
            aliases.extend([key] if key.endswith(('.gamma', '.beta')) else [])
            count += 1
    return {'status': 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED', 'tensors_compared': count,
            'exact_legacy_layernorm_aliases': aliases, 'revision': REVISION,
            'raw_model_sha256': manifest['model']['sha256']['model.safetensors']}


def git_info():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--short', '--branch')}
