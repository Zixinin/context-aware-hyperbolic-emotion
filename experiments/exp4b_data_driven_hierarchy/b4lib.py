"""Small provenance helpers. Never changes Exp2 or Exp4A files."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAMES = 'admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral'.split()
RARE = [16, 21, 23, 19, 12, 24, 14]
SIZES = {'train': 43410, 'validation': 5426, 'test': 5427}
REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
TREE_SHA = '8730c6b519c2ba7723c7cb93f0bddbcde57d79676443fbde406f0acb4195471d'
FROZEN_CFG_SHA = 'd425efc190e47c6100fe1ca7ed3265cd1dce596614935cdc0b9963e56ba87bb1'
RUN_SHA = '162490c8ed5a5d70f4f33c00c3801af69b998e247470a95e3bdf7875b6fdfc1d'
MODEL_SHA = {
    'models/prompt.py': '9484e5250aab1d7fa604fe74cf49d440d1d1e2b129a7f253ce04c4eec4ba7603',
    'models/graph.py': '9e2db40c05767cbc240e3349be8207a3756aaa2ed2c81eddeee94dad27024e8a',
    'models/loss.py': '6348ef910b52c6bba3b41cb2953c79c57e9132e26833028e445a467f1069371c',
    'models/attention.py': '97e28a990e06f5e128447c412bdff0265a3ff98de9659f7400bb4eea791ff280',
}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def objsha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, obj):
    """Atomic updates are limited by callers to their new output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temp, path)

def package_hashes():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.sh'}
        and '__pycache__' not in p.parts}

def notify(message):
    url = os.environ.get('FYP_NTFY_URL', 'https://ntfy.sh/zixin-fyp-4090-1119')
    try:
        request = urllib.request.Request(url, data=message.encode('utf-8'))
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        return {'delivered': True, 'message_id': result.get('id')}
    except Exception as exc:
        print('NTFY_WARNING:', type(exc).__name__, flush=True)
        return {'delivered': False, 'error_type': type(exc).__name__}

def signature(records):
    rows, gold, ids = hashlib.sha256(), hashlib.sha256(), []
    for r in records:
        labels = r['labels']
        require(isinstance(r['id'], str) and isinstance(r['text'], str), 'Invalid ID/text')
        require(labels and len(set(labels)) == len(labels)
            and all(type(i) is int and 0 <= i < 28 for i in labels), 'Invalid fine targets')
        row = {k: r[k] for k in ('id', 'text', 'labels')}
        rows.update((json.dumps(row, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')) + '\n').encode())
        ids.append(r['id'])
        gold.update(bytes(int(i in labels) for i in range(28)))
    require(ids and len(set(ids)) == len(ids), 'Duplicate/empty split IDs')
    return {'rows_sha256': rows.hexdigest(), 'ids_sha256': objsha(ids),
        'gold_sha256': gold.hexdigest(), 'size': len(ids)}
