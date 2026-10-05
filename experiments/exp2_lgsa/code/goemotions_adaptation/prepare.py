"""Read existing baseline RAW assets, independently fingerprint adaptation inputs."""
import json
import os
from pathlib import Path
from common import ROOT, PACKAGE, MODEL_REPO, MODEL_REV, SIZES, NAMES, read, write, sha, require, now, config
from hierarchy_goemotions import load_hierarchy, targets

ASSETS = ROOT / 'assets/goemotions-v0'


def verify_assets():
    manifest = read(ASSETS / 'manifest.json')
    require(manifest['status'] == 'GOEMOTIONS_ASSETS_READY', 'Assets not prepared')
    require(manifest['hierarchy_sha256'] == sha(PACKAGE / 'hierarchy.json'), 'Hierarchy changed')
    require(manifest['config_sha256'] == sha(PACKAGE / 'lgsa_goemotions_seed42.json'), 'Configuration changed')
    require(manifest['model']['repo_id'] == MODEL_REPO and manifest['model']['revision'] == MODEL_REV, 'Wrong raw model')
    for name, digest in manifest['model']['sha256'].items():
        require(sha(Path(manifest['model']['path']) / name) == digest, 'Raw model file changed: ' + name)
    for name, digest in manifest['files_sha256'].items():
        require(sha(ASSETS / name) == digest, 'Data file changed: ' + name)
    return manifest


def prepare():
    if (ASSETS / 'manifest.json').exists():
        verify_assets()
        print('GOEMOTIONS_ASSETS_READY (verified existing)', flush=True)
        return
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from data_goemotions import encode
    base = ROOT.parent / 'goemotions/assets'
    baseline = read(base / 'manifest.json')
    require(baseline['model_repo'] == MODEL_REPO and baseline['model_revision'] == MODEL_REV, 'Baseline raw model pin changed')
    model = Path(baseline['model_snapshot'])
    require(model.name == MODEL_REV and model.parent.name == 'snapshots', 'Expected pinned raw HF snapshot, never checkpoint')
    for name in ['config.json', 'vocab.txt', 'model.safetensors']:
        require(name in baseline['model_sha256'], 'Missing baseline raw asset hash: ' + name)
    for name, digest in baseline['model_sha256'].items():
        require(sha(model / name) == digest, 'Raw baseline asset checksum mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, clean_up_tokenization_spaces=False)
    require(getattr(tokenizer, 'do_lower_case', False) is False, 'Cased tokenizer required')
    hierarchy = load_hierarchy(PACKAGE / 'hierarchy.json')
    dataset = load_from_disk(str(base / 'dataset'))
    require(set(dataset) == set(SIZES), 'Unexpected baseline splits')
    require({s: dataset[s]._fingerprint for s in dataset} == baseline['disk_fingerprints'], 'Baseline dataset fingerprint changed')
    ASSETS.mkdir(parents=True, exist_ok=True)
    seen, frequencies, counts, digests = set(), [0] * 28, {}, {}
    for split, expected in SIZES.items():
        data = dataset[split]
        require(len(data) == expected, 'Wrong split size')
        names = getattr(getattr(data.features['labels'], 'feature', None), 'names', None)
        require(names is None or names == NAMES, 'Baseline label order differs')
        destination = ASSETS / (split + '.jsonl')
        temp = destination.with_suffix('.partial')
        with temp.open('w') as stream:
            for row in data:
                record = {k: row[k] for k in ('id', 'text', 'labels')}
                require(isinstance(record['id'], str) and record['id'] not in seen, 'Duplicate/malformed ID')
                require(isinstance(record['text'], str), 'Malformed text')
                seen.add(record['id'])
                targets(record['labels'], hierarchy)
                if split == 'train':
                    for i in record['labels']:
                        frequencies[i] += 1
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(temp, destination)
        # Local-only tokenizer compatibility; no BERT forward/backward on login node.
        encode(data[0], tokenizer, hierarchy, config()['text_max_length'])
        counts[split] = len(data)
        digests[destination.name] = sha(destination)
    rare = sorted(range(28), key=lambda i: (frequencies[i], i))[:7]
    require(rare == [16, 21, 23, 19, 12, 24, 14], 'Train frequency rare set differs from Experiment 1')
    write(ASSETS / 'manifest.json', {'status': 'GOEMOTIONS_ASSETS_READY', 'created_at': now(),
        'source': 'Read-only reuse of Experiment 1 official dataset assets; no trained checkpoint read',
        'baseline_manifest_sha256': sha(base / 'manifest.json'), 'dataset': baseline['dataset'],
        'split_sizes': counts, 'train_frequencies': frequencies, 'rare_ids': rare,
        'hierarchy_sha256': sha(PACKAGE / 'hierarchy.json'),
        'config_sha256': sha(PACKAGE / 'lgsa_goemotions_seed42.json'),
        'files_sha256': digests, 'model': {'repo_id': MODEL_REPO, 'revision': MODEL_REV,
        'path': str(model), 'sha256': baseline['model_sha256']}})
    verify_assets()
    print('GOEMOTIONS_ASSETS_READY', counts, flush=True)


if __name__ == '__main__':
    prepare()
