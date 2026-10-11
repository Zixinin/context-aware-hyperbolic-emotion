"""CPU-only pinned data preparation, checked against existing Exp2.1 signatures."""
import argparse
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

from baseline import DATASET, SIZES, dataset_signature, normalized_metadata
from protocol import REPO, REVISION, RARE, read, require, sha, source_audit, write


def prepare_records(dataset, target, expected_signatures, names):
    """Create new files only; retain any partial directory on verification failure."""
    target = Path(target)
    require(not target.exists(), 'Never overwrite existing prepared assets')
    require(set(dataset) == set(SIZES), 'Unexpected dataset splits')
    stage = target.with_name(target.name + '.exp3-preparing')
    stage.mkdir(parents=True, exist_ok=False)
    print('New preparation directory:', stage, flush=True)
    seen, frequencies, signatures, hashes = set(), [0] * 28, {}, {}
    for split, size in SIZES.items():
        require(len(dataset[split]) == size, 'Wrong downloaded split size: ' + split)
        feature_names = getattr(getattr(dataset[split].features['labels'], 'feature', None), 'names', None)
        require(feature_names == names, 'Downloaded label order differs')
        destination = stage / (split + '.jsonl')
        with destination.open('x') as stream:
            for row in dataset[split]:
                record = {k: row[k] for k in ('id', 'text', 'labels')}
                require(isinstance(record['id'], str) and record['id'] not in seen, 'Duplicate/malformed ID or split leakage')
                seen.add(record['id'])
                # Check full annotation validity while calculating the signature below.
                if split == 'train':
                    for label in record['labels']:
                        require(isinstance(label, int) and 0 <= label < 28, 'Malformed train label')
                        frequencies[label] += 1
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        with destination.open() as stream:
            signatures[split] = dataset_signature(json.loads(line) for line in stream)
        require(signatures[split] == expected_signatures[split], 'Data text/IDs/labels/order changed: ' + split)
        hashes[destination.name] = sha(destination)
        print(split, 'OFFICIAL_BASELINE_SIGNATURE_PASS', size, flush=True)
    require(sorted(range(28), key=lambda i: (frequencies[i], i))[:7] == RARE, 'Rare-label set changed')
    return stage, signatures, hashes, frequencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lgsa-root', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--pretrained-record', type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3', 'Prepare on the specified Ubuntu workstation')
    root = args.lgsa_root.resolve()
    require(root.is_relative_to(REPO) and root != REPO, 'Expected existing Exp2 runtime root inside repository')
    source_audit(root)
    target = root / 'assets/goemotions-v0'
    require(not target.exists(), 'Assets directory already exists; inspect it, never overwrite')
    m = normalized_metadata(args.baseline_run, root)
    raw = read(args.pretrained_record)
    require(raw['status'] == 'PINNED_RAW_BERT_ASSETS_READY' and raw['repository'] == 'google-bert/bert-base-cased'
            and raw['revision'] == REVISION, 'Wrong raw BERT assets')
    model = Path(raw['path']).resolve()
    require(model.name == REVISION, 'Expected pinned raw snapshot, never a trained checkpoint')
    expected_raw = read(REPO / 'experiments/exp2_1_multiseed/frozen/bert_archive_audit.json')['environment']['assets_manifest']['model_sha256']
    for name, expected in expected_raw.items():
        require(raw['files_sha256'].get(name) == expected and sha(model / name) == expected,
                'Pinned historical BERT file differs: ' + name)
    # No torch imports, model construction, CUDA calls, optimizer or inference.
    from datasets import load_dataset
    dataset = load_dataset(DATASET['repository'], DATASET['configuration'], revision=DATASET['revision'])
    names = [r['fine_label'] for r in read(root / 'goemotions_adaptation/hierarchy.json')['labels']]
    stage, signatures, hashes, frequencies = prepare_records(dataset, target, m['dataset_signatures'], names)
    manifest = {'status': 'GOEMOTIONS_ASSETS_READY', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': 'Ubuntu re-download of pinned official data; exact baseline text/IDs/GT/order signatures verified',
        'dataset': DATASET, 'split_sizes': SIZES, 'train_frequencies': frequencies, 'rare_ids': RARE,
        'dataset_signatures': signatures, 'files_sha256': hashes,
        'hierarchy_sha256': sha(root / 'goemotions_adaptation/hierarchy.json'),
        'config_sha256': sha(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json'),
        'model': {'repo_id': raw['repository'], 'revision': REVISION, 'path': str(model), 'sha256': expected_raw},
        'baseline_success_sha256': sha(args.baseline_run / 'SUCCESS.json'),
        'baseline_metadata_sha256': sha(args.baseline_run / 'experiment_metadata.json'),
        'pretrained_record_sha256': sha(args.pretrained_record),
        'scope': 'Data preparation/signature checks only; no test inference, no GPU training'}
    write(stage / 'manifest.json', manifest)
    source_audit(root)
    require(not target.exists(), 'Target appeared during preparation; partial directory retained')
    stage.rename(target)
    print('GOEMOTIONS_ASSETS_READY:', target, flush=True)


if __name__ == '__main__':
    main()
