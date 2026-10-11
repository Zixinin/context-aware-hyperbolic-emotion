"""Read-only gates for native and normalized Experiment2 seed42 results."""
import csv
import hashlib
import json
from pathlib import Path

from protocol import REPO, REVISION, RARE, read, require, sha

DATASET = {'repository': 'google-research-datasets/go_emotions',
           'configuration': 'simplified', 'revision': 'add492243ff905527e67aeb8b80c082af02207c3'}
SIZES = {'train': 43410, 'validation': 5426, 'test': 5427}
PATCH = '209f83762dae1b354ba40e68b6952482ecf96208'


def objsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def dataset_signature(records):
    """Same canonical text/ID/gold/order signature as Exp2.1 evidence_ms.py."""
    rows, gold = hashlib.sha256(), hashlib.sha256()
    ids = []
    for record in records:
        labels = list(map(int, record['labels']))
        require(isinstance(record['id'], str) and isinstance(record['text'], str), 'Malformed ID/text')
        require(labels and len(set(labels)) == len(labels) and all(0 <= x < 28 for x in labels),
                'Malformed labels')
        row = {'id': record['id'], 'text': record['text'], 'labels': labels}
        rows.update((json.dumps(row, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')) + '\n').encode())
        ids.append(row['id'])
        active = set(labels)
        gold.update(bytes(int(i in active) for i in range(28)))
    require(ids and len(ids) == len(set(ids)), 'Empty split or duplicate IDs')
    return {'rows_sha256': rows.hexdigest(), 'ids_sha256': objsha(ids),
            'gold_sha256': gold.hexdigest(), 'size': len(ids)}


def normalized_metadata(run, lgsa_root):
    """Recorded historical accounting, not a live Slurm/remote checkpoint check."""
    run, root = Path(run), Path(lgsa_root)
    success = read(run / 'SUCCESS.json')
    require(success['status'] == 'NORMALIZED_RUN_PASS' and success['model'] == 'lgsa'
            and success['seed'] == 42, 'Wrong normalized baseline identity')
    hashes = success['artifacts_sha256']
    needed = {'experiment_metadata.json', 'config.json', 'test_predictions.npz',
              'validation_predictions.npz', 'checkpoint_info.json', 'validation_metrics_by_epoch.csv'}
    require(needed <= hashes.keys(), 'Baseline evidence files missing from success hashes')
    for name, digest in hashes.items():
        require(Path(name).name == name, 'Unexpected normalized artifact path')
        require(sha(run / name) == digest, 'Normalized baseline artifact changed: ' + name)
    m = read(run / 'experiment_metadata.json')
    require(m['status'] == 'COMPLETED' and m['model'] == 'lgsa' and m['seed'] == 42
            and m['phase'] == 'full', 'Incomplete normalized baseline')
    require(m['slurm_state'] == 'COMPLETED' and m['slurm_exit_code'] == '0:0', 'Historical accounting failed')
    require(m['native_success_sha256'] == success['native_success_sha256'], 'Native lineage mismatch')
    cfg = read(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json')
    require(m['config'] == cfg and read(run / 'config.json') == cfg, 'Baseline config changed')
    expected = read(REPO / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
    require(m['legacy_source_sha256'] == expected, 'Baseline adapter provenance differs')
    for name, digest in expected.items():
        require(sha(root / 'goemotions_adaptation' / name) == digest, 'Adapter drift: ' + name)
    hierarchy = root / 'goemotions_adaptation/hierarchy.json'
    names = [label['fine_label'] for label in read(hierarchy)['labels']]
    require(m['hierarchy_sha256'] == sha(hierarchy) and m['lgsa_patch_commit'] == PATCH, 'Hierarchy/patch drift')
    require(m['raw_model_repo'] == 'google-bert/bert-base-cased' and m['raw_model_revision'] == REVISION,
            'Baseline raw model differs')
    require({k: m[v] for k, v in [('repository', 'dataset'), ('configuration', 'dataset_config'),
                                 ('revision', 'dataset_revision')]} == DATASET, 'Baseline dataset pin differs')
    require(m['dataset_split_sizes'] == SIZES and m['label_names'] == names
            and m['num_labels'] == 28 and m['rare_ids'] == RARE, 'Split/label protocol differs')
    require(m['fixed_threshold'] == cfg['fixed_threshold']
            and m['threshold_candidates'] == cfg['threshold_candidates'], 'Threshold protocol differs')
    require(m['checkpoint_reload_verified'] is True and m['checkpoint_reload_evidence'], 'Reload proof missing')
    with (run / 'validation_metrics_by_epoch.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    require([int(r['epoch']) for r in history] == [1, 2, 3, 4], 'Baseline epochs incomplete')
    chosen = max(history, key=lambda r: float(r['validation_macro_f1']))
    checkpoint = read(run / 'checkpoint_info.json')
    require(int(chosen['epoch']) == m['best_epoch'] == checkpoint['best_epoch']
            and checkpoint['selection'] == cfg['checkpoint_selection'], 'Baseline checkpoint selection differs')
    require(set(m['dataset_signatures']) == set(SIZES), 'Missing split signatures')
    for split, size in SIZES.items():
        require(m['dataset_signatures'][split]['size'] == size, 'Signature split size differs')
    return m


def verify_baseline(run, lgsa_root, manifest):
    run, root = Path(run), Path(lgsa_root)
    success = read(run / 'SUCCESS.json')
    if success['status'] == 'NORMALIZED_RUN_PASS':
        m = normalized_metadata(run, root)
        require(manifest['dataset'] == DATASET, 'Prepared dataset identity differs')
        for split in SIZES:
            with (root / 'assets/goemotions-v0' / (split + '.jsonl')).open() as stream:
                signature = dataset_signature(json.loads(line) for line in stream)
            require(signature == m['dataset_signatures'][split], 'Prepared split differs from baseline: ' + split)
        # Match the actual saved arrays to the recorded data, not just metadata.
        import numpy as np
        for split in ('validation', 'test'):
            with np.load(run / (split + '_predictions.npz'), allow_pickle=False) as arrays:
                y, z, ids = arrays['gold'], arrays['logits'], arrays['ids']
                require(y.shape == z.shape == (SIZES[split], 28) and ids.shape == (SIZES[split],),
                        'Baseline prediction dimensions differ')
                require(np.isin(y, [0, 1]).all() and np.isfinite(z).all(), 'Invalid baseline predictions')
                signature = m['dataset_signatures'][split]
                require(objsha(ids.tolist()) == signature['ids_sha256']
                        and hashlib.sha256(y.astype(np.uint8).tobytes()).hexdigest() == signature['gold_sha256'],
                        'Baseline predictions do not match official IDs/GT/order')
        return {'format': 'normalized', 'success_sha256': sha(run / 'SUCCESS.json'),
                'metadata_sha256': sha(run / 'experiment_metadata.json'),
                'native_success_sha256': m['native_success_sha256'],
                'scope': 'Local artifact/signature checks; historical accounting recorded, remote native checkpoint not accessed'}
    require(success['status'] == 'GOEMOTIONS_FULL_PASS' and success['config']['seed'] == 42, 'Wrong native baseline')
    require(success['config'] == read(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json'), 'Baseline protocol differs')
    expected = read(REPO / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
    require(success['adapter_source']['files_sha256'] == expected, 'Baseline provenance differs')
    for name, digest in expected.items():
        require(sha(root / 'goemotions_adaptation' / name) == digest, 'Baseline adapter drift: ' + name)
    require(success['hierarchy_sha256'] == manifest['hierarchy_sha256'], 'Baseline hierarchy differs')
    require(sha(run / 'test_predictions.npz') == success['artifacts_sha256']['test_predictions.npz'],
            'Baseline prediction hash mismatch')
    return {'format': 'native', 'success_sha256': sha(run / 'SUCCESS.json')}
