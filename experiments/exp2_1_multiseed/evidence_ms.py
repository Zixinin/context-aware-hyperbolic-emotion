"""Read-only audit of legacy seed42 and newly completed legacy-runner outputs."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from mslib import *
from metrics_ms import load_predictions, probabilities, score, threshold_search, close_metrics


def dataset_signature(records):
    digest, ids, labels = hashlib.sha256(), [], []
    for r in records:
        row = {'id': str(r['id']), 'text': r['text'], 'labels': list(map(int, r['labels']))}
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode())
        ids.append(row['id'])
        y = np.zeros(28, dtype=np.uint8)
        require(len(set(row['labels'])) == len(row['labels']) and row['labels'], 'Bad annotations')
        require(all(0 <= i < 28 for i in row['labels']), 'Bad label ID')
        y[row['labels']] = 1
        labels.append(y)
    require(len(ids) == len(set(ids)), 'Duplicate dataset IDs')
    y = np.stack(labels)
    return {'rows_sha256': digest.hexdigest(), 'ids_sha256': objsha(ids),
            'gold_sha256': hashlib.sha256(y.tobytes()).hexdigest(), 'size': len(ids)}, ids, y


def asset_data(model):
    root = PROJECTS[model]
    if model == 'bert':
        sys.path.insert(0, str(root))
        from src.data import load_prepared, LABEL_NAMES
        ds, manifest = load_prepared()
        require(LABEL_NAMES == NAMES, 'BERT label order drift')
        pinned = read(ROOT / 'frozen/bert_archive_audit.json')['environment']['assets_manifest']
        require(manifest == pinned, 'BERT original manifest changed')
        data = {s: dataset_signature(ds[s]) for s in SIZES}
    else:
        sys.path.insert(0, str(root / 'goemotions_adaptation'))
        from prepare import verify_assets, ASSETS
        from common import NAMES as legacy_names
        manifest = verify_assets()
        require(legacy_names == NAMES and manifest['rare_ids'] == RARE, 'LGSA labels/rare set drift')
        require(manifest['baseline_manifest_sha256'] == sha(PROJECTS['bert'] / 'assets/manifest.json'), 'LGSA dataset lineage mismatch')
        require(manifest['model']['revision'] == REVISION, 'Model revision changed')
        require(manifest['hierarchy_sha256'] == sha(ROOT / 'frozen/hierarchy.json'), 'Hierarchy drift')
        data = {}
        for split in SIZES:
            with (ASSETS / (split + '.jsonl')).open() as f:
                data[split] = dataset_signature(json.loads(line) for line in f)
    dataset = manifest['dataset']
    require(dataset == {'repository': 'google-research-datasets/go_emotions', 'configuration': 'simplified',
                        'revision': 'add492243ff905527e67aeb8b80c082af02207c3'}, 'Dataset identity drift')
    require({s: data[s][0]['size'] for s in SIZES} == SIZES, 'Dataset size mismatch')
    all_ids = [i for s in SIZES for i in data[s][1]]
    require(len(all_ids) == len(set(all_ids)), 'Split leakage')
    freq = data['train'][2].sum(0)
    require(sorted(range(28), key=lambda i: (freq[i], i))[:7] == RARE, 'Rare-label definition drift')
    return manifest, data


def check_geometry(out, cfg):
    g = read(out / 'geometry_acceptance.json')
    require(g['status'] == 'GOEMOTIONS_HGS_PASS' and g['map_matches'] is True and g['fine_vectors_changed'] is True,
            'HGS map/updates failed')
    require(g['norm_projection_preservation'] is True, 'HGS norm preservation failed')
    require(math.isfinite(g['top_max_abs_cosine']) and g['top_max_abs_cosine'] <= cfg['geometry_cosine_tolerance'], 'Top orthogonality failed')
    for f in g['families'].values():
        require(f['rank'] == f['children'] and math.isfinite(f['max_abs_cosine'])
                and f['max_abs_cosine'] <= cfg['geometry_cosine_tolerance'], 'Sibling residual geometry failed')
    return g


def validation_history(model, out):
    if model == 'bert':
        return [{'epoch': int(r['epoch']), 'validation_micro_f1': float(r['validation_micro_f1']),
                 'validation_macro_f1': float(r['validation_macro_f1'])} for r in csvread(out / 'training_log.csv')]
    rows = []
    for p in sorted(out.glob('validation_epoch_*.json')):
        r = read(p)
        rows.append({'epoch': int(p.stem.split('_')[-1]), 'validation_micro_f1': r['micro_f1'],
                     'validation_macro_f1': r['macro_f1']})
    return rows


def verify_legacy(model, out, seed=42, phase='full', data=None):
    out = Path(out)
    s = read(out / 'SUCCESS.json')
    hashes = artifacts(out, s)
    cfg = config(model, seed, phase)
    expected = read(ROOT / 'frozen/expected_sources.json')[model]
    full = phase == 'full'
    if model == 'bert':
        m = read(out / 'environment.json')
        require(m['status'] == 'completed' and s['mode'] == phase, 'Incomplete BERT run')
        require(m['config'] == cfg and read(out / 'config.json') == cfg, 'BERT config mismatch')
        require(m['source_sha256'] == expected and s['source_sha256'] == expected, 'BERT original source identity mismatch')
        require(m['bce_loss_verified'] is True, 'BCE check missing')
        require(m['assets_manifest']['model_revision'] == REVISION, 'BERT model revision mismatch')
        require(m['assets_manifest_sha256'] == sha(PROJECTS['bert'] / 'assets/manifest.json'), 'BERT data/model assets changed')
        require(read(out / 'dataset_analysis/rare_labels.json')['label_ids'] == RARE, 'BERT rare set mismatch')
        require(m['completed_optimizer_steps'] == (10856 if full else 63), 'BERT steps incomplete')
        require(m['warmup_steps'] == (1086 if full else 7), 'BERT ceil warmup changed')
        best = read(out / 'checkpoints/best/selection.json')
        require(best == m['best_checkpoint'], 'BERT checkpoint selection mismatch')
        require('checkpoints/best/model.safetensors' in hashes, 'BERT checkpoint not hashed')
        if not full:
            require(m['checkpoint_reload_verified'] is True and m['actual_split_sizes']['test_evaluated'] == 0,
                    'BERT smoke reload/test isolation failed')
        reload_evidence = ('Original hashed full runner reloads best checkpoint for sole test inference'
                           if full else 'Validation logits matched after checkpoint reload')
    else:
        m = read(out / 'experiment_metadata.json')
        require(s == m and m['status'] == ('GOEMOTIONS_FULL_PASS' if full else 'GOEMOTIONS_SMOKE_PASS'), 'LGSA completion mismatch')
        require(m['config'] == cfg and m['phase'] == phase, 'LGSA config mismatch')
        require(m['adapter_source']['files_sha256'] == expected, 'LGSA adapter source identity mismatch')
        require(m['original_LGSA_commit'] == UPSTREAM and m['tested_parentmap_commit'] == PATCH, 'LGSA patch mismatch')
        require(m['model']['revision'] == REVISION, 'LGSA model revision mismatch')
        require(m['dataset_manifest_sha256'] == sha(PROJECTS['lgsa'] / 'assets/goemotions-v0/manifest.json'), 'LGSA assets changed')
        require(m['hierarchy_sha256'] == sha(ROOT / 'frozen/hierarchy.json'), 'LGSA hierarchy mismatch')
        require(m['initialization'] == 'raw_pretrained' and m['initial_checkpoint'] is None, 'LGSA inherited checkpoint')
        require(m['optimizer_steps'] == (10856 if full else 63), 'LGSA steps incomplete')
        for f in ['checkpoint_reload_logits_match', 'backbone_gradient_passed', 'gat_gradient_passed', 'finite_training_passed']:
            require(m[f] is True, 'LGSA failed: ' + f)
        require(m['backbone_parameter_max_change'] > 0 and math.isfinite(m['backbone_parameter_max_change']), 'No backbone update')
        require(m['test_used'] == full, 'LGSA test-use mismatch')
        require('checkpoint_best.pt' in hashes, 'LGSA checkpoint not hashed')
        check_geometry(out, cfg)
        reload = read(out / 'checkpoint_reload.json')
        require(reload['logits_match'] is True and reload['selected_epoch'] == m['best_epoch'], 'LGSA checkpoint reload failed')
        best = {'epoch': m['best_epoch']}
        reload_evidence = 'CHECKPOINT_RELOAD_PASS: perturb, restore and logits comparison'
    history = validation_history(model, out)
    require([r['epoch'] for r in history] == list(range(1, (4 if full else 1) + 1)), 'Missing/repeated epochs')
    require(all(math.isfinite(r[k]) for r in history for k in ['validation_micro_f1', 'validation_macro_f1']), 'Nonfinite validation metrics')
    chosen = max(history, key=lambda r: r['validation_macro_f1'])
    require(best['epoch'] == chosen['epoch'], 'Wrong best checkpoint rule')
    gold, logits, ids = load_predictions(out / 'validation_predictions.npz')
    require(len(ids) == (SIZES['validation'] if full else 500), 'Validation size mismatch')
    val_score = score(gold, probabilities(logits, model), .3)[0]
    require(abs(val_score['macro_f1'] - chosen['validation_macro_f1']) < 1e-10, 'Best validation predictions mismatch')
    selected, trials = threshold_search(gold, logits, model)
    old_threshold = read(out / 'threshold_selection.json')
    require(selected == old_threshold['selected_threshold'] and old_threshold['selection_split'] == 'validation', 'Threshold not selected from validation')
    old_trials = csvread(out / 'threshold_search.csv') if model == 'bert' else old_threshold['trials']
    require([float(r['threshold']) for r in old_trials] == CANDIDATES, 'Changed threshold grid')
    for a, b in zip(trials, old_trials):
        require(abs(a['macro_f1'] - float(b['macro_f1'])) < 1e-10, 'Changed threshold search scores')
    pred_signatures = {}
    for split in (['validation', 'test'] if full else ['validation']):
        y, z, pids = load_predictions(out / (split + '_predictions.npz'))
        if full:
            require(len(pids) == SIZES[split], 'Prediction split size mismatch')
            if data is not None:
                require(pids.tolist() == data[split][1] and np.array_equal(y, data[split][2]), 'Prediction IDs/gold differ from official split')
        pred_signatures[split] = {'ids_sha256': objsha(pids.tolist()),
                                 'gold_sha256': hashlib.sha256(y.astype(np.uint8).tobytes()).hexdigest()}
        if split == 'test':
            for name, t in [('fixed_0.30', .3), ('validation_selected', selected)]:
                measured = score(y, probabilities(z, model), t)[0]
                original = read(out / 'test_metrics.json')[name] if model == 'bert' else read(out / ('test_' + name + '_metrics.json'))
                close_metrics(measured, original)
                if model == 'bert':
                    rare = read(out / 'rare_label_metrics.json')['test'][name]
                    require(abs(measured['rare_f1'] - rare['mean_rare_label_f1']) < 1e-10, 'Rare F1 differs')
                    require(abs(measured['nonrare_f1'] - rare['mean_non_rare_label_f1']) < 1e-10, 'Nonrare F1 differs')
    samples = read(out / 'sample_ids.json')
    if full and data is not None:
        require(samples['train'] == data['train'][1] and samples['validation'] == data['validation'][1], 'Training/validation sample IDs changed')
    return {'status': 'LEGACY_ARTIFACTS_PASS', 'model': model, 'seed': seed, 'phase': phase,
            'run_directory': str(out), 'job_id': str(m['slurm_job_id']), 'success_sha256': sha(out / 'SUCCESS.json'),
            'artifacts_count': len(hashes), 'best_epoch': best['epoch'],
            'best_validation_macro_f1_fixed_0.30': chosen['validation_macro_f1'],
            'selected_threshold': selected, 'checkpoint_reload_evidence': reload_evidence,
            'prediction_signatures': pred_signatures, 'config': cfg}
