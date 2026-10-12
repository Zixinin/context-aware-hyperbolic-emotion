"""Read-only Ubuntu protocol/source/data/baseline gates; no new data downloads."""
import ast
import csv
import json
from pathlib import Path
import numpy as np
from b4lib import (NAMES, RARE, SIZES, REVISION, FROZEN_CFG_SHA, RUN_SHA,
    MODEL_SHA, objsha, read, require, sha, signature)
from hierarchy_adapter import load_hierarchy
from loop_adapter import derive
from evaluate import metrics, probabilities

EXPECTED_SOURCES_SHA = 'e8987c34511a4afc47d59c5046563dfab46edaa2c4f1c7938136ff042a63fbdd'
DATASET = {'repository': 'google-research-datasets/go_emotions',
    'configuration': 'simplified', 'revision': 'add492243ff905527e67aeb8b80c082af02207c3'}
CANONICAL_TSV_SHA = {
    'train.tsv': '1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5',
    'dev.tsv': '575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62',
    'test.tsv': '0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4',
}

def canonical_records(path):
    """CSV-style quoted TSV decoding, exactly matching the frozen Exp2 records.

    QUOTE_NONE incorrectly retains enclosing/doubled quotes in some official
    texts. This is a read-only decoding fix, NOT a transformation of training
    inputs: the returned full-row signature must match the existing manifest.
    """
    with Path(path).open(newline='') as stream:
        rows = list(csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_MINIMAL))
    decoded = []
    for index, row in enumerate(rows):
        require(len(row) == 3, f'Malformed canonical TSV row {index + 1}')
        decoded.append({'text': row[0], 'labels': [int(i) for i in row[1].split(',')], 'id': row[2]})
    return decoded

def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]

def verify_baseline(run, root, manifest, expected_sources):
    run, root = Path(run), Path(root)
    success, m = read(run / 'SUCCESS.json'), read(run / 'experiment_metadata.json')
    require(success['status'] == 'NORMALIZED_RUN_PASS' and success['model'] == 'lgsa'
        and success['seed'] == 42, 'Need the completed normalized LGSA seed42 baseline')
    hashes = success['artifacts_sha256']
    needed = {'experiment_metadata.json', 'config.json', 'test_predictions.npz',
        'validation_predictions.npz', 'checkpoint_info.json', 'validation_metrics_by_epoch.csv'}
    require(needed <= hashes.keys(), 'Baseline evidence incomplete')
    for name, wanted in hashes.items():
        require(Path(name).name == name and sha(run / name) == wanted, 'Old baseline artifact changed: ' + name)
    cfg = read(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json')
    require(m['status'] == 'COMPLETED' and m['model'] == 'lgsa'
        and m['seed'] == 42 and m['phase'] == 'full', 'Wrong baseline identity')
    require(m['slurm_state'] == 'COMPLETED' and m['slurm_exit_code'] == '0:0', 'Historical baseline completion missing')
    require(m['native_success_sha256'] == success['native_success_sha256'], 'Baseline lineage mismatch')
    require(m['config'] == cfg == read(run / 'config.json'), 'Baseline config mismatch')
    require(m['legacy_source_sha256'] == expected_sources, 'Baseline trainer provenance mismatch')
    require(m['lgsa_patch_commit'] == '209f83762dae1b354ba40e68b6952482ecf96208', 'Wrong baseline patch')
    require(m['hierarchy_sha256'] == manifest['hierarchy_sha256'], 'Old hierarchy mismatch')
    require(m['raw_model_repo'] == 'google-bert/bert-base-cased'
        and m['raw_model_revision'] == REVISION, 'Wrong historical BERT backbone')
    require({k: m[v] for k, v in [('repository', 'dataset'), ('configuration', 'dataset_config'),
        ('revision', 'dataset_revision')]} == DATASET, 'Baseline data pin mismatch')
    require(m['dataset_split_sizes'] == SIZES and m['num_labels'] == 28
        and m['label_names'] == NAMES and m['rare_ids'] == RARE, 'Baseline labels/splits mismatch')
    require(m['fixed_threshold'] == .30 and m['threshold_candidates'] == cfg['threshold_candidates'], 'Baseline threshold changed')
    require(m['dataset_signatures'] == manifest['dataset_signatures'], 'Prepared records differ from historical baseline')
    require(m['checkpoint_reload_verified'] is True and m['checkpoint_reload_evidence'], 'Baseline reload proof missing')
    with (run / 'validation_metrics_by_epoch.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    require([int(r['epoch']) for r in history] == [1, 2, 3, 4], 'Baseline epochs incomplete')
    chosen = max(history, key=lambda r: float(r['validation_macro_f1']))
    checkpoint = read(run / 'checkpoint_info.json')
    require(int(chosen['epoch']) == m['best_epoch'] == checkpoint['best_epoch']
        and checkpoint['selection'] == cfg['checkpoint_selection'], 'Baseline checkpoint selection mismatch')
    arrays = {}
    for split in ('validation', 'test'):
        with np.load(run / (split + '_predictions.npz'), allow_pickle=False) as saved:
            y, z, ids = [saved[k].copy() for k in ('gold', 'logits', 'ids')]
        require(y.shape == z.shape == (SIZES[split], 28) and ids.shape == (SIZES[split],), 'Baseline array dimensions wrong')
        require(np.isin(y, [0, 1]).all() and np.isfinite(z).all(), 'Invalid baseline arrays')
        sig = manifest['dataset_signatures'][split]
        import hashlib
        require(objsha(ids.tolist()) == sig['ids_sha256']
            and hashlib.sha256(y.astype(np.uint8).tobytes()).hexdigest() == sig['gold_sha256'], 'Baseline GT/IDs differ')
        arrays[split] = (y, z, ids)
    score = metrics(arrays['test'][0], probabilities(arrays['test'][1]), .30)[0]
    recorded = read(run / 'test_fixed_0.30_metrics.json')
    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'):
        require(abs(score[key] - recorded[key]) < 1e-10, 'Old metrics cannot be reproduced: ' + key)
    return {'status': 'EXP2_SEED42_BASELINE_VERIFIED', 'success_sha256': sha(run / 'SUCCESS.json'),
        'metadata_sha256': sha(run / 'experiment_metadata.json'), 'prediction_sha256': sha(run / 'test_predictions.npz'),
        'recomputed_fixed_0_30_metrics': score,
        'scope': 'Local normalized artifacts verified; historical Slurm completion recorded, no remote HPC operations'}

def audit(repo, root, tree_path, baseline):
    repo, root, tree_path, baseline = map(lambda p: Path(p).resolve(), (repo, root, tree_path, baseline))
    require(root == repo / 'experiments/exp2_lgsa/LGSA-GoEmotions', 'Use the verified runtime, not code/ or another model copy')
    require(tree_path == repo / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json', 'Use approved primary tree only')
    require(baseline == repo / 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42', 'Use the verified paired seed42 baseline')
    frozen = repo / 'experiments/exp2_1_multiseed/frozen'
    require(sha(frozen / 'expected_sources.json') == EXPECTED_SOURCES_SHA, 'Frozen source manifest changed')
    require(sha(frozen / 'lgsa.json') == FROZEN_CFG_SHA, 'Frozen config changed')
    expected = read(frozen / 'expected_sources.json')['lgsa']
    source_records = {}
    for name, wanted in expected.items():
        p = root / 'goemotions_adaptation' / name
        require(p.is_file() and sha(p) == wanted, 'Frozen Exp2 adapter changed: ' + name)
        source_records['adapter/' + name] = wanted
    require(sha(root / 'goemotions_adaptation/run.py') == RUN_SHA, 'Unexpected trainer')
    for name, wanted in MODEL_SHA.items():
        p = root / 'working_lgsa' / name
        require(p.is_file() and sha(p) == wanted, 'Frozen LGSA model changed: ' + name)
        source_records['model/' + name] = wanted
    cfg = read(frozen / 'lgsa.json')
    require(cfg == read(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json'), 'Config copies differ')
    require(read(frozen / 'hierarchy.json') == read(root / 'goemotions_adaptation/hierarchy.json'), 'Old tree copies differ')
    require(cfg['text_max_length'] == 50 and cfg['total_max_length'] == 54, 'Old text/prompt protocol changed')
    manifest_path = root / 'assets/goemotions-v0/manifest.json'
    manifest = read(manifest_path)
    require(manifest['status'] == 'GOEMOTIONS_ASSETS_READY' and manifest['dataset'] == DATASET
        and manifest['split_sizes'] == SIZES and manifest['rare_ids'] == RARE, 'Wrong prepared assets')
    require(manifest['config_sha256'] == FROZEN_CFG_SHA
        and manifest['hierarchy_sha256'] == sha(root / 'goemotions_adaptation/hierarchy.json'), 'Asset config/hierarchy changed')
    all_ids, canonical_sha = set(), {}
    canonical = repo / '数据集/GoEmotions/processed'
    for split, size in SIZES.items():
        p = root / 'assets/goemotions-v0' / (split + '.jsonl')
        require(sha(p) == manifest['files_sha256'][p.name], 'Prepared split bytes changed: ' + split)
        data = records(p)
        sig = signature(data)
        require(sig == manifest['dataset_signatures'][split] and len(data) == size, 'Prepared split signature changed')
        ids = {r['id'] for r in data}
        require(not all_ids & ids, 'Train/dev/test leakage')
        all_ids |= ids
        tsv = canonical / ('dev.tsv' if split == 'validation' else split + '.tsv')
        require(sha(tsv) == CANONICAL_TSV_SHA[tsv.name], 'Canonical TSV bytes changed: ' + tsv.name)
        raw = canonical_records(tsv)
        require(len(raw) == size, 'Canonical TSV size differs')
        require(signature(raw) == manifest['dataset_signatures'][split], 'Canonical TSV full-row/ID/GT/order signatures differ from Exp2')
        for index, (row, record) in enumerate(zip(raw, data)):
            require(row == record, f'Canonical TSV differs from Exp2 record/order: {split}, row {index + 1}, id={row["id"]}')
        canonical_sha[tsv.name] = sha(tsv)
    raw = manifest['model']
    require(raw['repo_id'] == 'google-bert/bert-base-cased' and raw['revision'] == REVISION, 'Wrong BERT pin')
    model_path = Path(raw['path']).resolve()
    require(model_path == repo / 'assets/pretrained/bert-base-cased' / REVISION, 'Use the approved raw snapshot, not a checkpoint')
    historical_raw = read(frozen / 'bert_archive_audit.json')['environment']['assets_manifest']['model_sha256']
    for name, wanted in historical_raw.items():
        require(raw['sha256'].get(name) == wanted and sha(model_path / name) == wanted, 'Raw BERT file changed: ' + name)
    h = load_hierarchy(tree_path)
    success = read(tree_path.parent / 'SUCCESS.json')
    require(success['status'] == 'EXP4A_COMPLETE_VERIFIED' and success['leakage'] == 'ZERO'
        and success['recommended_artifact'] == tree_path.name, 'Exp4A provenance missing')
    # Verify approved tree is still in the original Exp4A manifest, not merely a filename.
    original_manifest = tree_path.parent / 'ARTIFACT_MANIFEST.json'
    require(sha(original_manifest) == success['artifact_manifest_sha256'], 'Exp4A original manifest changed')
    entries = [r for r in read(original_manifest)['files'] if r['filename'] == tree_path.name]
    require(len(entries) == 1 and entries[0]['sha256'] == sha(tree_path)
        and entries[0]['bytes'] == tree_path.stat().st_size, 'Primary tree not in original Exp4A manifest')
    baseline_proof = verify_baseline(baseline, root, manifest, expected)
    source = (root / 'goemotions_adaptation/run.py').read_text()
    old_tree, _, _ = derive(source, 32, 2)
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    require(ast.dump(old_tree.body[0]) == ast.dump(original), 'Generic loop no longer reproduces old train AST')
    new_cfg = dict(cfg, total_max_length=68)
    derived_keys = [k for k in cfg if cfg[k] != new_cfg[k]]
    require(derived_keys == ['total_max_length'], 'Unexpected formal hyperparameter changes')
    comparison = {'status': 'EXP4B_CONFIG_COMPARISON_PASS', 'old_config': cfg, 'new_config': new_cfg,
        'intended_experimental_variable': 'hierarchy replacement',
        'independent_hyperparameter_changes': [], 'derived_config_changes': derived_keys,
        'old_hierarchy_depth': 2, 'new_hierarchy_depth': 9,
        'old_total_input_length': 54, 'new_total_input_length': 68,
        'text_window_remains': 50, 'final_prediction_labels_remain': 28,
        'old_hierarchy_supervision_nodes': 32, 'new_hierarchy_supervision_nodes': 56,
        'old_internal_nodes_excluding_ROOT': 4, 'new_internal_nodes_excluding_ROOT': 28,
        'sequence_length_change': 'hierarchy-induced deterministic/derived sequence-length change',
        'formula': 'total_input_length = text_window + 2 * hierarchy_depth',
        'fine_label_ids': dict(enumerate(NAMES)), 'dropout': 'unchanged raw BERT config and upstream model source',
        'early_stopping': 'none: original four epochs; select best validation Macro-F1 at fixed0.30',
        'secondary_threshold': 'retain original Exp2 validation-only grid as auxiliary reporting; never used to choose tree',
        'loss': 'unchanged MLM CE + multi-label categorical CE + original ramped HGA; internal targets determined by tree',
        'warning': 'Not an edge-only ablation: full hierarchy determines node count, prompts, ancestor targets and sequence length'}
    return h, manifest, new_cfg, comparison, baseline_proof, {
        'source_records': source_records, 'dataset_manifest_sha256': sha(manifest_path),
        'canonical_tsv_sha256': canonical_sha, 'hierarchy_sha256': sha(tree_path),
        'canonical_tsv_parser': 'csv.reader: delimiter=TAB, quoting=QUOTE_MINIMAL; exact full-row signatures required',
        'exp4a_success_sha256': sha(tree_path.parent / 'SUCCESS.json'),
        'baseline_success_sha256': sha(baseline / 'SUCCESS.json')}

def expected_lgsa_backbone_keys():
    """Pinned 12-layer BertModel(add_pooling_layer=False): 5 + 12*16 = 197.

    The original LGSA Prompt never constructs a pooler. Standard BERT's two
    pooler tensors must NOT be added to LGSA to satisfy a checker.
    """
    names = {'embeddings.' + name for name in ('word_embeddings.weight',
        'position_embeddings.weight', 'token_type_embeddings.weight',
        'LayerNorm.weight', 'LayerNorm.bias')}
    blocks = ('attention.self.query', 'attention.self.key', 'attention.self.value',
        'attention.output.dense', 'attention.output.LayerNorm',
        'intermediate.dense', 'output.dense', 'output.LayerNorm')
    names.update(f'encoder.layer.{layer}.{block}.{parameter}'
        for layer in range(12) for block in blocks for parameter in ('weight', 'bias'))
    return names

def verify_raw(model, manifest):
    import torch
    from safetensors import safe_open
    path = Path(manifest['model']['path']) / 'model.safetensors'
    require(path.parent.name == REVISION, 'Wrong raw BERT revision path')
    require(sha(path) == manifest['model']['sha256']['model.safetensors'], 'Raw BERT file hash changed')
    state = model.bert.state_dict()
    expected = expected_lgsa_backbone_keys()
    require(getattr(model.bert, 'pooler', None) is None, 'Original LGSA must not contain a pooler')
    require(set(state) == expected,
        f'Unexpected LGSA backbone schema: count={len(state)}, expected=197; '
        f'missing={sorted(expected - set(state))}; extra={sorted(set(state) - expected)}')
    count, aliases = 0, []
    with safe_open(str(path), framework='pt', device='cpu') as raw:
        keys = set(raw.keys())
        for name, value in state.items():
            candidates = ['bert.' + name, name]
            for modern, legacy in [('weight', 'gamma'), ('bias', 'beta')]:
                if name.endswith('.LayerNorm.' + modern):
                    old = name[:-len(modern)] + legacy
                    candidates += ['bert.' + old, old]
            key = next((k for k in candidates if k in keys), None)
            require(key is not None, 'Missing raw backbone tensor: ' + name)
            ref = raw.get_tensor(key)
            require(value.dtype == ref.dtype and tuple(value.shape) == tuple(ref.shape)
                and torch.equal(value.detach().cpu(), ref), 'Backbone is not freshly RAW initialized: ' + name)
            if key.endswith(('.LayerNorm.gamma', '.LayerNorm.beta')):
                aliases.append(key)
            count += 1
    require(count == 197 and len(set(aliases)) == len(aliases) == 50,
        f'Unexpected LGSA raw tensor/alias counts: tensors={count}, legacy_aliases={len(aliases)}; expected=197/50')
    return {'status': 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED', 'tensors_compared': count,
        'backbone_architecture': 'BertModel(add_pooling_layer=False), unchanged original LGSA',
        'pooler_present': False, 'expected_backbone_tensors': 197,
        'legacy_LayerNorm_aliases_verified': aliases, 'raw_revision': REVISION,
        'raw_weight_sha256': manifest['model']['sha256']['model.safetensors'],
        'trained_checkpoint_loaded': False, 'smoke_weights_loaded': False}
