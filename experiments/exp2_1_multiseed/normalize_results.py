"""Create new standardized views from immutable original/native run artifacts."""
import hashlib
import shutil
import numpy as np
from mslib import *
from metrics_ms import load_predictions, probabilities, threshold_search, score
from evidence_ms import verify_legacy, validation_history


def normalize(entry, audit, destination):
    model, seed = entry['model'], entry['seed']
    src = Path(entry['run_directory'])
    require(sha(src / 'SUCCESS.json') == entry['success_sha256'], 'Native success marker changed')
    checked = verify_legacy(model, src, seed)
    frozen_data = audit['models'][model]['dataset_signatures']
    for split, signature in checked['prediction_signatures'].items():
        require(all(signature[k] == frozen_data[split][k] for k in signature), 'Predictions are not the audited official split')
    metadata = read(src / ('environment.json' if model == 'bert' else 'experiment_metadata.json'))
    if entry['decision'] == 'REUSED_VERIFIED':
        require(seed == 42 and audit['models'][model]['decision'] == 'REUSED_VERIFIED', 'Invalid reuse decision')
        initialization = {'evidence': audit['models'][model]['raw_initialization_evidence'],
                          'historical_tensor_digest_available': model == 'lgsa'}
        if model == 'lgsa':
            initialization.update({k: metadata[k] for k in ['initial_backbone_sha256', 'initial_model_sha256']})
        runtime = audit['models'][model]['environment']
        runtime = {**runtime, 'scope': 'Dependency versions verified against historical freeze; runtime defaults measured at audit, not during historical training',
                   'historical_flags_source': 'Hash-verified original training source; BERT explicitly deterministic/TF32 off, LGSA preserves original torch defaults'}
        timing = {'start_time': metadata['start_time'], 'end_time': metadata['end_time'],
                  'peak_gpu_memory_bytes': None, 'peak_memory_note': 'Not recorded per phase in historical run'}
    else:
        require(sha(src / 'RUN_PASS.json') == entry['run_pass_sha256'], 'Run evidence changed')
        proof = read(src / 'RUN_PASS.json')
        require(proof['status'] == 'RUN_PHASE_PASS' and proof['phase'] == 'full', 'Missing full run evidence')
        identity = proof['identity']
        require(identity['model'] == model and identity['seed'] == seed and identity['job_id'] == entry['job_id'], 'Run identity mismatch')
        require(identity['release_sha256'] == release() and identity['audit_sha256'] == sha(ROOT / 'audit/SEED42_AUDIT_PASS.json'), 'Run code/audit lineage mismatch')
        initialization, runtime = proof['raw_initialization'], proof['runtime_environment']
        require(initialization['status'] == 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED', 'Missing raw fingerprint')
        require(sha(src / 'raw_initialization.json') == proof['raw_initialization_sha256'], 'Raw fingerprint evidence changed')
        timing = {k: proof[k] for k in ['started_at', 'ended_at', 'elapsed_seconds', 'peak_gpu_memory_bytes']}
    destination = Path(destination)
    require(not destination.exists(), 'Never overwrite a normalized scientific result')
    temporary = destination.with_name(destination.name + '.partial')
    temporary.mkdir(parents=True, exist_ok=False)
    write(temporary / 'config.json', checked['config'])
    write(temporary / 'environment.json', {'runtime': runtime, 'legacy_metadata': metadata})
    native_success = read(src / 'SUCCESS.json')
    checkpoint_file = 'checkpoints/best/model.safetensors' if model == 'bert' else 'checkpoint_best.pt'
    checkpoint = {'path': str(src / checkpoint_file), 'sha256': native_success['artifacts_sha256'][checkpoint_file],
                  'best_epoch': checked['best_epoch'], 'selection': 'validation_macro_f1_at_fixed_0.30',
                  'best_validation_macro_f1_fixed_0.30': checked['best_validation_macro_f1_fixed_0.30'],
                  'reload_evidence': checked['checkpoint_reload_evidence']}
    write(temporary / 'checkpoint_info.json', checkpoint)
    if model == 'bert':
        shutil.copy2(src / 'training_log.csv', temporary / 'training_log.csv')
    else:
        rows = [json.loads(line) for line in (src / 'steps.jsonl').read_text().splitlines()]
        for r in rows:
            r['hga_components_upstream'] = json.dumps(r['hga_components_upstream'], sort_keys=True)
        csvwrite(temporary / 'training_log.csv', rows)
        shutil.copy2(src / 'geometry_acceptance.json', temporary / 'geometry_acceptance.json')
    csvwrite(temporary / 'validation_metrics_by_epoch.csv', validation_history(model, src))
    val_y, val_z, val_ids = load_predictions(src / 'validation_predictions.npz')
    selected, trials = threshold_search(val_y, val_z, model)
    csvwrite(temporary / 'threshold_search.csv', trials)
    write(temporary / 'threshold_selection.json', {'selected_threshold': selected, 'selection_split': 'validation',
          'objective': 'macro_f1', 'tie_break': 'smallest threshold', 'candidates': CANDIDATES})
    rare = {'label_ids': RARE, 'labels': [NAMES[i] for i in RARE], 'definition': 'Bottom 25% by full TRAIN frequency; frozen for all seeds'}
    for split in ['validation', 'test']:
        y, z, ids = load_predictions(src / (split + '_predictions.npz'))
        prob = probabilities(z, model)
        np.savez_compressed(temporary / (split + '_predictions.npz'), ids=ids, gold=y, logits=z, probabilities=prob)
        if split == 'test':
            for name, t in [('fixed_0.30', .3), ('validation_selected', selected)]:
                metrics, labels = score(y, prob, t)
                write(temporary / ('test_' + name + '_metrics.json'), metrics)
                csvwrite(temporary / ('per_label_' + name + '_metrics.csv'), labels)
                rare[name] = {'rare_f1': metrics['rare_f1'], 'nonrare_f1': metrics['nonrare_f1']}
    write(temporary / 'rare_label_metrics.json', rare)
    completed = entry['decision'] == 'REUSED_VERIFIED'
    record = {'status': 'COMPLETED' if completed else 'AWAITING_SLURM_COMPLETION',
        'experiment': 'FYP Experiment 2.1 — Multi-Seed Stability Verification',
        'model': model, 'seed': seed, 'phase': 'full', 'seed42_decision': entry['decision'], 'decision_reason': entry['reason'],
        'native_run_directory': str(src), 'native_success_sha256': entry['success_sha256'],
        'slurm_job_id': entry['job_id'], 'slurm_state': 'COMPLETED' if completed else 'RUNNING',
        'slurm_exit_code': '0:0' if completed else None,
        'raw_model_repo': 'google-bert/bert-base-cased', 'raw_model_revision': REVISION,
        'dataset': 'google-research-datasets/go_emotions', 'dataset_config': 'simplified',
        'dataset_revision': 'add492243ff905527e67aeb8b80c082af02207c3', 'dataset_split_sizes': SIZES,
        'dataset_signatures': frozen_data, 'num_labels': 28, 'label_names': NAMES, 'rare_ids': RARE,
        'threshold_candidates': CANDIDATES, 'fixed_threshold': .3, 'selected_threshold': selected,
        'best_epoch': checked['best_epoch'], 'best_validation_macro_f1_fixed_0.30': checked['best_validation_macro_f1_fixed_0.30'],
        'checkpoint_reload_verified': True, 'checkpoint_reload_evidence': checked['checkpoint_reload_evidence'],
        'initialization': initialization, 'config': checked['config'], 'resources': timing,
        'gpu': metadata.get('gpu_model', metadata.get('gpu')), 'number_of_gpus': 1,
        'legacy_git_commit': metadata.get('git_commit', metadata.get('adapter_source', {}).get('adaptation_commit')),
        'legacy_source_sha256': read(ROOT / 'frozen/expected_sources.json')[model],
        'orchestration_release_sha256': release(), 'orchestration_git_commit': read(ROOT / 'RELEASE.json').get('git_commit'),
        'lgsa_patch_commit': PATCH if model == 'lgsa' else None,
        'hierarchy_sha256': sha(ROOT / 'frozen/hierarchy.json') if model == 'lgsa' else None,
        'derived_at': now(), 'derivation': 'Metrics and counts from existing logits; no additional training or test inference'}
    write(temporary / 'experiment_metadata.json', record)
    write(temporary / 'SUCCESS.json', {'status': 'NORMALIZED_RUN_PASS' if completed else 'NORMALIZED_AWAITING_SLURM',
        'model': model, 'seed': seed,
        'native_success_sha256': entry['success_sha256'],
        'artifacts_sha256': {p.name: sha(p) for p in sorted(temporary.iterdir()) if p.is_file()}})
    temporary.rename(destination)
    return destination
