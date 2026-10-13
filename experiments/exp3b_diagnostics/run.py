"""Ubuntu-only read-only Exp3B diagnostics. Exp3B here does NOT mean training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

EXPECTED_HEAD = 'c87d5ccd0689c2c032f3a7126127bb0a54aabe78'
NTFY = 'https://ntfy.sh/zixin-fyp-4090-1119'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def notify(url, title, body):
    if not url:
        return
    try:
        request = urllib.request.Request(url, data=body.encode('utf-8'), headers={'Title': title})
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        print('NTFY_DELIVERED:', title, flush=True)
    except Exception as error:
        print('NTFY_UNAVAILABLE:', type(error).__name__, '(diagnostic continues)', flush=True)


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def snapshot(paths):
    result = {}
    for root in paths:
        require(root.exists(), 'Required existing input missing: ' + str(root))
        files = sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in files:
            require(not path.is_symlink(), 'Refuse symlink in protected inputs: ' + str(path))
            if path.is_file():
                result[str(path)] = sha(path)
    return result


def verify_success(directory, status, hashes):
    success = read(directory / 'SUCCESS.json')
    require(success['status'] == status, 'Unexpected success status: ' + str(directory))
    for name, expected in success['artifacts_sha256'].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), 'Unsafe artifact path')
        require(hashes.get(str(path)) == expected, 'Existing artifact hash mismatch: ' + str(path))
    return success


def audit(repo):
    import numpy as np
    import torch
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3', 'Run ONLY on Ubuntu D3020-A3')
    import transformers
    import torch_geometric
    require(Path(sys.prefix).resolve() == (repo / '.venv-lgsa').resolve(), 'Use the existing .venv-lgsa interpreter')
    require(transformers.__version__ == '4.44.2' and torch_geometric.__version__ == '2.6.1',
            'Transformer/PyG versions differ from the original Ubuntu environment')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == EXPECTED_HEAD, 'Git HEAD changed; inspect before diagnosis')
    require(torch.__version__ == '2.5.1+cu118', 'Use the original .venv-lgsa Torch environment')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    require(torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0), 'Unexpected GPU')
    require(torch.cuda.mem_get_info()[0] >= 8*1024**3, 'GPU has less than 8GiB free; inspect other processes')
    run = repo / 'outputs/exp3_dpt_peer/seed42_20261011'
    baseline = repo / 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42'
    previous = repo / 'outputs/exp3_diagnostics/seed42_20261011_v1'
    replay = repo / 'outputs/exp3_diagnostics/validation_epoch3_4_20261011_v1'
    lgsa = repo / 'experiments/exp2_lgsa/LGSA-GoEmotions'
    protected = [run, baseline, previous, replay]
    print('AUDIT_HASHING_EXISTING_RESULTS_AND_CHECKPOINTS', flush=True)
    hashes = snapshot(protected)
    success = verify_success(run, 'EXP3_FULL_PASS', hashes)
    verify_success(baseline, 'NORMALIZED_RUN_PASS', hashes)
    verify_success(previous, 'EXP3_EXISTING_EVIDENCE_DIAGNOSTIC_PASS', hashes)
    verify_success(replay, 'EXP3_VALIDATION_EPOCH3_4_REPLAY_PASS', hashes)
    required = {run: ['checkpoint_best.pt', 'checkpoint_resume.pt', 'test_predictions.npz',
        'validation_predictions.npz', 'steps.jsonl', 'config.json', 'initial_geometry.pt',
        'environment.json', 'model_data/value_dict.pt', 'model_data/slot.pt', 'sample_ids.json',
        'test_fixed_0.30_metrics.json', 'validation_epoch_3.json', 'validation_epoch_4.json',
        'sibling_aggregate_comparison.json', 'sibling_pair_comparison.csv', 'training_log.csv',
        'per_label_fixed_0.30.csv', 'exp2_vs_exp3_seed42.csv'],
        baseline: ['test_predictions.npz', 'test_fixed_0.30_metrics.json', 'experiment_metadata.json'],
        previous: ['diagnostic.json', 'loss_by_epoch.csv', 'gradient_diagnostic.json'],
        replay: ['validation_epoch_3_predictions.npz', 'validation_epoch_4_predictions.npz',
                 'replay_report.json', 'validation_epoch_3_metrics.json', 'validation_epoch_4_metrics.json']}
    for root, filenames in required.items():
        proof = read(root / 'SUCCESS.json')['artifacts_sha256']
        for filename in filenames:
            require(filename in proof and str(root / filename) in hashes, 'Required artifact missing: ' + str(root / filename))
    sys.path[:0] = [str(repo / 'experiments/exp3_dpt_peer'), str(lgsa / 'goemotions_adaptation'), str(lgsa / 'working_lgsa')]
    from protocol import package_hashes, source_audit, assets
    from baseline import verify_baseline
    from hierarchy_goemotions import load_hierarchy
    from data_goemotions import rows
    require(package_hashes() == success['identity']['package_sha256'], 'Frozen Exp3 package changed')
    require(source_audit(lgsa) == success['identity']['baseline_sources'], 'Frozen LGSA code changed')
    require(success['best_epoch'] == 3 and success['optimizer_steps'] == 10856, 'Unexpected completed run identity')
    cfg = read(run / 'config.json')
    require(cfg == success['config'] and cfg['epochs'] == 4 and cfg['seed'] == 42 and cfg['fixed_threshold'] == .30,
            'Config identity differs')
    require(cfg['peer_loss_enabled'] and cfg['peer_lambda'] == .5 and cfg['peer_temperature'] == .1 and
            cfg['peer_top_k'] == 1, 'Peer protocol differs from the completed pilot')
    manifest = assets(lgsa)
    require(sha(lgsa / 'assets/goemotions-v0/test.jsonl') == manifest['files_sha256']['test.jsonl'],
            'Prepared test data bytes differ from the original manifest')
    require(sha(lgsa / 'assets/goemotions-v0/manifest.json') == success['dataset_manifest_sha256'], 'Dataset manifest changed')
    baseline_proof = verify_baseline(baseline, lgsa, manifest)
    additional = [lgsa / 'assets/goemotions-v0', Path(manifest['model']['path'])]
    hashes.update(snapshot(additional))
    hierarchy = load_hierarchy(lgsa / 'goemotions_adaptation/hierarchy.json')
    labels = sorted(hierarchy['source']['labels'], key=lambda r: r['fine_id'])
    train = rows(lgsa / 'assets/goemotions-v0/train.jsonl')
    require(len(train) == 43410, 'Training row count differs')
    require([r['id'] for r in train] == read(run / 'sample_ids.json')['train'], 'Recorded training sample order differs')
    frequency = np.array([sum(i in r['labels'] for r in train) for i in range(28)])
    own = Path(__file__).resolve().parent
    source_hashes = {p.name: sha(p) for p in sorted(own.iterdir()) if p.suffix in ('.py', '.md')}
    delivered = read(own / 'SOURCE_MANIFEST.json')
    require(source_hashes == delivered['source_sha256'], 'Independent diagnostic package differs from delivered manifest')
    return dict(repo=repo, run=run, baseline=baseline, previous=previous, replay=replay, lgsa=lgsa,
                protected=protected+additional, hashes=hashes, success=success, cfg=cfg, hierarchy=hierarchy,
                labels=labels, train=train, frequency=frequency, baseline_proof=baseline_proof,
                reused_artifacts=[str(root / name) for root, names in required.items() for name in names],
                source_hashes=source_hashes, git_head=head,
                git_status=subprocess.check_output(['git', '-C', str(repo), 'status', '--short', '--branch'], text=True))


def reconcile(metrics, saved, rare_f1=None):
    import math
    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'macro_f1'):
        require(math.isclose(metrics[key], saved[key], rel_tol=0, abs_tol=1e-12), 'Saved metric mismatch: ' + key)
    if rare_f1 is not None:
        require(math.isclose(rare_f1, saved['rare_mean_f1'], rel_tol=0, abs_tol=1e-12), 'Saved rare F1 mismatch')


def diagnose(context, out):
    import numpy as np
    import torch
    from analysis import (load_predictions, matched, per_label, branches, gain_summary, frequency_correlations,
        sibling_correlations, score_shift, activity, example_groups, eligibility_labels, error_context,
        rare_details, csvwrite, jsonwrite, RARE)
    from probe import run_probes
    from report import figures, write_report
    c = context
    labels, parents = c['labels'], [r['coarse_id'] for r in c['labels']]
    jsonwrite(out / 'provenance_before.json', {'input_sha256': c['hashes'], 'baseline_proof': c['baseline_proof'],
        'diagnostic_sources_sha256': c['source_hashes'], 'git_head': c['git_head'], 'git_status': c['git_status'],
        'exact_reused_directories': [str(p) for p in c['protected']],
        'reused_artifacts': c['reused_artifacts'],
        'scope': 'Independent diagnostic only; saved test scores are explanatory, never selection/development.'})
    a = load_predictions(c['baseline'] / 'test_predictions.npz', 5427)
    b = load_predictions(c['run'] / 'test_predictions.npz', 5427)
    matched(a, b)
    y, z0, z1 = a['gold'], a['logits'], b['logits']
    # sample_ids.json records TRAIN/VALIDATION only. Baseline provenance verifies
    # official TEST IDs/GT; matched() above transfers that proof to Exp3.
    rows = per_label(y, z0, z1, labels, c['frequency'])
    branch = branches(rows)
    gains, freq = gain_summary(rows), frequency_correlations(rows)
    csvwrite(out / 'per_label_exp2_vs_exp3.csv', sorted(rows, key=lambda r: r['delta_f1']))
    csvwrite(out / 'per_label_exp2_vs_exp3_descending.csv', sorted(rows, key=lambda r: -r['delta_f1']))
    csvwrite(out / 'branch_level_exp2_vs_exp3.csv', branch)
    jsonwrite(out / 'per_label_gain_loss_summary.json', gains)
    jsonwrite(out / 'frequency_vs_effect.json', freq)
    # Restore the flags recorded for original forward numerical behaviour.
    environment = read(c['run'] / 'environment.json')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = environment['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = environment['cudnn_tf32']
    torch.use_deterministic_algorithms(environment['deterministic_algorithms'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    actual, true_loss, parent_loss, negative_loss, pair_loss, gradient = run_probes(c['repo'], c['lgsa'], c['run'],
         c['cfg'], c['hierarchy'], labels, c['train'], out, c['frequency'], state_digest)
    shifted = score_shift(y, z0, z1, labels)
    active = activity(y, z0, z1)
    import csv
    with (c['run'] / 'exp2_vs_exp3_seed42.csv').open() as stream:
        for row in csv.DictReader(stream):
            metric = row['metric']
            if metric in active['delta']:
                for tag in ('exp2', 'exp3', 'delta'):
                    require(abs(active[tag][metric]-float(row[tag])) < 1e-12, 'Saved Exp2/3 comparison mismatch: '+metric)
    with (c['run'] / 'per_label_fixed_0.30.csv').open() as stream:
        original_labels = list(csv.DictReader(stream))
    require(len(original_labels) == 28, 'Saved per-label table incomplete')
    for ours, original in zip(rows, original_labels):
        require(ours['label'] == original['label'], 'Saved per-label order differs')
        for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'prediction_count'):
            require(abs(ours['exp3_'+key]-float(original[key])) < 1e-12, 'Saved per-label metric differs: '+key)
    for tag, root in [('exp2', c['baseline']), ('exp3', c['run'])]:
        reconcile(active[tag], read(root / 'test_fixed_0.30_metrics.json'),
                  float(np.mean([r[tag+'_f1'] for r in rows if r['is_rare']])))
    csvwrite(out / 'per_label_score_shift.csv', shifted)
    jsonwrite(out / 'prediction_activity_summary.json', active)
    groups = example_groups(y, z0, z1, parents)
    csvwrite(out / 'single_vs_multilabel_comparison.csv', [r for r in groups if r['subset'] in ('single_gt', 'multi_gt')])
    csvwrite(out / 'peer_eligible_examples_comparison.csv', [r for r in groups if r['subset'] not in ('single_gt', 'multi_gt')])
    csvwrite(out / 'peer_eligible_labels_comparison.csv', eligibility_labels(y, z0, z1, parents))
    sibling_rows = [r for r in rows if r['structurally_peer_eligible']]
    csvwrite(out / 'sibling_metric_vs_f1.csv', sibling_rows)
    sibling_corr = sibling_correlations(rows)
    jsonwrite(out / 'sibling_metric_correlation.json', sibling_corr)
    errors = {'exp2': error_context(y, z0, parents), 'exp3': error_context(y, z1, parents)}
    jsonwrite(out / 'classification_error_context.json', errors)
    history = [json.loads(line) for line in (c['run'] / 'steps.jsonl').read_text().splitlines()]
    require([r['step'] for r in history] == list(range(1, 10857)) and
            Counter(r['epoch'] for r in history) == {1: 2714, 2: 2714, 3: 2714, 4: 2714}, 'Incomplete training history')
    rare = rare_details(y, z0, z1, labels, rows, history, pair_loss)
    csvwrite(out / 'grief_relief_exp2_vs_exp3.csv', rare)
    v3 = load_predictions(c['replay'] / 'validation_epoch_3_predictions.npz', 5426, cosine=True)
    v4 = load_predictions(c['replay'] / 'validation_epoch_4_predictions.npz', 5426, cosine=True)
    matched(v3, v4)
    saved = load_predictions(c['run'] / 'validation_predictions.npz', 5426)
    matched(saved, v3)
    require(np.array_equal(saved['logits'], v3['logits']), 'Epoch3 replay logits differ from selected checkpoint evidence')
    require(v3['ids'].tolist() == read(c['run'] / 'sample_ids.json')['validation'], 'Validation identity differs')
    val_rows = per_label(v3['gold'], v3['logits'], v4['logits'], labels, c['frequency'],
                         v3['cosine_similarities'], v4['cosine_similarities'])
    val_branches = branches(val_rows)
    def rename(row):
        return {k.replace('exp2_', 'epoch3_').replace('exp3_', 'epoch4_'): v for k, v in row.items()}
    csvwrite(out / 'epoch3_vs_epoch4_validation_labels.csv', [rename(r) for r in val_rows])
    csvwrite(out / 'epoch3_vs_epoch4_validation_branches.csv', [rename(r) for r in val_branches])
    val_activity = activity(v3['gold'], v3['logits'], v4['logits'])
    for tag, epoch in [('exp2', 3), ('exp3', 4)]:
        reconcile(val_activity[tag], read(c['run'] / f'validation_epoch_{epoch}.json'),
                  float(np.mean([r[tag+'_f1'] for r in val_rows if r['is_rare']])))
    paradox = [r['label'] for r in val_rows if r['delta_f1'] < -1e-12 and
               r['delta_sibling_margin'] is not None and r['delta_sibling_margin'] > 0]
    validation = {'epoch3': val_activity['exp2'], 'epoch4': val_activity['exp3'],
                  'labels_margin_improved_f1_decreased': paradox,
                  'existing_replay_comparison': read(c['replay'] / 'replay_report.json')['comparison'],
                  'cosine_margin_improved_f1_decreased': [r['label'] for r in val_rows if r['delta_f1'] < -1e-12 and
                       r['delta_sibling_cosine_margin'] is not None and r['delta_sibling_cosine_margin'] > 0],
                  'checkpoint_selection_unchanged': True}
    jsonwrite(out / 'epoch3_vs_epoch4_validation_summary.json', validation)
    figures(out, rows, branch, shifted, true_loss, negative_loss, active, gradient, sibling_rows)
    summary = {'status': 'EXP3B_ANALYSIS_ONLY_DIAGNOSTIC_PASS', 'optimizer_steps': 0,
        'test_model_inference': False, 'main_metrics': active, 'gain_loss': gains, 'branches': branch,
        'frequency_correlations': freq, 'peer_loss_snapshot': actual, 'peer_loss_by_parent': parent_loss,
        'top10_hard_negatives_by_actual_loss': negative_loss[:10], 'gradient_scopes': gradient,
        'sibling_correlations': sibling_corr, 'error_context': errors, 'grief_relief': rare,
        'validation_epoch3_4': validation, 'original_outputs_and_checkpoints_unchanged': None,
        'limits': ['Single seed; checkpoint snapshot is not historical per-term loss or gradient trajectory.',
                   'Descriptive label-wise correlations are not significance tests or causal proof.',
                   'No test-based model development or new hyperparameter choice is permitted.']}
    # Full byte-level preservation of every existing output and checkpoint, plus input asset trees.
    print('FINAL_PROTECTED_INPUT_HASH_VERIFICATION', flush=True)
    require(snapshot(c['protected']) == c['hashes'], 'Protected inputs added/deleted/modified during diagnosis')
    from protocol import package_hashes, source_audit
    require(package_hashes() == c['success']['identity']['package_sha256'] and
            source_audit(c['lgsa']) == c['success']['identity']['baseline_sources'], 'Frozen sources changed during diagnosis')
    own = Path(__file__).resolve().parent
    require({p.name: sha(p) for p in sorted(own.iterdir()) if p.suffix in ('.py', '.md')} == c['source_hashes'],
            'Diagnostic source changed while running')
    summary['original_outputs_and_checkpoints_unchanged'] = True
    summary['protected_file_count'] = len(c['hashes'])
    recommendation = write_report(out, summary, rows, parent_loss, true_loss, negative_loss, val_rows)
    summary['NEXT_STEP_RECOMMENDATION'] = recommendation
    jsonwrite(out / 'diagnostic_summary.json', summary)
    jsonwrite(out / 'SUCCESS.json', {'status': summary['status'], 'optimizer_steps': 0, 'test_model_inference': False,
        'original_outputs_and_checkpoints_unchanged': True, 'protected_file_count': len(c['hashes']),
        'NEXT_STEP_RECOMMENDATION': recommendation,
        'artifacts_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        'finished_at': datetime.now(timezone.utc).isoformat()})
    print((out / 'EXP3B_DIAGNOSTIC_SUMMARY.md').read_text(), flush=True)
    print('EXP3B_ANALYSIS_ONLY_DIAGNOSTIC_PASS', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--no-notify', action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    repo = args.repo.resolve()
    out = repo / 'outputs/exp3b_diagnostics_seed42'
    url = '' if args.no_notify else os.environ.get('FYP_NTFY_URL', NTFY)
    notify(url, 'FYP Exp3B diagnostic started', 'Analysis ONLY; no training, tuning or optimizer steps. Audit first.')
    created = False
    try:
        require(not out.exists(), 'New output directory already exists; STOP, never overwrite')
        context = audit(repo)
        if args.audit_only:
            print('EXP3B_READ_ONLY_AUDIT_PASS', json.dumps({'host': platform.node(), 'git_head': context['git_head'],
                  'protected_files': len(context['hashes']), 'train_rows': len(context['train']),
                  'output_created': False, 'gpu_forward_passes': 0}), flush=True)
        else:
            out.mkdir(parents=True, exist_ok=False); created = True
            diagnose(context, out)
    except BaseException as error:
        if created:
            from analysis import jsonwrite
            jsonwrite(out / 'FAILURE.json', {'status': 'FAILED', 'error': str(error), 'optimizer_steps': 0,
                                            'note': 'No optimizer exists; partial output retained, do not overwrite.'})
        notify(url, 'FYP Exp3B diagnostic failed', 'Inspect log; no training performed. ' + str(error)[:160])
        raise
    notify(url, 'FYP Exp3B diagnostic finished',
           'Read-only audit passed; no forward passes.' if args.audit_only else
           'Diagnostic artifacts hash-verified; zero optimizer updates; inspect report. ' + str(out))


if __name__ == '__main__':
    main()
