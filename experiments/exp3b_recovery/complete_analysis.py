"""CPU-only completion of the audited Exp3B v1 rare-F1 schema failure.

The original diagnostic package, failed output and all experiment inputs are
read-only. GPU records are validated and copied to a NEW output directory;
no checkpoint load, model forward, autograd or optimizer is performed here.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request

NTFY = 'https://ntfy.sh/zixin-fyp-4090-1119'
MANIFEST_SHA = '73a71803e92b6da49ba3e56975b18c09e7fcea171b9b9360d49f1bd298c62608'
GPU_FILES = (
    'fixed_batch_protocol.json', 'peer_loss_terms_snapshot.csv',
    'peer_loss_batches_snapshot.csv', 'peer_loss_by_true_label.csv',
    'peer_loss_by_parent.csv', 'hard_negative_loss_contribution.csv',
    'peer_loss_by_pair_snapshot.csv', 'peer_loss_contribution_summary.json',
    'gradient_conflict_batches.csv', 'gradient_conflict_by_layer.csv',
    'gradient_conflict_summary.json', 'model_state_integrity.json',
)
INT_FIELDS = {'batch_index', 'examples', 'eligible_terms', 'eligible_terms_in_batch',
              'train_frequency', 'selected_count', 'shared_parameter_tensors'}
TEXT_FIELDS = {'sample_id', 'true_label', 'negative_label', 'parent', 'label',
               'scope', 'negative_selection_counts'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def notify(url, title, body):
    if not url:
        return
    try:
        request = urllib.request.Request(url, data=body.encode(), headers={'Title': title})
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        print('NTFY_DELIVERED:', title, flush=True)
    except Exception as error:
        print('NTFY_UNAVAILABLE:', type(error).__name__, '(recovery continues)', flush=True)


def csvread(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        result = []
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                require(key is not None and value is not None, 'Malformed CSV row')
                if key in TEXT_FIELDS:
                    converted[key] = value
                elif value == '':
                    converted[key] = None
                elif key in INT_FIELDS:
                    converted[key] = int(value)
                else:
                    converted[key] = float(value)
                    require(math.isfinite(converted[key]), 'Nonfinite CSV value: '+key)
            result.append(converted)
    require(bool(result), 'Empty saved GPU table: '+str(path))
    return result


def equivalent(a, b, location='record'):
    """JSON/CSV round-trip tolerance only; reject changed keys/counts/categories."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), 'Changed keys: '+location)
        for key in a:
            equivalent(a[key], b[key], location+'.'+key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), 'Changed rows: '+location)
        for index, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, location+'.'+str(index))
    elif isinstance(a, int) and not isinstance(a, bool) and isinstance(b, float) and location.rsplit('.', 1)[-1] not in INT_FIELDS:
        # sum([]) is integer 0 in the original aggregates, while a numerical
        # CSV loss column round-trips as 0.0. Counts still require exact ints.
        require(math.isfinite(b) and a == b, 'Changed numerical zero: '+location)
    elif isinstance(a, float):
        require(isinstance(b, (int, float)) and not isinstance(b, bool) and
                math.isfinite(a) and math.isfinite(b) and
                math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), 'Changed value: '+location)
    else:
        require(type(a) is type(b) and a == b, 'Changed value/type: '+location)


def rare_metric(saved):
    keys = [key for key in ('rare_mean_f1', 'rare_f1') if key in saved]
    require(bool(keys), 'Saved rare F1 missing: neither supported field exists')
    values = [float(saved[key]) for key in keys]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Invalid saved rare F1')
    require(all(math.isclose(v, values[0], rel_tol=0, abs_tol=1e-12) for v in values),
            'Conflicting rare F1 aliases')
    return values[0]


def reconcile(metrics, saved, rare_f1=None):
    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'macro_f1'):
        require(math.isclose(metrics[key], saved[key], rel_tol=0, abs_tol=1e-12),
                'Saved metric mismatch: '+key)
    if rare_f1 is not None:
        require(math.isclose(rare_f1, rare_metric(saved), rel_tol=0, abs_tol=1e-12),
                'Saved rare F1 mismatch')


def context_and_records(repo, legacy):
    import numpy as np
    require(platform.system() == 'Linux' and platform.node() == 'D3020-A3', 'Run only on Ubuntu D3020-A3')
    require(Path(sys.prefix).resolve() == (repo / '.venv-lgsa').resolve(), 'Use .venv-lgsa interpreter')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == legacy.EXPECTED_HEAD, 'Git HEAD changed; inspect first')
    partial = repo / 'outputs/exp3b_diagnostics_seed42'
    require(read(partial / 'FAILURE.json')['error'] == "'rare_mean_f1'", 'Unexpected failure; do not reuse blindly')
    require(not (partial / 'SUCCESS.json').exists(), 'Original diagnostic already has SUCCESS; inspect first')
    before = read(partial / 'provenance_before.json')
    run = repo / 'outputs/exp3_dpt_peer/seed42_20261011'
    baseline = repo / 'experiments/exp2_1_multiseed/outputs/job3006610/lgsa_goemotions/seed42'
    previous = repo / 'outputs/exp3_diagnostics/seed42_20261011_v1'
    replay = repo / 'outputs/exp3_diagnostics/validation_epoch3_4_20261011_v1'
    lgsa = repo / 'experiments/exp2_lgsa/LGSA-GoEmotions'
    raw = repo / 'assets/pretrained/bert-base-cased/cd5ef92a9fb2f889e972770a36d4ed042daf221e'
    protected = [run, baseline, previous, replay, lgsa / 'assets/goemotions-v0', raw]
    require(before['exact_reused_directories'] == [str(p) for p in protected], 'Protected input paths differ')
    print('CPU_RECOVERY_HASHING_ORIGINAL_INPUTS_AND_PARTIAL_OUTPUT', flush=True)
    hashes = legacy.snapshot(protected)
    require(hashes == before['input_sha256'], 'Input bytes changed since original GPU diagnostic')
    success = legacy.verify_success(run, 'EXP3_FULL_PASS', hashes)
    legacy.verify_success(baseline, 'NORMALIZED_RUN_PASS', hashes)
    legacy.verify_success(previous, 'EXP3_EXISTING_EVIDENCE_DIAGNOSTIC_PASS', hashes)
    legacy.verify_success(replay, 'EXP3_VALIDATION_EPOCH3_4_REPLAY_PASS', hashes)
    own = repo / 'experiments/exp3b_diagnostics'
    source_hashes = {p.name: sha(p) for p in sorted(own.iterdir()) if p.suffix in ('.py', '.md')}
    require(source_hashes == read(own / 'SOURCE_MANIFEST.json')['source_sha256'] ==
            before['diagnostic_sources_sha256'], 'V1 diagnostic package changed')
    sys.path[:0] = [str(repo / 'experiments/exp3_dpt_peer'), str(lgsa / 'goemotions_adaptation'),
                   str(lgsa / 'working_lgsa')]
    from protocol import package_hashes, source_audit, assets
    from hierarchy_goemotions import load_hierarchy
    from data_goemotions import rows
    from baseline import verify_baseline
    require(package_hashes() == success['identity']['package_sha256'], 'Frozen Exp3 sources changed')
    require(source_audit(lgsa) == success['identity']['baseline_sources'], 'Frozen LGSA sources changed')
    manifest = assets(lgsa)
    require(Path(manifest['model']['path']).resolve() == raw.resolve(), 'Raw model path changed')
    require(sha(lgsa / 'assets/goemotions-v0/manifest.json') == success['dataset_manifest_sha256'], 'Data manifest changed')
    baseline_proof = verify_baseline(baseline, lgsa, manifest)
    require(baseline_proof == before['baseline_proof'], 'Baseline proof changed')
    cfg = read(run / 'config.json')
    require(cfg == success['config'] and cfg['seed'] == 42 and cfg['epochs'] == 4 and
            cfg['peer_loss_enabled'] and cfg['peer_lambda'] == .5 and cfg['peer_top_k'] == 1 and
            cfg['peer_temperature'] == .1 and cfg['fixed_threshold'] == .3 and
            success['best_epoch'] == 3 and success['optimizer_steps'] == 10856, 'Frozen run identity differs')
    hierarchy = load_hierarchy(lgsa / 'goemotions_adaptation/hierarchy.json')
    labels = sorted(hierarchy['source']['labels'], key=lambda r: r['fine_id'])
    train = rows(lgsa / 'assets/goemotions-v0/train.jsonl')
    require(len(train) == 43410 and [r['id'] for r in train] == read(run / 'sample_ids.json')['train'], 'Train IDs differ')
    frequency = np.array([sum(i in row['labels'] for row in train) for i in range(28)])
    # Hashes recorded NOW establish preservation during recovery, not a fictitious
    # pre-failure artifact manifest. Internal numerical reconciliation follows.
    partial_hashes = legacy.snapshot([partial])
    require(all(str(partial / name) in partial_hashes for name in GPU_FILES), 'GPU phase incomplete; STOP')
    hashes.update(partial_hashes)
    context = dict(repo=repo, run=run, baseline=baseline, previous=previous, replay=replay, lgsa=lgsa,
        protected=protected+[partial], hashes=hashes, success=success, cfg=cfg, hierarchy=hierarchy,
        labels=labels, train=train, frequency=frequency, baseline_proof=baseline_proof,
        reused_artifacts=before['reused_artifacts']+[str(partial / name) for name in GPU_FILES],
        source_hashes=source_hashes, git_head=head,
        git_status=subprocess.check_output(['git', '-C', str(repo), 'status', '--short', '--branch'], text=True))
    results = validate_records(partial, context)
    return context, partial, partial_hashes, results


def validate_records(partial, c):
    import numpy as np
    from analysis import gradient_summary
    from probe import aggregate_terms
    integrity = read(partial / 'model_state_integrity.json')
    require(integrity['all_parameters_and_buffers_exactly_unchanged'] is True and
            integrity['frozen_best_and_final_keys_equal_initial'] is True and
            integrity['optimizer_instances_created'] == integrity['optimizer_steps'] == 0 and
            integrity['before_sha256'] == integrity['after_sha256'] and
            len(integrity['before_sha256']) == 64, 'Original GPU model-state audit failed')
    protocol = read(partial / 'fixed_batch_protocol.json')
    order = np.random.RandomState(42).permutation(len(c['train'])).tolist()
    ordered = [c['train'][i] for i in order]
    require(protocol['batch_size'] == 16 and protocol['gradient_batch_indices'] == list(range(10)) and
            protocol['train_ids_in_forward_order'] == [r['id'] for r in ordered] and
            protocol['gradient_sample_ids'] == [[r['id'] for r in ordered[k*16:(k+1)*16]] for k in range(10)],
            'Saved fixed-batch protocol differs')
    terms = csvread(partial / 'peer_loss_terms_snapshot.csv')
    batches = csvread(partial / 'peer_loss_batches_snapshot.csv')
    require(len(batches) == 2714 and [r['batch_index'] for r in batches] == list(range(2714)), 'Incomplete full-train forwards')
    labels = c['labels']
    families = defaultdict(set)
    for i, label in enumerate(labels):
        families[label['coarse_id']].add(i)
    names = {r['fine_label']: i for i, r in enumerate(labels)}
    records = {r['id']: (b//16, r) for b, r in enumerate(ordered)}
    expected = Counter((r['id'], labels[t]['fine_label']) for r in ordered for t in r['labels']
                       if families[labels[t]['coarse_id']]-set(r['labels']))
    require(Counter((r['sample_id'], r['true_label']) for r in terms) == expected and len(terms) == 36884,
            'Missing/duplicated/noneligible peer terms')
    grouped = defaultdict(list)
    for term in terms:
        index, sample = records[term['sample_id']]
        t, n = names[term['true_label']], names[term['negative_label']]
        require(term['batch_index'] == index and n not in sample['labels'] and n != t and
                labels[n]['coarse_id'] == labels[t]['coarse_id'] and
                term['parent'] == labels[t]['coarse_label'], 'Wrong batch/sibling/co-GT in saved term')
        require(term['raw_loss'] >= 0 and term['eligible_terms_in_batch'] > 0 and
                math.isclose(term['weighted_batch_contribution'], .5*term['raw_loss']/term['eligible_terms_in_batch'],
                             rel_tol=1e-10, abs_tol=1e-12), 'Wrong weighted term contribution')
        margin = term['cosine_margin']
        x = -margin/.1
        softplus = max(x, 0)+math.log1p(math.exp(-abs(x)))
        require(math.isclose(softplus, term['raw_loss'], rel_tol=3e-6, abs_tol=2e-6), 'K=1 objective/margin mismatch')
        grouped[index].append(term)
    for batch in batches:
        index = batch['batch_index']
        part = grouped[index]
        n = len(part)
        require(batch['examples'] == len(ordered[index*16:(index+1)*16]) and
                batch['eligible_terms'] == n and all(r['eligible_terms_in_batch'] == n for r in part), 'Batch accounting differs')
        mean = sum(r['raw_loss'] for r in part)/n if n else 0.
        require(math.isclose(mean, batch['actual_peer_loss'], rel_tol=2e-6, abs_tol=1e-6) and
                math.isclose(.5*batch['actual_peer_loss'], batch['weighted_peer_loss'], rel_tol=1e-10, abs_tol=1e-12),
                'Native batch loss no longer reconciles')
    true_rows, parent_rows, negative_rows, pairs, actual = aggregate_terms(terms, labels, c['frequency'], 2714, .5)
    actual['train_examples'] = 43410
    equivalent(actual, read(partial / 'peer_loss_contribution_summary.json'), 'contribution summary')
    for table, filename in [(true_rows, 'peer_loss_by_true_label.csv'), (parent_rows, 'peer_loss_by_parent.csv'),
                            (negative_rows, 'hard_negative_loss_contribution.csv'), (pairs, 'peer_loss_by_pair_snapshot.csv')]:
        equivalent(table, csvread(partial / filename), filename)
    gradients = csvread(partial / 'gradient_conflict_batches.csv')
    layers = csvread(partial / 'gradient_conflict_by_layer.csv')
    wanted = {'full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform'}
    require(Counter((r['batch_index'], r['scope']) for r in gradients) ==
            Counter((b, s) for b in range(10) for s in wanted), 'Missing/duplicate gradient scopes')
    require(Counter((r['batch_index'], r['scope']) for r in layers) ==
            Counter((b, 'BERT_layer_'+str(k)) for b in range(10) for k in range(12)), 'Incomplete BERT-layer gradients')
    for row in gradients+layers:
        require(row['cosine'] is not None and -1 <= row['cosine'] <= 1 and row['hga_alpha'] == 1 and
                row['shared_parameter_tensors'] > 0 and row['original_gradient_norm'] > 0 and
                row['peer_gradient_norm_unweighted'] > 0 and
                math.isclose(row['peer_gradient_norm_weighted'], .5*row['peer_gradient_norm_unweighted'], rel_tol=1e-10) and
                math.isclose(row['weighted_peer_to_original_norm_ratio'],
                             row['peer_gradient_norm_weighted']/row['original_gradient_norm'], rel_tol=1e-10) and
                math.isclose(row['peer_loss'], batches[row['batch_index']]['actual_peer_loss'], rel_tol=2e-5, abs_tol=2e-6),
                'Gradient norms/objectives differ from completed probe')
    gradient = gradient_summary(gradients)
    saved_gradient = read(partial / 'gradient_conflict_summary.json')
    equivalent(gradient, saved_gradient['scopes'], 'gradient summary')
    require(saved_gradient['checkpoint_epoch'] == 3 and saved_gradient['frozen_raw_prompt_and_keys'] is True and
            len(saved_gradient['shared_parameter_names']) == len(set(saved_gradient['shared_parameter_names'])) > 0 and
            [r['batch_index'] for r in saved_gradient['excluded_by_batch']] == list(range(10)), 'Gradient identity differs')
    for filename in ('test_fixed_0.30_metrics.json', 'validation_epoch_3.json', 'validation_epoch_4.json'):
        rare_metric(read(c['run'] / filename))
    rare_metric(read(c['baseline'] / 'test_fixed_0.30_metrics.json'))
    print('COMPLETED_GPU_RECORDS_RECONCILED: 43410 rows, 36884 terms, 2714 forwards, 10 gradient batches', flush=True)
    return actual, true_rows, parent_rows, negative_rows, pairs, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--no-notify', action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    # Unconditional: GPU is hidden even if the workstation has a free GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    repo = args.repo.resolve()
    out = repo / 'outputs/exp3b_diagnostics_seed42_recovered_v1'
    url = '' if args.no_notify else os.environ.get('FYP_NTFY_URL', NTFY)
    created = False
    notify(url, 'FYP Exp3B CPU recovery started', 'Reuse completed GPU evidence; zero new forwards/gradients/optimizer steps.')
    try:
        require(not out.exists(), 'Recovery output exists; STOP, never overwrite')
        source = repo / 'experiments/exp3b_diagnostics'
        require(sha(source / 'SOURCE_MANIFEST.json') == MANIFEST_SHA, 'Delivered V1 manifest changed')
        expected = read(source / 'SOURCE_MANIFEST.json')['source_sha256']
        require({p.name: sha(p) for p in source.iterdir() if p.suffix in ('.py', '.md')} == expected, 'V1 sources changed')
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('exp3b_original_runner', source / 'run.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        context, partial, partial_hashes, results = context_and_records(repo, legacy)
        import torch
        require(torch.__version__ == '2.5.1+cu118' and not torch.cuda.is_initialized(), 'CPU-only environment differs')
        own = Path(__file__).resolve()
        own_hash = sha(own)
        context['protected'].append(own.parent)
        context['hashes'].update(legacy.snapshot([own.parent]))
        if args.audit_only:
            require(legacy.snapshot(context['protected']) == context['hashes'], 'Inputs changed during CPU audit')
            print('EXP3B_CPU_RECOVERY_AUDIT_PASS output_created=false new_gpu_forwards=0 new_autograd_batches=0 optimizer_steps=0', flush=True)
        else:
            out.mkdir(parents=True, exist_ok=False)
            created = True
            from analysis import jsonwrite
            jsonwrite(out / 'RECOVERY_PROVENANCE.json', {
                'mode': 'CPU_ONLY_COMPLETION_OF_SAVED_GPU_PROBES', 'new_model_forwards': 0,
                'new_autograd_batches': 0, 'optimizer_steps': 0, 'original_failure_directory': str(partial),
                'preserved_original_failure': read(partial / 'FAILURE.json'),
                'partial_file_hashes_recorded_at_recovery_start': partial_hashes,
                'partial_hash_limit': 'No pre-failure SUCCESS manifest existed; reuse additionally reconciles counts, losses, gradients and model-state audit.',
                'completed_gpu_probe_sources': list(GPU_FILES), 'recovery_script': str(own), 'recovery_script_sha256': own_hash,
                'metric_compatibility': 'Explicit rare_f1 / rare_mean_f1 aliases; both present must agree at 1e-12; missing/conflicting fields fail.',
                'v1_sources_unchanged': True, 'new_output_directory': str(out)})
            import probe
            calls = []
            def reuse_probes(*arguments):
                require(not calls and arguments[0] == repo and arguments[7] == out and
                        not torch.cuda.is_initialized(), 'Unexpected probe call; no GPU allowed')
                require(legacy.snapshot([partial]) == partial_hashes and sha(own) == own_hash,
                        'Saved GPU evidence or recovery source changed')
                calls.append(True)
                for filename in GPU_FILES:
                    with (partial / filename).open('rb') as incoming, (out / filename).open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    require(sha(out / filename) == partial_hashes[str(partial / filename)], 'Copied artifact differs')
                print('REUSING_SAVED_GPU_PROBES new_forwards=0 new_autograd_batches=0 optimizer_steps=0', flush=True)
                return results
            # Only the independent diagnostic orchestration is adapted in memory.
            # Frozen experiment/model sources and V1 files are not edited.
            probe.run_probes = reuse_probes
            legacy.reconcile = reconcile
            legacy.diagnose(context, out)
    except BaseException as error:
        if created and not (out / 'FAILURE.json').exists():
            from analysis import jsonwrite
            jsonwrite(out / 'FAILURE.json', {'status': 'CPU_RECOVERY_FAILED', 'error': str(error),
                'new_model_forwards': 0, 'new_autograd_batches': 0, 'optimizer_steps': 0,
                'note': 'Original failed directory and all original experiments preserved; recovery output retained.'})
        notify(url, 'FYP Exp3B CPU recovery failed', str(error)[:180])
        raise
    notify(url, 'FYP Exp3B CPU recovery finished',
           'CPU-only audit passed; no output created.' if args.audit_only else 'Hash-verified report completed; no GPU rerun. '+str(out))


if __name__ == '__main__':
    main()
