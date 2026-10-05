"""Refuse incomplete/mismatched comparisons. This verifies all six scientific runs."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mslib import *
from metrics_ms import load_predictions, probabilities, threshold_search, score, close_metrics
from evidence_ms import check_geometry


def verify(output, active_job=None, check_checkpoint_hashes=True):
    output = Path(output)
    records, reference = [], None
    jobs = {}
    required = {'config.json', 'experiment_metadata.json', 'environment.json', 'training_log.csv',
        'validation_metrics_by_epoch.csv', 'threshold_search.csv', 'test_fixed_0.30_metrics.json',
        'test_validation_selected_metrics.json', 'per_label_fixed_0.30_metrics.csv',
        'per_label_validation_selected_metrics.csv', 'rare_label_metrics.json',
        'validation_predictions.npz', 'test_predictions.npz', 'checkpoint_info.json'}
    for model in ['bert', 'lgsa']:
        base = output / MODEL_DIR[model]
        require({p.name for p in base.iterdir() if p.is_dir()} == {f'seed{s}' for s in SEEDS}, 'Missing/unexpected scientific seed directories')
        for seed in SEEDS:
            path = base / ('seed' + str(seed))
            s = read(path / 'SUCCESS.json')
            require(s['model'] == model and s['seed'] == seed, 'Success identity mismatch')
            hashes = artifacts(path, s)
            require(required <= hashes.keys(), 'Required artifacts missing')
            m = read(path / 'experiment_metadata.json')
            require(m['model'] == model and m['seed'] == seed and m['phase'] == 'full', 'Model/seed/phase mismatch')
            job = m['slurm_job_id']
            pending = active_job is not None and job == active_job
            if pending:
                require(m['status'] == 'AWAITING_SLURM_COMPLETION' and m['slurm_state'] == 'RUNNING'
                        and m['slurm_exit_code'] is None and s['status'] == 'NORMALIZED_AWAITING_SLURM',
                        'Active-job record falsely claims completion')
                jobs[job] = {'job_id': job, 'state': 'AWAITING_SLURM_COMPLETION'}
            else:
                require(m['status'] == 'COMPLETED' and m['slurm_state'] == 'COMPLETED'
                        and m['slurm_exit_code'] == '0:0' and s['status'] == 'NORMALIZED_RUN_PASS',
                        'Invalid recorded accounting')
            if not pending and job not in jobs:
                jobs[job] = sacct(job)
            require(m['dataset_split_sizes'] == SIZES and m['label_names'] == NAMES and m['num_labels'] == 28, 'Dataset/labels mismatch')
            require(m['raw_model_revision'] == REVISION and m['rare_ids'] == RARE and m['fixed_threshold'] == .3, 'Model/evaluation mismatch')
            require(m['threshold_candidates'] == CANDIDATES and m['best_epoch'] in [1, 2, 3, 4], 'Selection protocol mismatch')
            require(m['checkpoint_reload_verified'] is True and m['checkpoint_reload_evidence'], 'Checkpoint reload evidence missing')
            require(m['config'] == config(model, seed) and read(path / 'config.json') == m['config'], 'Frozen protocol mismatch')
            require(m['orchestration_release_sha256'] == release(), 'Code release mismatch')
            require(sha(Path(m['native_run_directory']) / 'SUCCESS.json') == m['native_success_sha256'], 'Native lineage changed')
            checkpoint = read(path / 'checkpoint_info.json')
            if check_checkpoint_hashes:
                require(sha(checkpoint['path']) == checkpoint['sha256'], 'Best checkpoint changed')
            if model == 'lgsa':
                require(m['lgsa_patch_commit'] == PATCH and m['hierarchy_sha256'] == sha(ROOT / 'frozen/hierarchy.json'), 'LGSA source/hierarchy mismatch')
                require('geometry_acceptance.json' in hashes, 'Geometry artifact missing')
                check_geometry(path, m['config'])
            curve = csvread(path / 'validation_metrics_by_epoch.csv')
            require([int(r['epoch']) for r in curve] == [1, 2, 3, 4], 'Incomplete validation curve')
            require(int(max(curve, key=lambda r: float(r['validation_macro_f1']))['epoch']) == m['best_epoch'], 'Wrong checkpoint selection')
            yv, zv, idv = load_predictions(path / 'validation_predictions.npz')
            yt, zt, idt = load_predictions(path / 'test_predictions.npz')
            require(len(idv) == 5426 and len(idt) == 5427, 'Prediction sizes mismatch')
            pair = (idv, yv, idt, yt)
            if reference is None:
                reference = pair
            require(all(np.array_equal(a, b) for a, b in zip(reference, pair)), 'Unpaired examples/gold labels')
            selected, search = threshold_search(yv, zv, model)
            require(selected == m['selected_threshold'], 'Threshold used test or wrong validation objective')
            saved = csvread(path / 'threshold_search.csv')
            require([float(r['threshold']) for r in saved] == CANDIDATES, 'Wrong threshold candidates')
            require(all(abs(a['macro_f1'] - float(b['macro_f1'])) < 1e-10 for a, b in zip(search, saved)), 'Threshold search differs')
            for setting, t in [('fixed_0.30', .3), ('validation_selected', selected)]:
                measured, rows = score(yt, probabilities(zt, model), t)
                stored = read(path / ('test_' + setting + '_metrics.json'))
                require(all(np.isfinite(stored[k]) and 0 <= stored[k] <= 1 for k in METRICS), 'Invalid metrics')
                close_metrics(measured, stored)
                labels = csvread(path / ('per_label_' + setting + '_metrics.csv'))
                require(len(labels) == 28, 'Wrong per-label count')
                for expected, actual in zip(rows, labels):
                    for k in expected:
                        if k == 'label':
                            require(actual[k] == expected[k], 'Label ordering differs')
                        else:
                            require(abs(float(actual[k]) - expected[k]) < 1e-10, 'Per-label metrics/counts differ')
            records.append({'model': model, 'seed': seed, 'path': str(path), 'metadata': m})
    require(len(records) == 6, 'All six scientific runs are required')
    return records, jobs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=Path, required=True)
    args = p.parse_args()
    records, jobs = verify(args.outputs)
    print('SIX_RUNS_INTEGRITY_PASS', len(records), 'runs;', len(jobs), 'completed training jobs')
