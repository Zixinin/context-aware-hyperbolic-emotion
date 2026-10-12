"""Descriptive fixed0.30 paired comparison; never changes the approved hierarchy."""
import argparse
from pathlib import Path
import numpy as np
from b4lib import NAMES, SIZES, read, require, sha, write
from hierarchy_adapter import PAIRS
from evaluate import metrics, probabilities, csvwrite

FIELDS = ['micro_precision', 'micro_recall', 'micro_f1', 'macro_precision',
    'macro_recall', 'macro_f1', 'weighted_f1', 'rare_mean_f1', 'nonrare_mean_f1']

def compare(baseline, candidate, metadata):
    baseline, out = Path(baseline), Path(candidate)
    arrays = []
    for path in (baseline / 'test_predictions.npz', out / 'test_predictions.npz'):
        with np.load(path, allow_pickle=False) as saved:
            y, z, ids = [saved[k].copy() for k in ('gold', 'logits', 'ids')]
        require(y.shape == z.shape == (SIZES['test'], 28) and ids.shape == (SIZES['test'],), 'Expected official N x 28 predictions')
        require(len(set(ids.tolist())) == len(ids) and np.isfinite(z).all(), 'Invalid prediction arrays')
        arrays.append((y, z, ids))
    y0, z0, ids0 = arrays[0]
    y1, z1, ids1 = arrays[1]
    require(np.array_equal(ids0, ids1) and np.array_equal(y0, y1), 'Canonical IDs/GT/order mismatch')
    m0, labels0 = metrics(y0, probabilities(z0), .30)
    m1, labels1 = metrics(y1, probabilities(z1), .30)
    table = [dict(metric=k, exp2=m0[k], exp4b=m1[k], delta=m1[k] - m0[k],
        delta_percentage_points=100 * (m1[k] - m0[k])) for k in FIELDS]
    csvwrite(out / 'exp2_vs_exp4b_seed42.csv', table)
    rows = [dict(label_id=i, label=NAMES[i], support=labels1[i]['support'],
        exp2_precision=labels0[i]['precision'], exp4b_precision=labels1[i]['precision'],
        exp2_recall=labels0[i]['recall'], exp4b_recall=labels1[i]['recall'],
        exp2_f1=labels0[i]['f1'], exp4b_f1=labels1[i]['f1'],
        delta_f1=labels1[i]['f1'] - labels0[i]['f1'],
        delta_f1_percentage_points=100 * (labels1[i]['f1'] - labels0[i]['f1'])) for i in range(28)]
    csvwrite(out / 'labelwise_metrics.csv', labels1)
    csvwrite(out / 'exp2_vs_exp4b_labelwise.csv', rows)
    pairs = []
    for a, b in PAIRS:
        i, j = NAMES.index(a), NAMES.index(b)
        pairs.append(dict(label_a=a, label_b=b, support_a=rows[i]['support'], support_b=rows[j]['support'],
            exp2_f1_a=rows[i]['exp2_f1'], exp4b_f1_a=rows[i]['exp4b_f1'], delta_f1_a=rows[i]['delta_f1'],
            exp2_f1_b=rows[j]['exp2_f1'], exp4b_f1_b=rows[j]['exp4b_f1'], delta_f1_b=rows[j]['delta_f1']))
    csvwrite(out / 'stable_pair_labelwise_comparison.csv', pairs)
    baseline_meta = read(baseline / 'experiment_metadata.json')
    record = {'status': 'EXP4B_SEED42_COMPARISON_VERIFIED', 'threshold': .30,
        'exp2': m0, 'exp4b': m1, 'deltas': {k: m1[k] - m0[k] for k in FIELDS},
        'baseline_prediction_sha256': sha(baseline / 'test_predictions.npz'),
        'candidate_prediction_sha256': sha(out / 'test_predictions.npz'),
        'old_best_epoch': baseline_meta['best_epoch'], 'new_best_epoch': metadata['best_epoch'],
        'new_train_and_evaluation_seconds': metadata['training_elapsed_seconds'],
        'improved_labels': [r['label'] for r in rows if r['delta_f1'] > 0],
        'worsened_labels': [r['label'] for r in rows if r['delta_f1'] < 0],
        'unchanged_labels': [r['label'] for r in rows if r['delta_f1'] == 0],
        'multi_seed_status': 'NOT_RUN: current authorized scope is seed42 pilot',
        'hierarchy_modified_after_metrics': False}
    write(out / 'comparison.json', record)
    line = ['# Exp4B seed42 pilot', '', 'Primary: threshold0.30; best validation Macro-F1 checkpoint; only28 emotion labels.', '',
        '| Metric | Exp2 (%) | Exp4B (%) | Delta (pp) |', '|---|---:|---:|---:|']
    line += [f"| {r['metric']} | {100*r['exp2']:.3f} | {100*r['exp4b']:.3f} | {r['delta_percentage_points']:+.3f} |" for r in table]
    line += ['', f"Best epoch: old {baseline_meta['best_epoch']}; new {metadata['best_epoch']}.",
        f"New training + evaluation time: {metadata['training_elapsed_seconds']:.2f}s.", '',
        '## Stable pair label-wise changes', '', '| Pair | First label Delta F1 (pp) | Second label Delta F1 (pp) |', '|---|---:|---:|']
    line += [f"| {r['label_a']} / {r['label_b']} | {100*r['delta_f1_a']:+.3f} | {100*r['delta_f1_b']:+.3f} |" for r in pairs]
    winners = sorted(rows, key=lambda r: r['delta_f1'], reverse=True)
    line += ['', '## Interpretation and limits', '',
        'This is one seed, not evidence of consistent multi-seed improvement or statistical significance.',
        'The frozen full hierarchy induces32→56 auxiliary supervision nodes,2→9 depths and54→68 total tokens; text window stays50.',
        'Internal nodes are auxiliary hierarchy targets, never exported emotion predictions. No hierarchy tuning follows test metrics.',
        'Old results were produced on HPC; new results on Ubuntu4090D. Same software/source/protocol pins are checked, but hardware/runtime are not identical.',
        'Largest descriptive gains: ' + ', '.join(f"{r['label']} {r['delta_f1_percentage_points']:+.2f}pp" for r in winners[:5]),
        'Largest descriptive losses: ' + ', '.join(f"{r['label']} {r['delta_f1_percentage_points']:+.2f}pp" for r in winners[-5:]), '',
        'Multi-seed and all pruning/flat-tree/peer-loss/hyperbolic ablations remain NOT RUN.', '']
    with (out / 'EXP4B_SUMMARY.md').open('x') as stream:
        stream.write('\n'.join(line))
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    from train import verify_run
    success = verify_run(args.candidate, 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED')
    require(success['baseline_success_sha256'] == sha(args.baseline / 'SUCCESS.json'), 'Wrong baseline')
    require((args.candidate / 'comparison.json').is_file(), 'No saved comparison')
    print(read(args.candidate / 'comparison.json'))

if __name__ == '__main__':
    main()
