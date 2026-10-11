"""Validation-only selection, frozen matched sibling metrics and pilot figures."""
import csv
import json
from pathlib import Path
import numpy as np
from protocol import RARE, read, require, sha, write

GROUPS = [('admiration', 'approval'), ('anger', 'annoyance'), ('curiosity', 'confusion'),
          ('gratitude', 'admiration'), ('love', 'admiration'), ('amusement', 'joy')]
DIRECTIONS = [(t, n) for a, b in GROUPS for t, n in [(a, b), (b, a)]]


def csvwrite(path, rows):
    require(bool(rows), 'Empty table: ' + str(path))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))


def training_tables(out):
    steps = [json.loads(row) for row in (out / 'steps.jsonl').read_text().splitlines()]
    require(steps and all(np.isfinite(r[k]) for r in steps for k in ('loss', 'peer_loss', 'original_lgsa_loss')),
            'Missing or nonfinite training diagnostics')
    trials = read(out / 'threshold_selection.json')
    require(trials['selection_split'] == 'validation', 'Threshold selection leaked test')
    csvwrite(out / 'threshold_search.csv', trials['trials'])
    training, validation = [], []
    for epoch in sorted({r['epoch'] for r in steps}):
        selected = [r for r in steps if r['epoch'] == epoch]
        row = {'epoch': epoch, 'train_total_loss': float(np.mean([r['loss'] for r in selected])),
               'train_original_lgsa_loss': float(np.mean([r['original_lgsa_loss'] for r in selected])),
               'train_peer_loss': float(np.mean([r['peer_loss'] for r in selected])),
               'number_of_peer_terms': sum(r['number_of_peer_terms'] for r in selected),
               'number_of_samples_with_peer_terms': sum(r['number_of_samples_with_peer_terms'] for r in selected)}
        denominator = row['number_of_peer_terms']
        for k in ('mean_positive_similarity', 'mean_hard_negative_similarity', 'mean_peer_margin'):
            row[k] = float(sum(r[k]*r['number_of_peer_terms'] for r in selected) / denominator) if denominator else 0.
        m = read(out / f'validation_epoch_{epoch}.json')
        row.update({'validation_' + k: v for k, v in m.items() if k not in ('threshold', 'examples', 'num_labels')})
        training.append(row)
        validation.append({'epoch': epoch, **m})
    csvwrite(out / 'training_log.csv', training)
    csvwrite(out / 'peer_training_diagnostics.csv', training)
    csvwrite(out / 'validation_metrics_by_epoch.csv', validation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for field, label in [('train_original_lgsa_loss', 'Original LGSA'), ('train_peer_loss', 'Peer'),
                          ('train_total_loss', 'Total')]:
        ax.plot([r['epoch'] for r in training], [r[field] for r in training], marker='o', label=label)
    ax.set(xlabel='Epoch', ylabel='Mean loss', title='Training losses (peer is unweighted)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'training_losses.png', dpi=180)
    plt.close(fig)


def validate_predictions(gold, logits, ids):
    require(gold.shape == logits.shape == (5427, 28), 'Expected official 5427 x 28 predictions')
    require(ids.shape == (5427,) and len(set(ids.tolist())) == 5427, 'Duplicate/missing IDs')
    require(np.isin(gold, [0, 1]).all() and np.isfinite(logits).all(), 'Invalid gold/logits')


def load_matched(baseline, candidate):
    with np.load(baseline / 'test_predictions.npz', allow_pickle=False) as raw:
        y0, z0, ids0 = [raw[k].copy() for k in ('gold', 'logits', 'ids')]
    with np.load(candidate / 'test_predictions.npz', allow_pickle=False) as raw:
        y1, z1, ids1 = [raw[k].copy() for k in ('gold', 'logits', 'ids')]
    validate_predictions(y0, z0, ids0)
    validate_predictions(y1, z1, ids1)
    require(set(ids0.tolist()) == set(ids1.tolist()), 'Test samples differ')
    index = {sid: i for i, sid in enumerate(ids1.tolist())}
    order = [index[sid] for sid in ids0.tolist()]
    require(np.array_equal(y0, y1[order]), 'Matched test GT differs')
    return y0.astype(bool), z0, z1[order]


def aggregate_siblings(y, z, parents):
    p = sigmoid(z)
    margins, inversions, false_positive = [], [], []
    for i, active in enumerate(y):
        gt = set(np.flatnonzero(active).tolist())
        for true in sorted(gt):
            candidates = [j for j in range(28) if parents[j] == parents[true] and j not in gt]
            if not candidates:
                continue
            hard = max(candidates, key=lambda j: z[i, j])
            margins.append(p[i, true] - p[i, hard])
            # Raw logits avoid spurious ties caused by sigmoid saturation.
            inversions.append(z[i, hard] > z[i, true])
            false_positive.append(p[i, hard] >= .30)
    require(bool(margins), 'No eligible sibling GT instances')
    return {'eligible_gt_instances': len(margins), 'mean_margin': float(np.mean(margins)),
            'median_margin': float(np.median(margins)), 'inversion_rate': float(np.mean(inversions)),
            'conditional_fp_rate': float(np.mean(false_positive))}


def frozen_pairs(y, z0, z1, names, parents):
    p0, p1 = sigmoid(z0), sigmoid(z1)
    rows = []
    for true, negative in DIRECTIONS:
        t, n = names.index(true), names.index(negative)
        require(parents[t] == parents[n], 'Frozen pair is not sibling')
        eligible = y[:, t] & ~y[:, n]
        count = int(eligible.sum())
        require(count > 0, 'Frozen pair has no eligible samples')
        row = {'true_label': true, 'sibling_negative': negative, 'N': count,
               'co_gt_excluded': int((y[:, t] & y[:, n]).sum())}
        for tag, z, p in [('exp2', z0, p0), ('exp3', z1, p1)]:
            row[tag + '_margin'] = float(np.mean(p[eligible, t] - p[eligible, n]))
            row[tag + '_inversion_rate'] = float(np.mean(z[eligible, n] > z[eligible, t]))
            row[tag + '_conditional_fp_rate'] = float(np.mean(p[eligible, n] >= .30))
        for key in ('margin', 'inversion_rate', 'conditional_fp_rate'):
            row['delta_' + key] = row['exp3_' + key] - row['exp2_' + key]
        rows.append(row)
    return rows


def compare_runs(baseline, candidate, hierarchy_path):
    from evaluation import metrics
    baseline, out = Path(baseline), Path(candidate)
    h = read(hierarchy_path)
    labels = sorted(h['labels'], key=lambda r: r['fine_id'])
    names, parents = [r['fine_label'] for r in labels], [r['coarse_id'] for r in labels]
    y, z0, z1 = load_matched(baseline, out)
    m0, labels0 = metrics(y, sigmoid(z0), .30, RARE)
    m1, labels1 = metrics(y, sigmoid(z1), .30, RARE)
    metrics_rows = [{'metric': key, 'exp2': m0[key], 'exp3': m1[key], 'delta': m1[key]-m0[key]}
                    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'macro_precision',
                                'macro_recall', 'macro_f1', 'rare_mean_f1', 'nonrare_mean_f1')]
    csvwrite(out / 'exp2_vs_exp3_seed42.csv', metrics_rows)
    csvwrite(out / 'per_label_fixed_0.30.csv', labels1)
    a0, a1 = aggregate_siblings(y, z0, parents), aggregate_siblings(y, z1, parents)
    require(a0['eligible_gt_instances'] == a1['eligible_gt_instances'], 'Aggregate eligibility differs')
    delta = {k: a1[k]-a0[k] for k in a0 if k != 'eligible_gt_instances'}
    write(out / 'sibling_aggregate_comparison.json', {'threshold': .30, 'exp2': a0, 'exp3': a1, 'delta': delta,
        'sample_selection': 'same GT instance; all co-GT excluded; neutral singleton skips',
        'provenance': {'exp2_predictions_sha256': sha(baseline / 'test_predictions.npz'),
                       'exp3_predictions_sha256': sha(out / 'test_predictions.npz'), 'hierarchy_sha256': sha(hierarchy_path)}})
    pairs = frozen_pairs(y, z0, z1, names, parents)
    csvwrite(out / 'sibling_pair_comparison.csv', pairs)
    plot_comparisons(baseline, out, m0, m1, a0, a1, pairs)
    improved = [r['true_label']+' → '+r['sibling_negative'] for r in pairs
                if r['delta_margin'] > 0 and r['delta_inversion_rate'] <= 0 and r['delta_conditional_fp_rate'] <= 0]
    worsened = [r['true_label']+' → '+r['sibling_negative'] for r in pairs
                if r['delta_margin'] < 0 and r['delta_inversion_rate'] >= 0 and r['delta_conditional_fp_rate'] >= 0]
    mixed = [r['true_label']+' → '+r['sibling_negative'] for r in pairs
             if r['true_label']+' → '+r['sibling_negative'] not in improved+worsened]
    promising = (m1['macro_f1'] >= m0['macro_f1'] and m1['rare_mean_f1'] >= m0['rare_mean_f1']
                 and delta['mean_margin'] > 0 and delta['inversion_rate'] <= 0 and delta['conditional_fp_rate'] <= 0)
    header = '| Metric | Exp2 | Exp3 | Delta (pp) |\n|---|---:|---:|---:|'
    lines = [header] + [f"| {r['metric']} | {r['exp2']*100:.3f} | {r['exp3']*100:.3f} | {r['delta']*100:+.3f} |"
                        for r in metrics_rows]
    rare_note = '\n'.join(f"- {names[i]}: F1 {labels0[i]['f1']*100:.2f}% → {labels1[i]['f1']*100:.2f}%, "
                          f"support={labels1[i]['support']}, FP {labels0[i]['fp']} → {labels1[i]['fp']}" for i in (16, 23))
    report = f'''# Experiment 3 seed42 pilot

Primary evaluation: fixed threshold0.30, validation Macro-F1 checkpoint selection.
Secondary evaluation uses the saved validation-selected threshold. No test tuning.

{chr(10).join(lines)}

Macro-F1 delta: {(m1['macro_f1']-m0['macro_f1'])*100:+.3f}pp.
Rare mean F1 delta: {(m1['rare_mean_f1']-m0['rare_mean_f1'])*100:+.3f}pp.
Micro precision/recall deltas: {(m1['micro_precision']-m0['micro_precision'])*100:+.3f}/
{(m1['micro_recall']-m0['micro_recall'])*100:+.3f}pp. Interpret their trade-off jointly.

Aggregate matched sibling GT instances: {a0['eligible_gt_instances']}.
Mean margin: {a0['mean_margin']:.6f} → {a1['mean_margin']:.6f} (delta {delta['mean_margin']:+.6f}).
Median margin delta: {delta['median_margin']:+.6f}.
Inversion rate: {a0['inversion_rate']*100:.3f}% → {a1['inversion_rate']*100:.3f}%
(delta {delta['inversion_rate']*100:+.3f}pp).
Conditional sibling FP: {a0['conditional_fp_rate']*100:.3f}% → {a1['conditional_fp_rate']*100:.3f}%
(delta {delta['conditional_fp_rate']*100:+.3f}pp).

Frozen directed pairs (joint descriptive rule: margin improves, inversion and FP do not rise):
Improved: {', '.join(improved) or 'none'}.
Worsened (reverse rule): {', '.join(worsened) or 'none'}.
Mixed/unchanged: {', '.join(mixed) or 'none'}. See all12 rows in sibling_pair_comparison.csv.
These categories are descriptive, not significance tests. No pairs were selected after training.

Grief and relief:

{rare_note}

Any negative metric delta in the overall table, rare-label table or targeted comparison
is retained as a possible regression. Do not classify the entire pilot by one pair.
Strict predeclared joint promising signal: {promising}. If true, recommend (do not launch)
Experiment3.1 with paired seeds42/123/2026. If false, discuss the observed trade-offs
before another run; do not tune this pilot until it wins.

This single-seed result shows/does not show a promising signal under the stated rule.
It does not establish statistical significance, general effectiveness or full DPT reproduction.
Keys are the frozen HGS/pre-GAT classifier rows; peer loss trains the existing anchor path.
'''
    with (out / 'EXP3_SUMMARY.md').open('x') as stream:
        stream.write(report)


def validation_curve(directory):
    directory = Path(directory)
    csv_path = directory / 'validation_metrics_by_epoch.csv'
    if csv_path.exists():
        with csv_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        # Normalized Exp2 uses validation_macro_f1; Exp3 retains macro_f1.
        values = [(int(r['epoch']), float(r.get('validation_macro_f1', r.get('macro_f1')))) for r in rows]
    else:
        values = [(i, read(directory / f'validation_epoch_{i}.json')['macro_f1']) for i in range(1, 5)]
    require([i for i, _ in values] == [1, 2, 3, 4]
            and all(np.isfinite(v) and 0 <= v <= 1 for _, v in values), 'Invalid four-epoch validation curve')
    return values


def plot_comparisons(baseline, out, m0, m1, a0, a1, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for directory, label in ((baseline, 'Exp2'), (out, 'Exp3')):
        values = validation_curve(directory)
        ax.plot([i for i, _ in values], [100*s for _, s in values], marker='o', label=label)
    ax.set(xlabel='Epoch', ylabel='Validation Macro-F1 (%) @0.30')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'validation_macro_by_epoch.png', dpi=180)
    plt.close(fig)
    keys = ['micro_f1', 'macro_f1', 'rare_mean_f1', 'nonrare_mean_f1']
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(keys))
    ax.bar(x-.18, [m0[k]*100 for k in keys], .36, label='Exp2')
    ax.bar(x+.18, [m1[k]*100 for k in keys], .36, label='Exp3')
    ax.set_xticks(x, ['Micro F1', 'Macro F1', 'Rare F1', 'Nonrare F1'])
    ax.set_ylabel('F1 (%) @0.30')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'overall_metrics.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, key in zip(axes, ('mean_margin', 'inversion_rate', 'conditional_fp_rate')):
        scale = 1 if key == 'mean_margin' else 100
        ax.bar(['Exp2', 'Exp3'], [a0[key]*scale, a1[key]*scale])
        ax.set_title(key)
        ax.set_ylabel('Probability margin' if scale == 1 else 'Rate (%)')
    fig.tight_layout()
    fig.savefig(figures / 'aggregate_sibling_metrics.png', dpi=180)
    plt.close(fig)
    for field, unit in [('margin', 'Probability margin'), ('conditional_fp_rate', 'Rate (pp)')]:
        scale = 1 if field == 'margin' else 100
        fig, ax = plt.subplots(figsize=(10, 6))
        values = [r['delta_' + field]*scale for r in pairs]
        ax.barh(np.arange(12), values)
        ax.set_yticks(np.arange(12), [r['true_label']+' -> '+r['sibling_negative'] for r in pairs])
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_xlabel('Exp3 - Exp2: ' + unit)
        ax.set_title('Frozen12 pairs: ' + ('higher is favorable' if field == 'margin' else 'lower is favorable'))
        fig.tight_layout()
        fig.savefig(figures / ('frozen_pair_delta_' + field + '.png'), dpi=180)
        plt.close(fig)
