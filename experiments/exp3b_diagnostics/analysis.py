"""Pure-array analysis of already completed runs. No model selection or fitting."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np

THRESHOLD = .30
RARE = [16, 21, 23, 19, 12, 24, 14]
BRANCHES = ['Positive', 'Negative', 'Ambiguous', 'Neutral']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def jsonwrite(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def csvwrite(path, rows):
    require(bool(rows), 'Empty diagnostic table: ' + str(path))
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probability(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -80, 80)))


def divide(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def counts(y, p):
    y, pred = np.asarray(y, dtype=bool), np.asarray(p) >= THRESHOLD
    require(y.shape == pred.shape and y.ndim == 2, 'Bad metric shapes')
    tp, fp, fn = [(mask).sum(0) for mask in (y & pred, ~y & pred, y & ~pred)]
    return dict(tp=tp, fp=fp, fn=fn, support=tp+fn, prediction_count=tp+fp,
                precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
                f1=divide(2*tp, 2*tp+fp+fn))


def overall(y, p, indices=None):
    if indices is not None:
        y, p = y[:, indices], p[:, indices]
    c = counts(y, p)
    tp, fp, fn = [int(c[k].sum()) for k in ('tp', 'fp', 'fn')]
    supported = c['support'] > 0
    return {'examples': len(y), 'num_labels': y.shape[1], 'tp': tp, 'fp': fp, 'fn': fn,
            'micro_precision': float(divide(tp, tp+fp)), 'micro_recall': float(divide(tp, tp+fn)),
            'micro_f1': float(divide(2*tp, 2*tp+fp+fn)), 'macro_f1': float(c['f1'].mean()),
            'macro_precision': float(c['precision'].mean()), 'macro_recall': float(c['recall'].mean()),
            'macro_f1_supported_labels': float(c['f1'][supported].mean()) if supported.any() else None,
            'supported_labels': int(supported.sum()),
            'average_predicted_label_count': (tp+fp)/len(y) if len(y) else None,
            'fp_per_example': fp/len(y) if len(y) else None}


def ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1)/2
        start = stop
    return result


def spearman(x, y):
    valid = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and
             np.isfinite(a) and np.isfinite(b)]
    if len(valid) < 3:
        return {'n': len(valid), 'rho': None, 'reason': 'fewer than 3 available observations'}
    a, b = [ranks(v) for v in zip(*valid)]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'n': len(valid), 'rho': None, 'reason': 'constant ranked variable'}
    return {'n': len(valid), 'rho': float(np.corrcoef(a, b)[0, 1]),
            'interpretation': 'descriptive; no significance or causal claim'}


def load_predictions(path, expected_rows, cosine=False):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in ('gold', 'logits', 'ids')}
        if cosine:
            arrays['cosine_similarities'] = saved['cosine_similarities'].copy()
    require(arrays['gold'].shape == arrays['logits'].shape == (expected_rows, 28), 'Prediction shape mismatch')
    require(arrays['ids'].shape == (expected_rows,) and len(set(arrays['ids'].tolist())) == expected_rows,
            'Prediction IDs duplicate or missing')
    require(np.isin(arrays['gold'], [0, 1]).all() and np.isfinite(arrays['logits']).all(), 'Invalid predictions')
    if cosine:
        require(arrays['cosine_similarities'].shape == (expected_rows, 28) and
                np.isfinite(arrays['cosine_similarities']).all(), 'Invalid cosine matrix')
    return arrays


def matched(a, b):
    # Require identical recorded order; do not silently drop or duplicate samples.
    require(np.array_equal(a['ids'], b['ids']) and np.array_equal(a['gold'], b['gold']),
            'Prediction sample IDs/order or ground truth differ')


def eligible_mask(y, parents):
    masks = np.zeros_like(y, dtype=bool)
    parents = np.asarray(parents)
    for label, parent in enumerate(parents):
        family = np.flatnonzero(parents == parent)
        masks[:, label] = y[:, label].astype(bool) & (~y[:, family].astype(bool)).any(1)
    return masks


def sibling_per_label(y, z, parents, cosines=None):
    p, y, parents = probability(z), y.astype(bool), np.asarray(parents)
    result = []
    for label in range(y.shape[1]):
        values = []
        for i in np.flatnonzero(y[:, label]):
            candidates = np.flatnonzero((parents == parents[label]) & ~y[i])
            if not len(candidates):
                continue
            neg = int(candidates[np.argmax(z[i, candidates])])
            values.append((p[i, label]-p[i, neg], float(z[i, label])-float(z[i, neg]),
                           z[i, neg] > z[i, label], p[i, neg] >= THRESHOLD,
                           float(cosines[i, label])-float(cosines[i, neg]) if cosines is not None else None))
        def mean(index):
            return float(np.mean([row[index] for row in values])) if values else None
        result.append({'eligible_gt_instances': len(values), 'margin': mean(0), 'logit_margin': mean(1),
                       'inversion_rate': mean(2), 'conditional_fp_rate': mean(3),
                       'cosine_margin': mean(4) if cosines is not None else None})
    return result


def delta(a, b):
    return b-a if a is not None and b is not None else None


def per_label(y, z0, z1, labels, frequencies, cos0=None, cos1=None):
    c0, c1 = counts(y, probability(z0)), counts(y, probability(z1))
    parents = [r['coarse_id'] for r in labels]
    s0, s1 = sibling_per_label(y, z0, parents, cos0), sibling_per_label(y, z1, parents, cos1)
    rows = []
    for i, label in enumerate(labels):
        row = {'label_id': i, 'label': label['fine_label'], 'branch': label['coarse_label'],
               'is_rare': i in RARE, 'structurally_peer_eligible': parents.count(parents[i]) >= 2,
               'train_frequency': int(frequencies[i]), 'support': int(c0['support'][i])}
        for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'prediction_count'):
            cast = int if key in ('tp', 'fp', 'fn', 'prediction_count') else float
            row['exp2_'+key], row['exp3_'+key] = cast(c0[key][i]), cast(c1[key][i])
            row['delta_'+key] = row['exp3_'+key]-row['exp2_'+key]
        row['eligible_gt_instances'] = s0[i]['eligible_gt_instances']
        require(row['eligible_gt_instances'] == s1[i]['eligible_gt_instances'], 'Sibling denominator differs')
        for key in ('margin', 'logit_margin', 'inversion_rate', 'conditional_fp_rate', 'cosine_margin'):
            row['exp2_sibling_'+key], row['exp3_sibling_'+key] = s0[i][key], s1[i][key]
            row['delta_sibling_'+key] = delta(s0[i][key], s1[i][key])
        rows.append(row)
    return rows


def branches(rows):
    result = []
    for branch in BRANCHES:
        part = [r for r in rows if r['branch'] == branch]
        require(part, 'Missing frozen hierarchy branch')
        row = {'branch': branch, 'fine_labels': len(part), 'total_support': sum(r['support'] for r in part)}
        for tag in ('exp2', 'exp3', 'delta'):
            for key in ('f1', 'precision', 'recall'):
                row[tag+'_mean_'+key] = float(np.mean([r[tag+'_'+key] for r in part]))
            for key in ('tp', 'fp', 'fn'):
                row[tag+'_total_'+key] = sum(r[tag+'_'+key] for r in part)
        result.append(row)
    return result


def gain_summary(rows):
    def subset(part):
        improved = [r for r in part if r['delta_f1'] > 1e-12]
        decreased = [r for r in part if r['delta_f1'] < -1e-12]
        return {'labels': len(part), 'improved': len(improved), 'decreased': len(decreased),
                'tied': len(part)-len(improved)-len(decreased),
                'improved_labels': [r['label'] for r in improved],
                'decreased_labels': [r['label'] for r in decreased],
                'tied_labels': [r['label'] for r in part if abs(r['delta_f1']) <= 1e-12]}
    return {**{name: subset(part) for name, part in {
        'all': rows, 'rare': [r for r in rows if r['is_rare']],
        'nonrare': [r for r in rows if not r['is_rare']],
        'peer_eligible': [r for r in rows if r['structurally_peer_eligible']],
        'neutral': [r for r in rows if r['branch'] == 'Neutral']}.items()},
        'top5_improved': sorted([r for r in rows if r['delta_f1'] > 1e-12], key=lambda r: -r['delta_f1'])[:5],
        'top5_degraded': sorted([r for r in rows if r['delta_f1'] < -1e-12], key=lambda r: r['delta_f1'])[:5]}


def frequency_correlations(rows):
    return {key: spearman([r['train_frequency'] for r in rows], [r[key] for r in rows]) for key in
            ('delta_f1', 'delta_precision', 'delta_recall', 'delta_fp', 'delta_prediction_count')}


def sibling_correlations(rows):
    part = [r for r in rows if r['structurally_peer_eligible']]
    return {key: spearman([r[key] for r in part], [r['delta_f1'] for r in part]) for key in
            ('delta_sibling_margin', 'delta_sibling_inversion_rate', 'delta_sibling_conditional_fp_rate')}


def score_shift(y, z0, z1, labels):
    rows = []
    for i, label in enumerate(labels):
        row = {'label': label['fine_label'], 'support': int(y[:, i].sum())}
        for tag, z in [('exp2', z0), ('exp3', z1)]:
            p = probability(z)[:, i]
            for key, mask in [('gt_positive_score', y[:, i].astype(bool)),
                              ('gt_negative_score', ~y[:, i].astype(bool)), ('predicted_positive_score', p >= THRESHOLD)]:
                row[tag+'_'+key] = float(p[mask].mean()) if mask.any() else None
            row[tag+'_separation'] = delta(row[tag+'_gt_negative_score'], row[tag+'_gt_positive_score'])
        for key in ('gt_positive_score', 'gt_negative_score', 'predicted_positive_score', 'separation'):
            row['delta_'+key] = delta(row['exp2_'+key], row['exp3_'+key])
        rows.append(row)
    return rows


def activity(y, z0, z1):
    result = {}
    for tag, z in [('exp2', z0), ('exp3', z1)]:
        p = probability(z)
        n = (p >= THRESHOLD).sum(1)
        label_f1 = counts(y, p)['f1']
        result[tag] = {**overall(y, p), 'total_predicted_positives': int(n.sum()),
                       'rare_mean_f1': float(label_f1[RARE].mean()),
                       'nonrare_mean_f1': float(label_f1[[i for i in range(28) if i not in RARE]].mean()),
                       'count_distribution': {key: {'examples': int(mask.sum()), 'fraction': float(mask.mean())}
                           for key, mask in [('0', n == 0), ('1', n == 1), ('2', n == 2), ('3+', n >= 3)]}}
    result['delta'] = {key: delta(result['exp2'][key], result['exp3'][key]) for key in
        ('micro_f1', 'macro_f1', 'rare_mean_f1', 'nonrare_mean_f1', 'micro_precision', 'micro_recall', 'average_predicted_label_count',
         'total_predicted_positives', 'tp', 'fp', 'fn')}
    result['protocol'] = 'Fixed threshold 0.30; Micro pools TP/FP/FN; Macro includes all 28 labels with zero_division=0.'
    return result


def example_groups(y, z0, z1, parents):
    mask = eligible_mask(y, parents)
    subsets = {'single_gt': y.sum(1) == 1, 'multi_gt': y.sum(1) > 1,
               'actual_peer_eligible_example': mask.any(1), 'no_eligible_peer_term_example': ~mask.any(1)}
    rows = []
    for subset, active in subsets.items():
        if not active.any():
            continue
        a, b = overall(y[active], probability(z0[active])), overall(y[active], probability(z1[active]))
        for tag, value in [('exp2', a), ('exp3', b)]:
            rows.append({'subset': subset, 'model': tag, **value})
        rows.append({'subset': subset, 'model': 'delta', **{k: delta(a[k], b[k]) for k in a}})
    return rows


def eligibility_labels(y, z0, z1, parents):
    rows = []
    for tag, indices in [('structural_peer_eligible', [i for i, p in enumerate(parents) if parents.count(p) >= 2]),
                         ('structural_noneligible', [i for i, p in enumerate(parents) if parents.count(p) < 2])]:
        a, b = overall(y, probability(z0), indices), overall(y, probability(z1), indices)
        for model, values in [('exp2', a), ('exp3', b), ('delta', {k: delta(a[k], b[k]) for k in a})]:
            rows.append({'group': tag, 'model': model, **values})
    return rows


def error_context(y, z, parents):
    """Descriptive overlap, NOT causal attribution of errors to sibling confusion."""
    y, pred, parents = y.astype(bool), probability(z) >= THRESHOLD, np.asarray(parents)
    linked_fp = total_fp = fn_with_fp_sibling = total_fn = fn_with_outranking_sibling = 0
    for i in range(len(y)):
        gt, wrong, missed = np.flatnonzero(y[i]), np.flatnonzero(pred[i] & ~y[i]), np.flatnonzero(~pred[i] & y[i])
        total_fp += len(wrong); total_fn += len(missed)
        linked_fp += sum(any(parents[n] == parents[t] for t in gt) for n in wrong)
        for t in missed:
            fn_with_fp_sibling += any(parents[n] == parents[t] for n in wrong)
            candidates = np.flatnonzero((parents == parents[t]) & ~y[i])
            fn_with_outranking_sibling += bool(len(candidates) and z[i, candidates].max() > z[i, t])
    return {'total_fp': total_fp, 'sibling_linked_fp': linked_fp, 'other_fp': total_fp-linked_fp,
            'sibling_linked_fp_fraction': linked_fp/total_fp if total_fp else None,
            'total_fn': total_fn, 'fn_with_predicted_fp_sibling': fn_with_fp_sibling,
            'fn_without_predicted_fp_sibling': total_fn-fn_with_fp_sibling,
            'fn_with_outranking_non_gt_sibling': fn_with_outranking_sibling,
            'definition': 'FP linked if any GT shares parent; FN conditions can overlap and are not causal categories.'}


def rare_details(y, z0, z1, labels, perlabel, history, snapshot_pairs):
    parents = [r['coarse_id'] for r in labels]
    result = []
    for name in ('grief', 'relief'):
        t = next(i for i, r in enumerate(labels) if r['fine_label'] == name)
        row = {k: perlabel[t][k] for k in ('label', 'support', 'exp2_tp', 'exp3_tp', 'exp2_fp', 'exp3_fp',
                  'exp2_fn', 'exp3_fn', 'exp2_precision', 'exp3_precision', 'exp2_recall', 'exp3_recall',
                  'exp2_f1', 'exp3_f1', 'exp2_prediction_count', 'exp3_prediction_count')}
        for tag, z in [('exp2', z0), ('exp3', z1)]:
            p, gt_scores, wrong_scores, sibling_scores = probability(z), [], [], []
            for i in np.flatnonzero(y[:, t]):
                candidates = np.flatnonzero(~y[i].astype(bool))
                siblings = [n for n in candidates if parents[n] == parents[t]]
                gt_scores.append(p[i, t]); wrong_scores.append(float(p[i, candidates].max()))
                if siblings:
                    sibling_scores.append(float(p[i, siblings].max()))
            row[tag+'_gt_positive_score'] = float(np.mean(gt_scores))
            row[tag+'_top_wrong_all_labels_score'] = float(np.mean(wrong_scores))
            row[tag+'_margin_to_top_wrong_all_labels'] = float(np.mean(np.array(gt_scores)-wrong_scores))
            row[tag+'_top_wrong_sibling_score'] = float(np.mean(sibling_scores)) if sibling_scores else None
        historical = Counter()
        for step in history:
            for pair, n in step['selected_pairs'].items():
                a, b = map(int, pair.split('->'))
                if a == t:
                    historical[labels[b]['fine_label']] += n
        snapshot = [r for r in snapshot_pairs if r['true_label'] == name]
        row['historical_negative_counts'] = json.dumps(dict(historical.most_common()), sort_keys=True)
        row['snapshot_negative_loss_sums'] = json.dumps({r['negative_label']: r['raw_loss_sum'] for r in snapshot}, sort_keys=True)
        row['interpretation_limit'] = 'Tiny support; counts/score shifts describe cases, not a general mechanism.'
        result.append(row)
    return result


def gradient_summary(rows, near=.05):
    result = {}
    for scope in sorted({r['scope'] for r in rows}):
        part = [r for r in rows if r['scope'] == scope]
        values = np.array([r['cosine'] for r in part if r['cosine'] is not None])
        result[scope] = {'batches': len(part), 'valid_cosines': len(values),
            'mean': float(values.mean()) if len(values) else None,
            'sd_sample': float(values.std(ddof=1)) if len(values) > 1 else None,
            'min': float(values.min()) if len(values) else None, 'max': float(values.max()) if len(values) else None,
            'fraction_negative': float((values < 0).mean()) if len(values) else None,
            'fraction_positive': float((values > 0).mean()) if len(values) else None,
            'fraction_near_zero_abs_le_0.05': float((abs(values) <= near).mean()) if len(values) else None,
            'fraction_conflicting_lt_minus_0.05': float((values < -near).mean()) if len(values) else None,
            'fraction_aligned_gt_0.05': float((values > near).mean()) if len(values) else None,
            'note': 'Negative/positive fractions overlap the near-zero category; strict bins do not. Zero norm is unavailable.'}
    return result
