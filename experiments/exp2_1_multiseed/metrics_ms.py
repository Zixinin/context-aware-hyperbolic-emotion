"""Recompute reports from saved predictions, preserving each legacy sigmoid precision."""
import numpy as np
from mslib import NAMES, RARE, CANDIDATES, METRICS, require


def probabilities(logits, model):
    require(np.isfinite(logits).all(), 'Nonfinite logits')
    if model == 'bert':
        return np.exp(-np.logaddexp(0., -np.asarray(logits, dtype=np.float64)))
    return 1. / (1. + np.exp(-np.clip(logits, -80, 80)))


def score(gold, probs, threshold):
    require(gold.shape == probs.shape and gold.ndim == 2 and gold.shape[1] == 28, 'Expected N x 28')
    require(np.isfinite(probs).all() and np.isin(gold, [0, 1]).all(), 'Invalid predictions/targets')
    y, p = gold.astype(bool), probs >= threshold
    tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    prec, rec, f1 = div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    other = [i for i in range(28) if i not in RARE]
    summary = dict(zip(METRICS, map(float, [div(tp.sum(), (tp+fp).sum()), div(tp.sum(), (tp+fn).sum()),
        div(2*tp.sum(), (2*tp+fp+fn).sum()), prec.mean(), rec.mean(), f1.mean(), f1[RARE].mean(), f1[other].mean()])))
    summary.update(threshold=float(threshold), num_examples=len(y), num_labels=28)
    rows = [dict(label_id=i, label=NAMES[i], support=int(tp[i]+fn[i]), precision=float(prec[i]),
                 recall=float(rec[i]), f1=float(f1[i]), tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
                 number_predicted_positive=int(tp[i]+fp[i])) for i in range(28)]
    return summary, rows


def load_predictions(path):
    with np.load(path, allow_pickle=False) as f:
        y = f['gold'] if 'gold' in f else f['targets']
        z, ids = f['logits'], f['ids'].astype(str)
    require(y.shape == z.shape and z.shape == (len(ids), 28), 'Prediction shape mismatch')
    require(len(set(ids)) == len(ids), 'Duplicate prediction IDs')
    require(np.isfinite(z).all() and np.isin(y, [0, 1]).all(), 'Invalid saved predictions')
    return y, z, ids


def threshold_search(gold, logits, model):
    # BERT uses its original sklearn-based macro arithmetic (possible last-bit ties).
    p = probabilities(logits, model)
    rows = [score(gold, p, t)[0] for t in CANDIDATES]
    if model == 'bert':
        from sklearn.metrics import precision_recall_fscore_support
        for row in rows:
            row['macro_f1'] = float(precision_recall_fscore_support(gold, p >= row['threshold'],
                                        average='macro', zero_division=0)[2])
    return max(rows, key=lambda r: r['macro_f1'])['threshold'], rows


def close_metrics(actual, expected):
    aliases = {'rare_f1': 'rare_mean_f1', 'nonrare_f1': 'nonrare_mean_f1'}
    for key in METRICS:
        ekey = key if key in expected else aliases.get(key, key)
        if ekey in expected:
            require(abs(actual[key] - expected[ekey]) < 1e-10, 'Recomputed metric differs: ' + key)
