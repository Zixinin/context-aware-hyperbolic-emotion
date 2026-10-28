"""Correct argument direction, inclusive 0.30 threshold, full saved-prediction audit."""
from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from common import RARE, require, write


def probabilities(logits):
    require(np.isfinite(logits).all(), "Nonfinite saved logits")
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64)))


def score(gold, probs, names):
    gold, probs = np.asarray(gold), np.asarray(probs)
    require(gold.shape == probs.shape and gold.ndim == 2 and gold.shape[1] == 27 and len(gold) > 0, "Expected [N,27]")
    require(len(names) == 27 and len(set(names)) == 27 and set(RARE) <= set(names), "Wrong metric label mapping")
    require(np.isin(gold, [0, 1]).all() and np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all(), "Invalid gold/probabilities")
    pred = (probs >= 0.30).astype(np.int64)
    report = dict(threshold=0.30, num_examples=len(gold))
    for average in ["micro", "macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(gold, pred, average=average, zero_division=0)
        report.update({average + "_precision": float(p), average + "_recall": float(r), average + "_f1": float(f)})
    p, r, f, support = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)
    tp = ((gold == 1) & (pred == 1)).sum(0)
    fp = ((gold == 0) & (pred == 1)).sum(0)
    fn = ((gold == 1) & (pred == 0)).sum(0)
    rows = [dict(label_id=i, label=n, support=int(support[i]), tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
                 precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                 predicted_positive_count=int(pred[:, i].sum()), true_positive_count=int(gold[:, i].sum()))
            for i, n in enumerate(names)]
    rare = [rows[names.index(n)] for n in RARE]
    report.update(rare_f1=float(np.mean([x["f1"] for x in rare])),
                  total_predicted_positives=int(pred.sum()), total_gold_positives=int(gold.sum()),
                  mean_predictions_per_example=float(pred.sum(1).mean()),
                  mean_gold_positives_per_example=float(gold.sum(1).mean()),
                  zero_prediction_examples=int((pred.sum(1) == 0).sum()),
                  zero_prediction_labels=[names[i] for i in range(27) if pred[:, i].sum() == 0])
    return report, rows, dict(definition="unweighted mean per-label F1 over fixed seven labels", labels=RARE,
                              rare_f1=report["rare_f1"], per_label=rare), pred


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(root, split, logits, gold, ids, names):
    probs = probabilities(logits)
    metrics, rows, rare, pred = score(gold, probs, names)
    np.savez_compressed(root / f"{split}_predictions.npz", logits=logits, probabilities=probs,
        targets=gold, ids=np.asarray(ids, dtype=str), label_names=np.asarray(names, dtype=str),
        predictions=pred, predicted_positive_counts=pred.sum(1), true_positive_counts=gold.sum(1))
    write(root / f"{split}_metrics.json", metrics)
    write_csv(root / f"{split}_per_label_metrics.csv", rows)
    return metrics, rare
