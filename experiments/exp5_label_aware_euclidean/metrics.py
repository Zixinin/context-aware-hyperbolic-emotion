"""Fixed-threshold multi-label metrics; test data is never used for tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from data import LABEL_NAMES, RARE_IDS


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    return np.exp(-np.logaddexp(0.0, -logits))


def score(targets, probs, threshold):
    targets, probs = np.asarray(targets), np.asarray(probs)
    if targets.shape != probs.shape or targets.ndim != 2 or targets.shape[1] != 28:
        raise ValueError("Expected matching [N, 28] targets/probabilities")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probs).all():
        raise ValueError("Invalid target/probability values")
    predicted = (probs >= threshold).astype(np.int64)
    result = {"threshold": float(threshold), "num_examples": int(len(targets))}
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predicted, average=average, zero_division=0
        )
        result.update(
            {
                f"{average}_precision": float(precision),
                f"{average}_recall": float(recall),
                f"{average}_f1": float(f1),
            }
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, average=None, zero_division=0
    )
    frame = pd.DataFrame(
        {
            "label_id": range(28),
            "label": LABEL_NAMES,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return result, frame.sort_values(["f1", "label_id"]).reset_index(drop=True)


def tune_validation_threshold(targets, probs, candidates):
    rows = [score(targets, probs, threshold)[0] for threshold in sorted(candidates)]
    selected = max(rows, key=lambda row: row["macro_f1"])["threshold"]
    return selected, pd.DataFrame(rows)


def rare_scores(frame):
    is_rare = frame["label_id"].isin(RARE_IDS)
    return {
        "rare_label_names": [LABEL_NAMES[index] for index in RARE_IDS],
        "mean_rare_label_f1": float(frame.loc[is_rare, "f1"].mean()),
        "mean_non_rare_label_f1": float(frame.loc[~is_rare, "f1"].mean()),
    }
