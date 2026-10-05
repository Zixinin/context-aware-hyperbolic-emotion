"""Independent sigmoid decisions; no forced top-1 label or test-set tuning."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .data import LABEL_NAMES


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits.")
    return np.exp(-np.logaddexp(0.0, -logits))


def score(targets, probs, threshold):
    targets, probs = np.asarray(targets), np.asarray(probs)
    if targets.shape != probs.shape or targets.ndim != 2 or targets.shape[1] != 28:
        raise ValueError("Expected matching (N, 28) targets/probabilities.")
    if not np.isfinite(probs).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Invalid targets or predictions.")
    predicted = (probs >= threshold).astype(np.int64)
    result = {"threshold": float(threshold), "num_examples": len(targets)}
    for average in ("micro", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predicted, average=average, zero_division=0)
        result.update({f"{average}_precision": float(precision), f"{average}_recall": float(recall),
                       f"{average}_f1": float(f1)})
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, average=None, zero_division=0)
    frame = pd.DataFrame({"label_id": range(28), "label": LABEL_NAMES, "support": support.astype(int),
                          "precision": precision, "recall": recall, "f1": f1})
    return result, frame.sort_values(["f1", "label_id"]).reset_index(drop=True)


def tune_validation_threshold(targets, probs, candidates):
    rows = [score(targets, probs, threshold)[0] for threshold in sorted(candidates)]
    # Python max keeps the first maximum: lower threshold wins exact ties.
    selected = max(rows, key=lambda row: row["macro_f1"])["threshold"]
    return selected, pd.DataFrame(rows)


def rare_label_scores(frame, rare_ids):
    is_rare = frame["label_id"].isin(rare_ids)
    return {"rare_label_names": [LABEL_NAMES[i] for i in rare_ids],
            "mean_rare_label_f1": float(frame.loc[is_rare, "f1"].mean()),
            "mean_non_rare_label_f1": float(frame.loc[~is_rare, "f1"].mean())}
