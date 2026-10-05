"""Reusable evaluation on already-fixed logits; tuning API accepts validation only."""
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .common import write_json
from .metrics import probabilities, rare_label_scores, score, tune_validation_threshold
from .visualize import evaluation_figures


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, targets = [], []
    loss_sum, sample_count = 0.0, 0
    for batch in tqdm(loader, desc="Evaluation"):
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
            raise RuntimeError("Non-finite evaluation loss/logits.")
        size = batch["labels"].shape[0]
        loss_sum += float(output.loss) * size
        sample_count += size
        logits.append(output.logits.cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets), loss_sum / sample_count


def select_threshold(validation_logits, validation_targets, config, output_dir):
    selected, search = tune_validation_threshold(validation_targets, probabilities(validation_logits), config["threshold_candidates"])
    search.to_csv(Path(output_dir) / "threshold_search.csv", index=False)
    write_json(Path(output_dir) / "threshold_selection.json", {
        "selected_threshold": selected, "selection_split": "validation", "objective": "macro_f1",
        "tie_break": "lowest threshold", "checkpoint_selection_threshold": config["fixed_threshold"],
        "per_label_thresholds": False})
    return selected


def report_split(split, logits, targets, selected, config, output_dir, rare, frequencies):
    output_dir = Path(output_dir)
    probs = probabilities(logits)
    metrics, rare_metrics = {}, {}
    for protocol, threshold in (("fixed_0.30", config["fixed_threshold"]), ("validation_selected", selected)):
        values, frame = score(targets, probs, threshold)
        metrics[protocol] = values
        rare_metrics[protocol] = rare_label_scores(frame, rare["label_ids"])
        frame.to_csv(output_dir / f"{split}_per_label_{protocol}.csv", index=False)
        if protocol == "fixed_0.30":
            if split == "test" or config["mode"] == "smoke":
                frame.to_csv(output_dir / "per_label_metrics.csv", index=False)
            evaluation_figures(frame, frequencies, output_dir / "figures" / split, f"{split}: fixed threshold 0.30")
            if split == "test" or config["mode"] == "smoke":
                evaluation_figures(frame, frequencies, output_dir / "figures", f"{split}: fixed threshold 0.30")
    write_json(output_dir / f"{split}_metrics.json", metrics)
    return metrics, rare_metrics
