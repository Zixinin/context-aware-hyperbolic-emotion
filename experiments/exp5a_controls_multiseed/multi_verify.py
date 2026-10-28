"""Independent CPU verification of control artifacts, histories and saved predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multi_common import (check_sources, checked_artifacts, config_for, freeze, load_all_verified,
                            read, require, sha, verify_prediction_metrics)
from train import subset
from metrics import probabilities, tune_validation_threshold
from data import LABEL_NAMES, RARE_IDS, frequencies


def verify_control(root, require_success=True):
    root = Path(root)
    config = read(root / "config.json")
    mode, representation = config["mode"], config["representation"]
    require(config == config_for(mode, representation, config["seed"]), "Control config drift")
    env = read(root / "environment.json")
    require(env["status"] == "completed", "Control run incomplete")
    require(env["source_sha256"] == check_sources(), "Source changed")
    require(env["hostname"] == "D3020-A3" and "4090" in env["gpu"], "Wrong target host/GPU")
    require(env["first_batch_backward_verified"] and env["checkpoint_reload_verified"], "Training/reload gates missing")
    preflight = read(env["preflight_path"])
    require(preflight["status"] == "EXP5A_MULTISEED_PREFLIGHT_PASS", "Preflight status invalid")
    require(sha(env["preflight_path"]) == env["preflight_sha256"], "Preflight proof changed")
    require(preflight["source_sha256"] == env["source_sha256"], "Audit source differs")
    require(preflight["parameters"][representation] == env["parameters_runtime"], "Parameter proof differs")
    require(env["seed42_reference"] == preflight["seed42_reference"], "C provenance differs")
    require(env["packages"] == preflight["packages"], "Environment versions differ")
    require((root / "pip-freeze.txt").read_text() == preflight["pip_freeze"] == freeze(), "Pip environment drift")
    required = ["checkpoint_best.pt", "checkpoint_tokenizer/tokenizer_config.json", "first_batch_shapes.json",
                "dataset_statistics.json", "sample_ids.json", "checkpoint_selection.json", "training_log.csv",
                "threshold_search.csv", "threshold_selection.json", "per_label_metrics.csv"]
    require(all((root / name).is_file() and (root / name).stat().st_size > 0 for name in required), "Required artifact missing")
    shapes = read(root / "first_batch_shapes.json")
    require(shapes["logits"] == shapes["targets"] and shapes["logits"][1] == 28 and
            shapes["pooled"] == [16, 768] and shapes["query_parameters"] == 0, "Shape/query proof invalid")
    datasets, _ = load_all_verified()
    statistics = read(root / "dataset_statistics.json")
    require(statistics == dict(splits={k: len(v) for k, v in datasets.items()}, num_labels=28,
            label_order={str(i): name for i, name in enumerate(LABEL_NAMES)},
            train_label_counts=frequencies(datasets["train"]).tolist(), rare_ids=RARE_IDS), "Dataset statistics differ")
    training, validation = datasets["train"], datasets["validation"]
    if mode == "smoke":
        training = subset(training, config["smoke_train_samples"], config["seed"])
        validation = subset(validation, config["smoke_validation_samples"], config["seed"])
        require(not (root / "test_predictions.npz").exists() and not (root / "test_metrics.json").exists(), "Smoke touched test predictions")
    require(read(root / "sample_ids.json") == dict(train=[r["id"] for r in training], validation=[r["id"] for r in validation]), "Actual training/validation IDs differ")
    val_metrics, _, _ = verify_prediction_metrics(root, "validation", validation)
    history = pd.read_csv(root / "training_log.csv")
    expected_steps = 10856 if mode == "full" else 63
    require(len(history) == config["epochs"] and history.epoch.tolist() == list(range(1, config["epochs"] + 1)), "Epoch count/order differs")
    require(history.global_step.tolist() == [expected_steps // config["epochs"] * i for i in range(1, config["epochs"] + 1)], "Step history differs")
    require(np.isfinite(history.to_numpy()).all(), "Non-finite training log")
    require(env["completed_optimizer_steps"] == env["total_optimizer_steps"] == expected_steps, "Optimizer steps incomplete")
    best_index = int(history.validation_macro_f1.to_numpy().argmax())
    selection = read(root / "checkpoint_selection.json")
    require(selection["epoch"] == best_index + 1 and selection["global_step"] == int(history.iloc[best_index].global_step), "Wrong checkpoint epoch/tie rule")
    require(selection["threshold"] == 0.30, "Wrong checkpoint threshold")
    require(abs(selection["validation_macro_f1"] - val_metrics["macro_f1"]) < 1e-12, "Checkpoint metric differs")
    require(abs(history.iloc[best_index].validation_macro_f1 - val_metrics["macro_f1"]) < 1e-12, "Selected history differs")
    require(abs(history.iloc[best_index].validation_micro_f1 - val_metrics["micro_f1"]) < 1e-12, "Selected Micro-F1 differs")
    with np.load(root / "validation_predictions.npz", allow_pickle=False) as pred:
        selected, grid = tune_validation_threshold(pred["targets"], probabilities(pred["logits"]), config["threshold_candidates"])
    saved_grid = pd.read_csv(root / "threshold_search.csv")
    require(list(saved_grid.columns) == list(grid.columns), "Threshold grid columns differ")
    np.testing.assert_allclose(saved_grid, grid, rtol=0, atol=1e-12)
    require(read(root / "threshold_selection.json") == dict(selected_threshold=selected, selection_split="validation",
            objective="macro_f1", tie_break="lowest threshold", formal_comparison_threshold=0.30,
            per_label_thresholds=False), "Threshold selection protocol differs")
    if mode == "full":
        verify_prediction_metrics(root, "test", datasets["test"])
    canonical_per_label = root / ("test_fixed_0.30_per_label.csv" if mode == "full" else "validation_fixed_0.30_per_label.csv")
    require(sha(root / "per_label_metrics.csv") == sha(canonical_per_label), "Per-label copies disagree")
    if require_success:
        success = checked_artifacts(root)
        expected_status = "EXP5A_MULTISEED_MODEL_VERIFIED" if mode == "full" else "EXP5A_MULTISEED_SMOKE_VERIFIED"
        require(success["status"] == expected_status and success["source_sha256"] == env["source_sha256"] and
                success["representation"] == representation and success["mode"] == mode and success["seed"] == config["seed"], "Success identity invalid")
    return dict(status="EXP5A_MULTISEED_ARTIFACT_CHECK_PASS", mode=mode, seed=config["seed"], representation=representation, run_dir=str(root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(verify_control(args.run_dir), flush=True)
