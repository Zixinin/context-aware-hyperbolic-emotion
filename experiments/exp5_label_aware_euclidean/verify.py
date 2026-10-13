"""Artifact and provenance verification for Exp5A smoke/full runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from exp5lib import read, require, sha, source_hashes


def verify_run(run_dir, require_success=True):
    run_dir = Path(run_dir)
    config = read(run_dir / "config.json")
    metadata = read(run_dir / "environment.json")
    require(metadata["status"] == "completed", "Run metadata is not completed")
    require(metadata.get("first_batch_backward_verified") is True, "First-batch backward gate missing")
    require(metadata["source_sha256"] == source_hashes(), "Current source differs from run source")
    required = [
        "config.json", "environment.json", "pip-freeze.txt", "sample_ids.json",
        "dataset_statistics.json", "training_log.csv", "first_batch_shapes.json",
        "checkpoint_best.pt", "checkpoint_tokenizer/tokenizer_config.json",
        "checkpoint_selection.json", "validation_predictions.npz",
        "validation_metrics.json", "validation_fixed_0.30_metrics.json",
        "validation_fixed_0.30_per_label.csv", "threshold_search.csv",
        "threshold_selection.json", "rare_label_metrics.json", "per_label_metrics.csv",
    ]
    if config["mode"] == "full":
        required += [
            "test_predictions.npz", "test_metrics.json", "test_fixed_0.30_metrics.json",
            "test_fixed_0.30_per_label.csv", "exp1_vs_exp5a_per_label.csv",
            "attention_examples.json", "attention_examples.md", "EXP5A_SUMMARY.md",
        ]
    else:
        require(metadata.get("checkpoint_reload_verified") is True, "Smoke checkpoint reload gate missing")
        require(not (run_dir / "test_metrics.json").exists(), "Smoke must not evaluate test data")
    for name in required:
        path = run_dir / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty artifact: {name}")
        if path.suffix == ".json":
            read(path)
    with (run_dir / "training_log.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    require(len(history) == config["epochs"], "Training epoch count is incomplete")
    require(metadata["completed_optimizer_steps"] == metadata["total_optimizer_steps"], "Optimizer steps are incomplete")
    with (run_dir / "threshold_search.csv").open(newline="") as stream:
        require(len(list(csv.DictReader(stream))) == 18, "Threshold grid is incomplete")
    with (run_dir / "per_label_metrics.csv").open(newline="") as stream:
        require(len(list(csv.DictReader(stream))) == 28, "Per-label metrics must contain 28 labels")
    shapes = read(run_dir / "first_batch_shapes.json")
    require(shapes["label_queries"][0] == 28 and shapes["logits"][1] == 28, "First-batch shape proof is invalid")
    require(shapes["pad_attention_max"] == 0.0, "PAD attention proof is invalid")
    if require_success:
        success = read(run_dir / "SUCCESS.json")
        expected = "EXPERIMENT_5A_COMPLETE_VERIFIED" if config["mode"] == "full" else "EXP5A_SMOKE_COMPLETE_VERIFIED"
        require(success["status"] == expected, "Success marker has wrong identity")
        for name, digest in success["artifacts_sha256"].items():
            require(sha(run_dir / name) == digest, f"Artifact changed after completion: {name}")
    return {"status": "EXP5A_ARTIFACT_CHECK_PASS", "mode": config["mode"], "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify_run(args.run_dir)
    print(result["status"], result["mode"], result["run_dir"], flush=True)


if __name__ == "__main__":
    main()
