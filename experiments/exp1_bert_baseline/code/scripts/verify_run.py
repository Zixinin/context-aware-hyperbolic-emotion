"""Artifact integrity + sacct + log checks; a PID or SUCCESS marker alone is insufficient."""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from src.common import ROOT, load_config, read_json, sha256, source_hashes


def verify_artifacts(run_dir, require_success=True):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    meta = read_json(run_dir / "environment.json")
    if meta["status"] != "completed" or not meta.get("bce_loss_verified"):
        raise RuntimeError("Run did not complete verified BCE training.")
    required = ["config.json", "environment.json", "pip-freeze.txt", "training_log.csv", "sample_ids.json",
                "validation_metrics.json", "threshold_selection.json", "threshold_search.csv", "per_label_metrics.csv",
                "rare_label_metrics.json", "validation_predictions.npz", "summary.md", "checkpoints/best/config.json",
                "checkpoints/best/model.safetensors", "checkpoints/best/tokenizer_config.json", "checkpoints/best/selection.json",
                "dataset_analysis/rare_labels.json", "dataset_analysis/dataset_statistics.json",
                "figures/training_loss.png", "figures/validation_macro_f1.png", "figures/label_frequency.png",
                "figures/per_label_f1.png", "figures/frequency_vs_f1.png"]
    if config["mode"] == "full":
        required += ["test_metrics.json", "test_predictions.npz", "test_per_label_fixed_0.30.csv", "test_per_label_validation_selected.csv"]
    elif not meta.get("checkpoint_reload_verified"):
        raise RuntimeError("Smoke checkpoint reload did not pass.")
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing/empty artifact: {path}")
        if path.suffix == ".json":
            read_json(path)
    with (run_dir / "training_log.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["epochs"] or int(rows[-1]["epoch"]) != config["epochs"]:
        raise RuntimeError("Incorrect number of completed training epochs.")
    if meta["completed_optimizer_steps"] != meta["total_optimizer_steps"]:
        raise RuntimeError("Incomplete optimizer steps.")
    with (run_dir / "threshold_search.csv").open() as stream:
        if len(list(csv.DictReader(stream))) != 18:
            raise RuntimeError("Expected all 18 global threshold candidates.")
    with (run_dir / "per_label_metrics.csv").open() as stream:
        if len(list(csv.DictReader(stream))) != 28:
            raise RuntimeError("Expected metrics for all 28 labels.")
    if config["mode"] == "smoke" and (run_dir / "test_metrics.json").exists():
        raise RuntimeError("Smoke must not evaluate the test set.")
    if require_success:
        success = read_json(run_dir / "SUCCESS.json")
        if success["slurm_job_id"] != meta["slurm_job_id"]:
            raise RuntimeError("Success marker has wrong Slurm job ID.")
        for name, digest in success["artifacts_sha256"].items():
            if sha256(run_dir / name) != digest:
                raise RuntimeError(f"Artifact checksum mismatch: {name}")
    return meta, config


def verify_slurm_and_logs(meta, mode):
    job_id = meta["slurm_job_id"]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Expected numeric Slurm job ID.")
    result = subprocess.check_output(["sacct", "-n", "-P", "-j", job_id,
                                      "--format=JobIDRaw,State,ExitCode"], text=True)
    rows = [line.strip().split("|") for line in result.splitlines() if line.strip()]
    main_rows = [row for row in rows if row[0] == job_id]
    if len(main_rows) != 1 or main_rows[0][1:3] != ["COMPLETED", "0:0"]:
        raise RuntimeError(f"Job {job_id} must be COMPLETED / 0:0. sacct may need a few seconds to update.\n{result}")
    for row in rows:
        if row[0].startswith(job_id + ".") and row[1:3] != ["COMPLETED", "0:0"]:
            raise RuntimeError(f"A Slurm step failed: {row}")
    stem = "smoke" if mode == "smoke" else "bert42"
    for suffix in ("out", "err"):
        log = ROOT / f"logs/slurm-{stem}-{job_id}.{suffix}"
        if not log.is_file():
            raise RuntimeError(f"Missing Slurm log {log}")
        if re.search(r"Traceback|CUDA out of memory|Killed|Segmentation fault|Exception", log.read_text(errors="replace")):
            raise RuntimeError(f"Serious error marker found in {log}")
    return result


def verify_smoke_gate(run_dir, full_config):
    meta, smoke = verify_artifacts(run_dir)
    if smoke["mode"] != "smoke":
        raise RuntimeError("Expected smoke run.")
    verify_slurm_and_logs(meta, "smoke")
    if meta["source_sha256"] != source_hashes():
        raise RuntimeError("Code/config changed since smoke; run smoke again.")
    if meta["assets_manifest_sha256"] != sha256(ROOT / "assets/manifest.json"):
        raise RuntimeError("Prepared data/model changed since smoke.")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if freeze != (Path(run_dir) / "pip-freeze.txt").read_text():
        raise RuntimeError("Python packages changed since smoke.")
    for key, value in full_config.items():
        if key not in ("epochs", "mode") and smoke.get(key) != value:
            raise RuntimeError(f"Configuration differs from smoke: {key}")
    print(f"SMOKE GATE PASS: job {meta['slurm_job_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-slurm", action="store_true")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.gate:
        config = load_config()
        config["mode"] = "full"
        verify_smoke_gate(args.run_dir, config)
    else:
        meta, config = verify_artifacts(args.run_dir)
        if args.check_slurm:
            print(verify_slurm_and_logs(meta, config["mode"]))
        print("ARTIFACT CHECK PASS", args.run_dir)


if __name__ == "__main__":
    main()
