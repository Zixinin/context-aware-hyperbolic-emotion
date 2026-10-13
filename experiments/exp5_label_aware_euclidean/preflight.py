"""Read-only Ubuntu gate for environment, canonical data, raw BERT and Exp1 reference."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from data import LABEL_NAMES, RARE_IDS, SPLIT_SIZES, load_all_verified
from exp5lib import (
    ASSET_MANIFEST, EXPECTED_HOST, MODEL_DIR, MODEL_REVISION, REPO, exp1_reference_dir,
    load_config, package_versions, read, require, require_runtime, sha, source_hashes,
    verify_package_manifest, write,
)
from model import LabelAwareBert, parameter_report

EXPECTED_VERSIONS = {
    "torch": "2.6.0+cu118",
    "transformers": "4.51.3",
    "datasets": "3.5.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "safetensors": "0.5.3",
}


def verify_reference():
    root = exp1_reference_dir()
    manifest = read(root / "reference_manifest.json")
    required = {
        "test_metrics.json", "test_per_label_fixed_0.30.csv", "rare_label_metrics.json",
    }
    for name in required:
        require((root / name).is_file(), f"Exp1 reference missing: {name}")
        require(sha(root / name) == manifest["artifacts_sha256"][name], f"Exp1 reference hash mismatch: {name}")
    metrics = read(root / "test_metrics.json")["fixed_0.30"]
    require(metrics == manifest["metrics"]["fixed_0.30"], "Exp1 metric copies disagree")
    require(metrics["threshold"] == 0.3 and metrics["num_examples"] == 5427, "Wrong Exp1 formal protocol")
    with (root / "test_per_label_fixed_0.30.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 28 and {row["label"] for row in rows} == set(LABEL_NAMES), "Exp1 per-label table invalid")
    rare = read(root / "rare_label_metrics.json")
    require(rare["definition"]["label_ids"] == RARE_IDS, "Exp1 rare-label definition differs")
    return {
        "path": str(root),
        "manifest_sha256": sha(root / "reference_manifest.json"),
        "verified_present_artifacts": sorted(required),
        "historical_archive_claim": manifest["artifact_integrity"],
        "live_sacct_verified_this_session": manifest["live_sacct_verified_this_session"],
        "formal_metrics": metrics,
        "rare_f1_fixed_0.30": manifest["rare_f1_fixed_030"],
    }


def verify_raw_encoder(model, manifest):
    raw_path = MODEL_DIR / "model.safetensors"
    require(sha(raw_path) == manifest["model"]["sha256"]["model.safetensors"], "Raw BERT weights changed")
    count, aliases = 0, []
    with safe_open(str(raw_path), framework="pt", device="cpu") as raw:
        keys = set(raw.keys())
        for name, value in model.encoder.state_dict().items():
            candidates = ["bert." + name, name]
            for modern, legacy in (("weight", "gamma"), ("bias", "beta")):
                if name.endswith(".LayerNorm." + modern):
                    old = name[: -len(modern)] + legacy
                    candidates += ["bert." + old, old]
            key = next((candidate for candidate in candidates if candidate in keys), None)
            require(key is not None, f"Raw BERT tensor missing: {name}")
            reference = raw.get_tensor(key)
            require(value.dtype == reference.dtype, f"Raw BERT dtype mismatch: {name}")
            require(tuple(value.shape) == tuple(reference.shape), f"Raw BERT shape mismatch: {name}")
            require(torch.equal(value.detach().cpu(), reference), f"Raw BERT value mismatch: {name}")
            if key.endswith((".LayerNorm.gamma", ".LayerNorm.beta")):
                aliases.append(key)
            count += 1
    require(count == 197, f"Expected 197 no-pooler BERT tensors, got {count}")
    return {"tensors_compared": count, "legacy_layernorm_aliases": len(aliases), "pooler_present": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_runtime(require_cuda=True)
    package_manifest = verify_package_manifest()
    config = load_config()
    versions = package_versions()
    require({key: versions[key] for key in EXPECTED_VERSIONS} == EXPECTED_VERSIONS, f"Unexpected package versions: {versions}")
    unit = read(args.unit_proof)
    require(unit["status"] == "EXP5A_UNIT_TESTS_PASS", "Unit proof did not pass")
    require(unit["source_sha256"] == source_hashes(), "Source changed after unit tests")
    datasets, manifest = load_all_verified()
    require({split: len(rows) for split, rows in datasets.items()} == SPLIT_SIZES, "Split sizes changed")
    require(manifest["model"]["revision"] == MODEL_REVISION, "Wrong raw BERT revision")
    require(Path(manifest["model"]["path"]).resolve() == MODEL_DIR.resolve(), "Wrong raw BERT path")
    for name, digest in manifest["model"]["sha256"].items():
        require(sha(MODEL_DIR / name) == digest, f"Raw BERT asset changed: {name}")
    model = LabelAwareBert.from_pretrained(str(MODEL_DIR), config)
    raw = verify_raw_encoder(model, manifest)
    parameters = parameter_report(model)
    del model
    from transformers import AutoModelForSequenceClassification
    baseline = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True, use_safetensors=True, num_labels=28,
        problem_type="multi_label_classification", attn_implementation="eager",
    )
    parameters["exp1_total_trainable_parameters"] = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    parameters["exp5a_minus_exp1_trainable_parameters"] = (
        parameters["exp5a_trainable_parameters"] - parameters["exp1_total_trainable_parameters"]
    )
    del baseline
    report = {
        "status": "EXP5A_PREFLIGHT_PASS",
        "host": EXPECTED_HOST,
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": versions,
        "cuda": {"available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0), "runtime": torch.version.cuda},
        "config": config,
        "dataset_manifest_sha256": sha(ASSET_MANIFEST),
        "split_sizes": SPLIT_SIZES,
        "label_order": dict(enumerate(LABEL_NAMES)),
        "rare_ids": RARE_IDS,
        "raw_initialization": raw,
        "parameters": parameters,
        "exp1_reference": verify_reference(),
        "source_sha256": source_hashes(),
        "package_manifest": package_manifest,
        "unit_proof_sha256": sha(args.unit_proof),
        "research_boundary": "No LGSA, hierarchy, graph, hyperbolic, prototype, contrastive, imbalance loss or resampling",
    }
    write(args.out, report)
    print("EXP5A_PREFLIGHT_PASS", flush=True)
    print(json.dumps({"split_sizes": SPLIT_SIZES, "label_order": report["label_order"], "parameters": parameters}, indent=2), flush=True)


if __name__ == "__main__":
    main()
