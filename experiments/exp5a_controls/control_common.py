"""Read-only reuse of frozen Exp5A infrastructure, with independent control gates."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = REPO / "experiments/exp5_label_aware_euclidean"
C_RUN = REPO / "results/exp5a/run_20261013T093754Z/seed42"
sys.path.insert(0, str(FROZEN))

from exp5lib import MODEL_DIR, ASSET_MANIFEST, read, write, sha, require, now, package_versions, require_runtime
from exp5lib import load_config as frozen_config, source_hashes as frozen_hashes
from data import LABEL_NAMES, RARE_IDS, load_all_verified


def sources():
    return {p.name: sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in {".py", ".json", ".sh", ".md"}}


def check_sources():
    expected = read(HERE / "frozen_dependency.json")
    require(frozen_hashes() == expected["source_sha256"], "Frozen Exp5A source changed")
    require(sha(FROZEN / "PACKAGE_MANIFEST.sha256") == expected["manifest_sha256"], "Frozen manifest changed")
    from exp5lib import verify_package_manifest
    verify_package_manifest()
    entries = {}
    for line in (HERE / "PACKAGE_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(Path(name).name == name, "Unsafe control manifest entry")
        entries[name] = digest
    require(entries == sources(), "Control package changed")
    return {"controls": sources(), "exp5a": frozen_hashes()}


def config_for(mode, representation):
    require(mode in {"smoke", "full"} and representation in {"cls", "mean"}, "Invalid control")
    cfg = frozen_config()
    return dict(cfg, experiment="Exp5A-Control", representation=representation,
                mode=mode, epochs=1 if mode == "smoke" else cfg["epochs"],
                dropout_layout="independent_per_label_matching_frozen_exp5a",
                head_initialization="skip_query_draw_then_normal_0.02_matching_exp5a")


def freeze():
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)


def checked_artifacts(root, success_name="SUCCESS.json"):
    root = Path(root)
    proof = read(root / success_name)
    expected = proof["artifacts_sha256"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != success_name}
    require(actual == set(expected), f"Artifact inventory differs: {root}")
    for name, digest in expected.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
        require(sha(root / relative) == digest, f"Artifact changed: {name}")
    return proof


def seal(root, status, **extra):
    root = Path(root)
    require(not (root / "SUCCESS.json").exists(), "Refusing to overwrite success proof")
    write(root / "SUCCESS.json", dict(status=status, completed_at=now(), seed=42,
          source_sha256=check_sources(), artifacts_sha256={str(p.relative_to(root)): sha(p)
          for p in sorted(root.rglob("*")) if p.is_file()}, **extra))


def verify_c():
    from verify import verify_run
    check_sources()
    verify_run(C_RUN)
    proof = checked_artifacts(C_RUN)
    cfg = read(C_RUN / "config.json")
    require(cfg == dict(frozen_config(), mode="full"), "C protocol differs from frozen config")
    env = read(C_RUN / "environment.json")
    require(env["hostname"] == "D3020-A3" and "4090" in env["gpu"], "Wrong C host/GPU")
    require(env["packages"] == package_versions(), "Current environment differs from C")
    require((C_RUN / "pip-freeze.txt").read_text() == freeze(), "Pip environment differs from C")
    require(env["completed_optimizer_steps"] == 10856, "C training incomplete")
    return {"path": str(C_RUN), "success_sha256": sha(C_RUN / "SUCCESS.json"),
            "artifacts_sha256": proof["artifacts_sha256"], "parameters": env["parameters_runtime"]}


def verify_prediction_metrics(root, split, records):
    """Recompute reported metrics on CPU from saved logits; never rerun GPU/test tuning."""
    import numpy as np
    import pandas as pd
    from metrics import score, probabilities, rare_scores
    from data import multi_hot
    root = Path(root)
    with np.load(root / f"{split}_predictions.npz", allow_pickle=False) as npz:
        logits, gold, ids = npz["logits"], npz["targets"], npz["ids"]
    expected_gold = np.stack([multi_hot(r["labels"]) for r in records])
    require(np.array_equal(ids, np.asarray([r["id"] for r in records])), "Prediction IDs/order differ")
    require(np.array_equal(gold, expected_gold), "Prediction gold labels differ")
    measured, frame = score(gold, probabilities(logits), 0.30)
    recorded = read(root / f"{split}_metrics.json")["fixed_0.30"]
    require(set(measured) == set(recorded), "Metric keys differ")
    for key, value in measured.items():
        require(abs(value - recorded[key]) < 1e-12, f"Metric recomputation differs: {key}")
    require(read(root / f"{split}_fixed_0.30_metrics.json") == recorded, "Metric copies disagree")
    actual_frame = pd.read_csv(root / f"{split}_fixed_0.30_per_label.csv").sort_values("label_id").reset_index(drop=True)
    expected_frame = frame.sort_values("label_id").reset_index(drop=True)
    require(actual_frame[["label_id", "label", "support"]].equals(expected_frame[["label_id", "label", "support"]]), "Label support/order differs")
    np.testing.assert_allclose(actual_frame[["precision", "recall", "f1"]], expected_frame[["precision", "recall", "f1"]], rtol=0, atol=1e-12)
    rare = rare_scores(frame)
    report = read(root / "rare_label_metrics.json")
    require(report["definition"]["label_ids"] == RARE_IDS, "Rare IDs differ")
    for key in ("mean_rare_label_f1", "mean_non_rare_label_f1"):
        require(abs(report[split]["fixed_0.30"][key] - rare[key]) < 1e-12, "Rare score differs")
    return measured, rare, expected_frame
