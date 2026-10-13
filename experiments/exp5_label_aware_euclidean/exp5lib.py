"""Strict paths, provenance, atomic JSON and experiment invariants for Exp5A."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "config.json"
PACKAGE_MANIFEST = HERE / "PACKAGE_MANIFEST.sha256"
DATA_DIR = REPO / "数据集/GoEmotions/processed"
ASSET_MANIFEST = REPO / "experiments/exp2_lgsa/LGSA-GoEmotions/assets/goemotions-v0/manifest.json"
MODEL_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
MODEL_DIR = REPO / "assets/pretrained/bert-base-cased" / MODEL_REVISION
RESULTS = REPO / "results/exp5a"
EXPECTED_HOST = "D3020-A3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def objsha(value) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_hashes():
    suffixes = {".py", ".json", ".md", ".sh"}
    return {
        str(path.relative_to(HERE)): sha(path)
        for path in sorted(HERE.rglob("*"))
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
    }


def verify_package_manifest():
    require(PACKAGE_MANIFEST.is_file(), "PACKAGE_MANIFEST.sha256 is missing")
    expected = {}
    for raw_line in PACKAGE_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = raw_line.split("  ", 1)
        require("/" not in name and name != PACKAGE_MANIFEST.name, f"Unsafe package manifest entry: {name}")
        expected[name] = digest
    actual = {
        path.name: sha(path)
        for path in sorted(HERE.iterdir())
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".sh"}
    }
    require(expected == actual, "Package files differ from PACKAGE_MANIFEST.sha256")
    return expected


def load_config():
    config = read(CONFIG)
    require(config["num_labels"] == 28, "Exp5A requires exactly 28 labels")
    require(config["seed"] == 42 and config["authorized_seeds"] == [42], "Only seed42 is authorized")
    require(config["epochs"] == 4, "Formal protocol requires four epochs")
    require(config["effective_batch_size"] == 16, "Effective batch must remain 16")
    require(
        config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] == 16,
        "Physical batch times accumulation must equal 16",
    )
    require(config["fixed_threshold"] == 0.30, "Formal threshold must remain 0.30")
    require(config["max_length"] == 50, "Text max length must remain 50")
    require(config["learning_rate"] == 5e-5, "Learning rate must match Exp1")
    return config


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions():
    names = (
        "torch", "transformers", "datasets", "numpy", "pandas", "scikit-learn",
        "safetensors", "tokenizers", "huggingface-hub",
    )
    return {name: version(name) for name in names}


def require_runtime(require_cuda=True):
    import torch

    require(platform.system() == "Linux", "Exp5A GPU phases run only on Linux")
    require(platform.node() == EXPECTED_HOST, f"Expected host {EXPECTED_HOST}, got {platform.node()}")
    if require_cuda:
        require(torch.cuda.is_available(), "CUDA is not available")
        require(torch.cuda.device_count() == 1, "Expected exactly one visible CUDA GPU")
        require("4090" in torch.cuda.get_device_name(0), "Expected the Ubuntu RTX 4090")


def exp1_reference_dir():
    candidates = [
        REPO / "experiments/exp2_lgsa/LGSA-GoEmotions/goemotions_adaptation/baseline_reference",
        REPO / "experiments/exp2_lgsa/code/goemotions_adaptation/baseline_reference",
    ]
    found = [path for path in candidates if (path / "reference_manifest.json").is_file()]
    require(found, "Verified Exp1 baseline_reference was not found")
    if len(found) == 2:
        names = ("reference_manifest.json", "test_metrics.json", "rare_label_metrics.json", "test_per_label_fixed_0.30.csv")
        require(
            all(sha(found[0] / name) == sha(found[1] / name) for name in names),
            "Two Exp1 reference copies disagree",
        )
    return found[0]
