"""Portable paths, strict JSON, source provenance and runtime checks."""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    files = [p for folder in ("src", "scripts", "hpc", "slurm", "config")
             for p in (ROOT / folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".json")]
    files.extend(ROOT.glob("requirements*.txt"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_config(path=None):
    config = read_json(path or ROOT / "config/baseline_bert.json")
    if config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] != 16:
        raise ValueError("Effective batch size must remain 16 on one GPU.")
    if config["num_labels"] != 28 or config["fixed_threshold"] != 0.30:
        raise ValueError("Unexpected label count or fixed threshold.")
    return config


def require_slurm_gpu():
    import torch
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training is allowed only inside a Linux Slurm GPU allocation.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU; check Slurm allocation.")


def package_versions():
    return {name: version(name) for name in (
        "torch", "transformers", "datasets", "accelerate", "numpy", "pandas",
        "scikit-learn", "matplotlib", "pyarrow", "huggingface-hub", "tokenizers", "safetensors")}
