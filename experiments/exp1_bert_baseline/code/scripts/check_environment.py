"""Fail early on wrong interpreter, dependencies or GPU allocation."""
import argparse
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

from src.common import ROOT, capture, package_versions, require_slurm_gpu, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    require_slurm_gpu()
    expected_python = Path(os.environ["CONDA_ENV"]) / "bin/python"
    if Path(sys.executable).resolve() != expected_python.resolve() or sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Wrong interpreter: {sys.executable}, expected Python 3.10 at {expected_python}")
    requirements = ["torch==2.6.0+cu118"] + (ROOT / "requirements-nlp.txt").read_text().splitlines()
    for line in requirements:
        if "==" in line and not line.startswith("#"):
            name, expected = line.strip().split("==")
            if version(name) != expected:
                raise RuntimeError(f"Version mismatch {name}: {version(name)} != {expected}")
    # Import the whole project closure, not just torch.
    from src import data, model, metrics, train, evaluate, visualize  # noqa: F401
    from transformers.models.bert.modeling_bert import BertForSequenceClassification  # noqa: F401
    if torch.version.cuda != "11.8":
        raise RuntimeError(f"Expected cu118, got {torch.version.cuda}")
    info = {"python": platform.python_version(), "executable": sys.executable, "packages": package_versions(),
            "torch_cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0), "visible_gpu_count": torch.cuda.device_count(),
            "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "nvidia_smi": capture(["nvidia-smi"])}
    print(info, flush=True)
    if args.save:
        write_json(args.save, info)


if __name__ == "__main__":
    main()
