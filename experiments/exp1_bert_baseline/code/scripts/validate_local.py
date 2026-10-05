"""Delivery checks only. No BERT instance, weights, forward pass or training."""
import ast
import importlib
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from src.common import ROOT, package_versions, source_hashes, utc_now, write_json


def main():
    if platform.system() != "Darwin":
        raise RuntimeError("This delivery checker is Mac-only; do not invoke it on HPC.")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fyp-baseline-matplotlib"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).glob("*.py"):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))
    for folder, suffix in (("hpc", "*.sh"), ("slurm", "*.sbatch")):
        for path in (ROOT / folder).glob(suffix):
            subprocess.run(["bash", "-n", str(path)], check=True)
    modules = ["src.data", "src.model", "src.metrics", "src.train", "src.evaluate", "src.visualize",
               "scripts.prepare_assets", "scripts.analyze_dataset", "scripts.check_environment", "scripts.verify_run",
               "scripts.summarize_results", "transformers.models.bert.modeling_bert"]
    for name in modules:
        importlib.import_module(name)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    from src.data import load_official_tsv, validate_dataset
    dataset = load_official_tsv(ROOT / "data/official")
    validate_dataset(dataset)
    guard = subprocess.run([sys.executable, "-m", "src.train", "--mode", "smoke"], cwd=ROOT, capture_output=True, text=True)
    if platform.system() == "Darwin":
        if guard.returncode == 0 or "Training is allowed only inside a Linux Slurm GPU allocation" not in guard.stderr:
            raise RuntimeError("Mac training guard did not reject execution.")
    report = {"checked_at": utc_now(), "status": "PASS", "scope": "static checks, import closure, synthetic CPU logic and official-file integrity ONLY",
              "platform": platform.platform(), "python": platform.python_version(), "packages": package_versions(),
              "checks": ["Python 3.10 grammar", "all shell/Slurm syntax", "complete imports including actual BERT class",
                         "10 synthetic CPU unit tests", "official TSV SHA-256 / labels / sizes / disjoint IDs", "Mac training rejected"],
              "official_split_sizes": {s: len(dataset[s]) for s in dataset}, "source_sha256": source_hashes(),
              "not_executed": ["HPC Python 3.10/cu118 runtime", "model download", "BERT forward/backward", "Slurm smoke", "full baseline"],
              "limitation": "Temporary venv pins NLP packages and reuses existing local CPU PyTorch 2.6.0; this is not a Linux GPU environment test."}
    write_json(ROOT / "environment/local_validation.json", report)
    print("LOCAL DELIVERY CHECKS PASS (not an HPC experiment)")


if __name__ == "__main__":
    main()
