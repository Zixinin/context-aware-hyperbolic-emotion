"""Stdlib-only host/environment/assets inspection. Never installs or loads a model."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import argparse
from pathlib import Path

from common import REPO, check_package, check_reference, find_snapshots, require


def dataset_audit():
    # This branch runs only in a discovered existing environment with NumPy.
    from data27 import load_dataset
    data, manifest = load_dataset()
    print("CANONICAL_DATASET_VERIFIED", json.dumps({s: {
        k: v for k, v in manifest['splits'][s].items() if k in {
            'original_count','retained_count','removed_neutral_only','single_emotion','multi_emotion',
            'multi_fraction','retained_neutral_plus_emotion'}} for s in data}, sort_keys=True))
    print("CANONICAL_LABEL_MAPPING", json.dumps(manifest["mapping"]))
    print("EXACT_TEXT_OVERLAP", json.dumps(manifest["text_overlap_audit"]))


def main():
    require(platform.node() == "D3020-A3" and platform.system() == "Linux", "Expected Ubuntu D3020-A3")
    require(REPO == Path("/home/zixinfeng/Documents/context-aware-hyperbolic-emotion"), "Wrong repository")
    print("SOURCE_PACKAGE_VERIFIED", len(check_package()))
    print("HISTORICAL_PROTOCOL_FILES_VERIFIED", len(check_reference()))
    subprocess.run(["git", "-C", str(REPO), "status", "--short"], check=True)
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["df", "-h", str(REPO)], check=True)
    # User must return any applicable remote-only instructions before GPU work.
    instructions = []
    for parent in [Path.home(), Path.home() / "Documents", REPO, REPO / "experiments", Path(__file__).parent]:
        p = parent / "AGENTS.md"
        if p.is_file():
            print("REMOTE_INSTRUCTIONS", p)
            print(p.read_text())
            instructions.append(str(p))
    candidates = [REPO / ".venv-bert/bin/python",
                  Path("/home/zixinfeng/Documents/HypEmo-Reproduction/.venv-hypemo-torch1131-cu116/bin/python")]
    query = """
import json,sys,importlib.metadata as m,importlib.util
names=['torch','transformers','numpy','scikit-learn','tokenizers','safetensors']
packages={}
for p in names:
 try: packages[p]=m.version(p)
 except m.PackageNotFoundError: packages[p]=None
print(json.dumps({'executable':sys.executable,'python':sys.version,'packages':packages}))
"""
    dataset_python = None
    for p in candidates:
        if p.is_file():
            r = subprocess.run([str(p), "-c", query], text=True, capture_output=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1"))
            print("ENVIRONMENT_CANDIDATE", p, "exit", r.returncode, r.stdout.strip(), r.stderr.strip())
            if r.returncode == 0:
                details = json.loads(r.stdout)
                if details["packages"].get("numpy") and dataset_python is None:
                    dataset_python = p
    snapshots = find_snapshots()
    for p in snapshots:
        print("LOCAL_ROBERTA_SNAPSHOT", p)
    require(dataset_python is not None,
            "No existing candidate environment with NumPy; return inspection output, do not install into base")
    print("DATASET_AUDIT_PYTHON", dataset_python, flush=True)
    subprocess.run([str(dataset_python), str(Path(__file__).resolve()), "--dataset-only"],
                   check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1"))
    print("DEFAULT_REUSE_ENV", REPO / ".venv-bert/bin/python")
    if not snapshots:
        print("BLOCKED: no complete local immutable RoBERTa snapshot; do not install/download/change environments yet.")
    if len(snapshots) > 1:
        print("ACTION: specify the audited snapshot using --snapshot when launching smoke.")
    if instructions:
        print("ACTION: return remote AGENTS.md above for review before smoke.")
    print("EXP6C0_INSPECTION_COMPLETE_NO_TRAINING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard-library environment inspection; dataset audit in an existing NumPy environment")
    parser.add_argument("--dataset-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dataset_only:
        dataset_audit()
    else:
        main()
