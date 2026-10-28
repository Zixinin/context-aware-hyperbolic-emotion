"""Exp6C-0 provenance and immutable-input gates; no third-party HypEmo imports."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results/exp6c0_roberta27_baseline"
RARE = ["grief", "pride", "relief", "nervousness", "embarrassment", "remorse", "fear"]
CONFIG = dict(experiment="Exp6C-0 A0", model_name="roberta-base", num_labels=27,
              seed=42, epochs=4, max_length=50, batch_size=16, eval_batch_size=64,
              gradient_accumulation_steps=1, learning_rate=5e-5, weight_decay=0.01,
              adam_betas=[0.9, 0.999], adam_epsilon=1e-8, warmup_ratio=0.10,
              warmup_rounding="ceil", scheduler="linear", max_grad_norm=1.0,
              dropout=0.1, initializer_std=0.02, threshold=0.30,
              pooling="last_hidden_state[:,0,:]", loss="BCEWithLogitsLoss(mean)",
              dropout_layout="independent_per_label_matching_Exp5A_CLS",
              head_initialization="discard_one_normal_draw_then_normal_0.02_matching_CLS",
              precision="float32", attention_implementation="eager", num_workers=0,
              deterministic_algorithms=True, rare_labels=RARE,
              checkpoint_rule="highest_validation_macro_f1_at_0.30_earliest_tie",
              filtering="keep_any_non_neutral_positive_preserve_official_split_order",
              neutral_diagnostic="skipped_optional_diagnostic_to_keep_primary_test_single_pass")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def capture(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def sources():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()
            and p.name != "PACKAGE_MANIFEST.sha256"}


def check_package():
    expected = {}
    for line in (HERE / "PACKAGE_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(Path(name).name == name, "Unsafe package manifest")
        expected[name] = digest
    require(expected == sources(), "Experiment package changed; stop and inspect")
    return expected


def check_reference():
    expected = read(HERE / "protocol_reference.json")
    for name, digest in expected.items():
        require(sha(REPO / name) == digest, f"Historical protocol source differs: {name}")
    return expected


def runtime():
    import torch
    require(platform.system() == "Linux" and platform.node() == "D3020-A3", "GPU work requires Ubuntu D3020-A3")
    require(REPO == Path("/home/zixinfeng/Documents/context-aware-hyperbolic-emotion"), "Wrong Ubuntu repository")
    require(os.environ.get("PYTHONHASHSEED") == "42", "Use launch.sh: PYTHONHASHSEED must be set before Python")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Missing deterministic CUBLAS setting")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Need exactly one visible CUDA GPU")
    require("4090" in torch.cuda.get_device_name(0), "Expected RTX 4090")
    return dict(host=platform.node(), python=sys.version, executable=sys.executable,
                packages={p: version(p) for p in ["torch", "transformers", "numpy", "scikit-learn", "tokenizers", "safetensors"]},
                pip_freeze=capture([sys.executable, "-m", "pip", "freeze"]),
                gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                nvidia_smi=capture(["nvidia-smi"]), git_commit=capture(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
                git_status=capture(["git", "-C", str(REPO), "status", "--short"]),
                env={k: os.environ.get(k) for k in ["PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE"]})


def protected_inventory():
    """Hash existing research inputs/results, including untracked files, read-only."""
    paths = set()
    roots = [REPO / x for x in ["experiments", "results", "outputs", "数据集", "assets/pretrained", "docs"]]
    external = Path("/home/zixinfeng/Documents")
    roots += [external / "HypEmo-Reproduction" / x for x in ["outputs", "reports", "HypEmo"]]
    roots += [external / "HypEmo-Exp6B" / x for x in ["stage_a", "stage_b", "reports"]]
    for root in roots:
        if not root.exists():
            continue
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}
                       and not d.startswith(".venv")
                       and Path(base) / d not in {HERE, RESULTS}]
            for name in files:
                if name.endswith((".pyc", ".DS_Store")):
                    continue
                p = Path(base) / name
                if p.is_file():
                    paths.add(p)
    return {str(p): sha(p) for p in sorted(paths)}


def snapshot_inventory(snapshot):
    return {p.name: sha(p) for p in sorted(Path(snapshot).iterdir()) if p.is_file()}


def find_snapshots():
    hubs = [Path.home() / ".cache/huggingface/hub",
            Path("/home/zixinfeng/Documents/HypEmo-Reproduction/environment/huggingface/hub"),
            REPO / "assets/pretrained/roberta-base"]
    found = set()
    for hub in hubs:
        if not hub.exists():
            continue
        candidates = list(hub.glob("models--*roberta-base/snapshots/*"))
        if hub.name == "roberta-base":
            candidates += list(hub.iterdir())
        for p in candidates:
            if p.is_dir() and (p / "config.json").is_file():
                cfg = read(p / "config.json")
                if (cfg.get("model_type") == "roberta" and cfg.get("hidden_size") == 768
                        and cfg.get("num_hidden_layers") == 12
                        and ((p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file())
                        and (p / "vocab.json").is_file() and (p / "merges.txt").is_file()):
                    found.add(str(p.absolute()))
    return sorted(found)


def run_path(root):
    root = Path(root).resolve()
    require(root.parent == RESULTS.resolve() and root.name.startswith("run_"), "Wrong result directory")
    return root


def seal(root, filename, payload):
    require(not (root / filename).exists(), f"Refusing to replace {filename}")
    hashes = artifact_hashes(root, filename)
    write(root / filename, dict(payload, artifacts_sha256=hashes))


def artifact_hashes(root, filename):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
            and p.name != filename and not p.name.endswith((".log", "_exit_code.txt"))
            and "runtime_cache" not in p.relative_to(root).parts}


def check_seal(root, filename):
    proof = read(root / filename)
    require(artifact_hashes(root, filename) == proof["artifacts_sha256"], "Artifact inventory/content differs")
    for name, digest in proof["artifacts_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe seal path")
        require(sha(root / name) == digest, f"Artifact modified: {name}")
    return proof
