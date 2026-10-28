"""Independent A1 provenance; frozen A0 is imported read-only, never patched."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A0_SOURCE = REPO / "experiments/exp6c0_roberta27_baseline"
sys.path.insert(0, str(A0_SOURCE))
import common as a0
from common import read, write, sha, require, now, object_sha

A0_RUN = REPO / "results/exp6c0_roberta27_baseline/run_20261020T033547Z_vQWCGEh8"
RESULTS = REPO / "results/exp6c1_roberta27_hypaux"
UPSTREAM = Path("/home/zixinfeng/Documents/HypEmo-Reproduction/HypEmo")
EXP6B = Path("/home/zixinfeng/Documents/HypEmo-Exp6B/stage_b/smoke_vYTTGsIO")
CONFIG = dict(a0.CONFIG, experiment="Exp6C-1 A1", loss="mean_BCE_plus_0.1_FormulaA_aux",
              auxiliary_weight=0.1, auxiliary_formula="A_positive_mean_batch_mean_v1_aux_only",
              hyperbolic_dim=100, curvature=1.0, auxiliary_dropout=0.1,
              hyperbolic_initialization="fork_torch_RNG_after_shared_A0_constructor",
              auxiliary_input="same_encoder_last_hidden_first_token_before_classifier_dropout",
              classification_distance_weighting=False)


def check_sources():
    expected = read(HERE / "PACKAGE_MANIFEST.json")
    actual = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "PACKAGE_MANIFEST.json"}
    require(actual == expected, "A1 source/package manifest changed")
    reference = read(HERE / "A0_SOURCE_REFERENCE.json")
    require({p.name: sha(p) for p in A0_SOURCE.iterdir() if p.is_file()} == reference, "Frozen A0 code changed")
    a0.check_package()
    return dict(a1=actual, a0=reference)


def run_path(root):
    root = Path(root).resolve()
    require(root.parent == RESULTS.resolve() and root.name.startswith("run_"), "Wrong A1 run directory")
    return root


def artifact_hashes(root, seal_name):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
            and p.name != seal_name and not p.name.endswith((".log", "_exit_code.txt"))
            and "runtime_cache" not in p.relative_to(root).parts}


def seal(root, name, **payload):
    require(not (root / name).exists(), "Refusing to replace a success proof")
    write(root / name, dict(payload, artifacts_sha256=artifact_hashes(root, name)))


def check_seal(root, name):
    proof = read(root / name)
    require(proof["artifacts_sha256"] == artifact_hashes(root, name), f"Sealed artifacts differ: {root}")
    return proof


def protected_inventory():
    # The old A0 global inventory predates A1. Do NOT rewrite its historical
    # manifest or run its global-additions check. Protect A0 explicitly here.
    roots = [REPO / n for n in ["experiments", "results", "outputs", "数据集", "assets/pretrained", "docs"]]
    roots += [UPSTREAM]
    roots += [UPSTREAM.parent / n for n in ["outputs", "reports"]]
    roots += [EXP6B.parents[1] / n for n in ["stage_a", "stage_b", "reports"]]
    files = set()
    for root in roots:
        if not root.exists():
            continue
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}
                       and not d.startswith(".venv") and Path(base) / d not in {HERE, RESULTS}]
            for name in names:
                p = Path(base) / name
                if p.is_file() and not name.endswith((".pyc", ".DS_Store")):
                    files.add(p)
    return {str(p): sha(p) for p in sorted(files)}


def baseline_evidence():
    """Verify A0's own exact seal and saved predictions, allowing authorized A1 additions."""
    import csv
    import numpy as np
    from data27 import load_dataset, targets
    from metrics27 import score, probabilities
    check_sources()
    proof = a0.check_seal(A0_RUN, "SUCCESS.json")
    require(proof["status"] == "EXP6C0_ROBERTA27_MULTILABEL_BASELINE_VERIFIED", "Wrong A0 status")
    require((A0_RUN / "train_exit_code.txt").read_text().strip() == "0", "A0 training failed")
    require(read(A0_RUN / "config.json") == a0.CONFIG, "A0 protocol differs")
    env = read(A0_RUN / "environment.json")
    require(env["status"] == "COMPLETE" and env["completed_epochs"] == 4 and
            env["completed_optimizer_steps"] == 7648 and env["formal_test_inference_calls"] == 1, "A0 incomplete")
    data, manifest = load_dataset()
    require(manifest == read(A0_RUN / "dataset_manifest.json"), "A0 dataset identity differs")
    require([len(data[s]) for s in ["train", "validation", "test"]] == [30587, 3834, 3821], "Wrong canonical sample counts")
    for split in ["validation", "test"]:
        with np.load(A0_RUN / f"{split}_predictions.npz", allow_pickle=False) as z:
            require(np.array_equal(z["ids"], [r["id"] for r in data[split]]), "A0 IDs mismatch")
            require(np.array_equal(z["targets"], targets(data[split])), "A0 gold mismatch")
            require(np.array_equal(z["label_names"], manifest["names27"]), "A0 label order mismatch")
            values, rows, rare, pred = score(z["targets"], probabilities(z["logits"]), manifest["names27"])
        require(values == read(A0_RUN / f"{split}_metrics.json"), "A0 saved metric recomputation differs")
        with (A0_RUN / f"{split}_per_label_metrics.csv").open() as f:
            recorded = list(csv.DictReader(f))
        require(len(recorded) == 27, "A0 per-label metric count differs")
        for a, b in zip(recorded, rows):
            require(set(a) == set(b), "A0 per-label schema differs")
            for k, v in b.items():
                require(a[k] == v if isinstance(v, str) else abs(float(a[k]) - v) < 1e-12, "A0 per-label metrics differ")
        require(rare == read(A0_RUN / "rare_label_metrics.json")[split], "A0 rare metrics differ")
    reload = read(A0_RUN / "checkpoint_reload_verification.json")
    require(reload["num_examples"] == 3834 and reload["metrics_equal"] and reload["threshold_predictions_equal"], "A0 reload proof missing")
    selection = read(A0_RUN / "checkpoint_selection.json")
    require(sha(A0_RUN / "best_checkpoint/model.pt") == selection["checkpoint_sha256"], "A0 checkpoint hash differs")
    return data, manifest, dict(run=str(A0_RUN), success_sha256=sha(A0_RUN / "SUCCESS.json"),
        config_sha256=sha(A0_RUN / "config.json"), selected_checkpoint=selection,
        test_metrics=read(A0_RUN / "test_metrics.json"),
        smoke_weights_sha256=sha(A0_RUN / "smoke/smoke_weights.pt"),
        source_hashes=read(HERE / "A0_SOURCE_REFERENCE.json"))


def runtime():
    current = a0.runtime()
    historical = read(A0_RUN / "plan.json")["runtime"]
    for k in ["host", "python", "executable", "packages", "pip_freeze", "gpu", "cuda_runtime"]:
        require(current[k] == historical[k], f"A1 runtime differs from A0: {k}")
    return current


def verify_upstream():
    expected = read(HERE / "UPSTREAM_REFERENCE.json")
    require(a0.capture(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]) == "97ec6af6b19b831f69cb35a82dc65ea1395fa157", "Wrong HypEmo commit")
    tracked = a0.capture(["git", "-C", str(UPSTREAM), "ls-files", "-z"]).rstrip("\0").split("\0")
    require(set(tracked) == set(expected), "Upstream tracked inventory differs")
    for name, digest in expected.items():
        require(sha(UPSTREAM / name) == digest, f"Upstream input changed: {name}")
    reference = read(HERE / "EXP6B_REFERENCE.json")
    require(sha(EXP6B / "SMOKE_SUCCESS.json") == reference["smoke_success_sha256"], "Exp6B successful smoke proof changed")
    proof = read(EXP6B / "SMOKE_SUCCESS.json")
    require(proof["status"] == "EXP6B_STAGE_B_BERT_HYPEMO_SMOKE_VERIFIED", "Exp6B smoke not verified")
    for name in ["original_emotion_embeddings.npy", "upstream_inventory.json", "label_mapping.json", "tools/approved_multilabel_loss.py"]:
        require(sha(EXP6B / name) == proof["artifacts_sha256"][name], f"Frozen Exp6B artifact changed: {name}")
    require(sha(EXP6B / "original_emotion_embeddings.npy") == reference["original_emotion_embeddings_sha256"], "Fixed label vectors changed")
    require(sha(EXP6B / "tools/approved_multilabel_loss.py") == reference["approved_loss_source_sha256"], "Formula A reference changed")
    require(read(EXP6B / "upstream_inventory.json") == expected, "Exp6B/upstream source identities differ")
    return reference


def load_reference_loss():
    path = EXP6B / "tools/approved_multilabel_loss.py"
    require(sha(path) == read(HERE / "EXP6B_REFERENCE.json")["approved_loss_source_sha256"], "Reference loss changed")
    spec = importlib.util.spec_from_file_location("exp6b_formula_a_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.loss_components


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode() + str(tuple(value.shape)).encode() + str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def build_hyperbolic(names, device="cuda:0"):
    """Import verified original HNN/decoder in place; never copy upstream into FYP."""
    import numpy as np
    import torch
    verify_upstream()
    sys.path.append(str(UPSTREAM))
    old_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        import config as upstream_config
        from models.base_models import FGTCModel
    finally:
        os.chdir(old_cwd)
    require(Path(upstream_config.__file__).resolve() == UPSTREAM / "config.py", "Wrong upstream config module")
    for name in ["models.base_models", "models.encoders", "models.decoders", "layers.hyp_layers", "manifolds.poincare"]:
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(UPSTREAM), f"Wrong imported module {name}")
    upstream_names = [n for _, n in sorted(upstream_config.label_dicts[1].items())]
    require(set(upstream_names) == set(names) and len(names) == 27, "Upstream/frozen 27 labels differ")
    source = np.stack([np.asarray(upstream_config.emb_dicts[1][i], dtype=np.float32) for i in range(27)])
    require(np.array_equal(source, np.load(EXP6B / "original_emotion_embeddings.npy", allow_pickle=False)), "Label vectors differ from validated Exp6B")
    aligned = np.stack([source[upstream_names.index(n)] for n in names])
    require(aligned.shape == (27, 100) and np.isfinite(aligned).all() and (np.linalg.norm(aligned, axis=1) < 1).all(), "Invalid fixed ball vectors")
    args = upstream_config.parser.parse_args([])
    args.device = torch.device(device)
    args.feat_dim, args.n_classes, args.n_samples = 768, 27, 30587
    require((args.dim, args.manifold, args.c, args.dropout, args.num_layers, args.act, args.bias, args.pos_weight)
            == (100, "PoincareBall", 1.0, 0.1, 2, "relu", 1, 0), "Original HNN profile differs")
    # Caller surrounds this constructor and its imports with fork_rng.
    model = FGTCModel(args)
    require(sum(p.numel() for p in model.parameters()) == 79627, "Unexpected HNN/decoder parameters")
    require(model.encoder.encode_graph is False, "Only text HNN is authorized, not a graph encoder")
    return model, torch.from_numpy(aligned.copy()), dict(
        upstream=str(UPSTREAM), source_commit="97ec6af6b19b831f69cb35a82dc65ea1395fa157",
        source_vector_sha256=sha(UPSTREAM / "label_tree/go_emotion.bin"),
        upstream_names=upstream_names, aligned_names=names,
        original_vector_npy_sha256=sha(EXP6B / "original_emotion_embeddings.npy"),
        hyperbolic_parameters=79627, curvature=1.0, dropout=0.1, dim=100, num_layers=2,
        source_copied_into_fyp=False)
