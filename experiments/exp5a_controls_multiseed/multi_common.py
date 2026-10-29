"""Immutable seed42 dependencies; authorize only CLS/mean seeds123/2026."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTROL_SOURCE = REPO / "experiments/exp5a_controls"
SEED42_ROOT = REPO / "results/exp5a_controls/run_20261013T103802Z"
sys.path.insert(0, str(CONTROL_SOURCE))
import control_common as frozen
from control_common import (ASSET_MANIFEST, MODEL_DIR, LABEL_NAMES, RARE_IDS, read, write, sha,
                            require, now, freeze, package_versions, require_runtime,
                            checked_artifacts, load_all_verified, verify_prediction_metrics)

AUTHORIZED = (123, 2026)
ALL_SEEDS = (42, 123, 2026)


def sources():
    return {p.name: sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in {".py", ".json", ".md", ".sh"}}


def check_sources():
    old = frozen.check_sources()
    pinned = read(HERE / "frozen_dependency.json")
    require(old == pinned["source_sha256"], "Frozen seed42/Query sources changed")
    require(sha(CONTROL_SOURCE / "PACKAGE_MANIFEST.sha256") == pinned["control_manifest_sha256"], "Frozen control manifest changed")
    entries = {}
    for line in (HERE / "PACKAGE_MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(Path(name).name == name, "Unsafe package entry")
        entries[name] = digest
    require(entries == sources(), "Multi-seed package changed")
    return dict(multiseed=sources(), frozen=old)


def config_for(mode, representation, seed):
    require(type(seed) is int and seed in AUTHORIZED, "Only NEW seeds123/2026 are authorized")
    cfg = frozen.config_for(mode, representation)
    return dict(cfg, experiment="Exp5A-Control-MultiSeed", seed=seed,
                authorized_seeds=list(AUTHORIZED), aggregation_seeds=list(ALL_SEEDS))


def model_directory(root, mode, representation, seed):
    config_for(mode, representation, seed)
    name = f"smoke_{representation}_seed{seed}" if mode == "smoke" else f"{representation}_seed{seed}"
    return Path(root) / name


def seal(root, status, seed, **extra):
    root = Path(root)
    require(seed in AUTHORIZED and not (root / "SUCCESS.json").exists(), "Invalid/duplicate success proof")
    write(root / "SUCCESS.json", dict(status=status, seed=seed, completed_at=now(),
          source_sha256=check_sources(), artifacts_sha256={str(p.relative_to(root)): sha(p)
          for p in sorted(root.rglob("*")) if p.is_file()}, **extra))


def verify_seed42():
    check_sources()
    # This is a read-only CPU recomputation from saved logits, never a GPU forward.
    from control_compare import compare
    compare(SEED42_ROOT, verify_only=True)
    return dict(path=str(SEED42_ROOT), comparison_success_sha256=sha(SEED42_ROOT / "CONTROL_SUCCESS.json"),
                comparison_sha256=sha(SEED42_ROOT / "CONTROL_COMPARISON.json"),
                cls_success_sha256=sha(SEED42_ROOT / "cls_seed42/SUCCESS.json"),
                mean_success_sha256=sha(SEED42_ROOT / "mean_seed42/SUCCESS.json"),
                c_success_sha256=sha(frozen.C_RUN / "SUCCESS.json"))
