"""Three paired seeds: mean±sample-SD, raw paired deltas, all labels, tiny-label share."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multi_common import (ALL_SEEDS, RARE_IDS, SEED42_ROOT, check_sources, load_all_verified,
                          now, read, require, sha, verify_seed42, verify_prediction_metrics, write)
from multi_verify import verify_control
from control_compare import contribution
from control_verify import verify_control as verify_seed42_model

METRICS = [("Micro-F1", "micro_f1"), ("Macro-F1", "macro_f1"),
           ("Macro Precision", "macro_precision"), ("Macro Recall", "macro_recall"),
           ("Weighted-F1", "weighted_f1"), ("Rare-F1", "rare_f1")]


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (3,) and np.isfinite(array).all(), "Expected exactly three finite paired seeds")
    return dict(mean=float(array.mean()), sample_sd=float(array.std(ddof=1)),
                positive_seeds=int((array > 1e-12).sum()), negative_seeds=int((array < -1e-12).sum()),
                zero_seeds=int((np.abs(array) <= 1e-12).sum()))


def build(root):
    root = Path(root)
    reference = verify_seed42()
    datasets, _ = load_all_verified()
    rows, frames, diagnostics, contributions = [], {}, [], {}
    provenance = {}
    for seed in ALL_SEEDS:
        roots = {m: (SEED42_ROOT if seed == 42 else root) / f"{m}_seed{seed}" for m in ("cls", "mean")}
        for model, directory in roots.items():
            if seed == 42:
                verify_seed42_model(directory)
            else:
                verify_control(directory)
                env = read(directory / "environment.json")
                require(env["seed42_reference"] == reference, "Frozen seed42 provenance drift")
            cfg = read(directory / "config.json")
            require(cfg["seed"] == seed and cfg["representation"] == model and cfg["mode"] == "full", "Paired run identity differs")
            metrics, rare, frames[seed, model] = verify_prediction_metrics(directory, "test", datasets["test"])
            val, _, _ = verify_prediction_metrics(directory, "validation", datasets["validation"])
            rows.append(dict(seed=seed, model=model, **metrics, rare_f1=rare["mean_rare_label_f1"]))
            env = read(directory / "environment.json")
            selection = read(directory / "checkpoint_selection.json")
            diagnostics.append(dict(seed=seed, model=model, best_epoch=selection["epoch"],
                                    validation_macro_f1=val["macro_f1"], validation_micro_f1=val["micro_f1"],
                                    trainable_parameters=env["parameters_runtime"]["trainable_parameters"]))
            provenance[f"{model}_seed{seed}"] = dict(path=str(directory), success_sha256=sha(directory / "SUCCESS.json"))
        require(sha(roots["cls"] / "sample_ids.json") == sha(roots["mean"] / "sample_ids.json"), "Paired training/validation records differ")
        a, b = frames[seed, "cls"], frames[seed, "mean"]
        require(a[["label_id", "label", "support"]].equals(b[["label_id", "label", "support"]]), "Paired label supports differ")
        local = a[["label_id", "label", "support", "f1"]].rename(columns={"f1": "cls_f1"})
        local["mean_f1"] = b.f1
        contributions[str(seed)] = contribution(local, "cls_f1", "mean_f1")
    results = pd.DataFrame(rows)
    paired = []
    aggregates = {}
    for label, key in METRICS:
        a = results.loc[results.model == "cls", key].to_numpy()
        b = results.loc[results.model == "mean", key].to_numpy()
        aggregates[key] = dict(cls=stats(a), mean=stats(b), mean_minus_cls=stats(b-a))
    for seed in ALL_SEEDS:
        a = next(row for row in rows if row["seed"] == seed and row["model"] == "cls")
        b = next(row for row in rows if row["seed"] == seed and row["model"] == "mean")
        paired.append(dict(seed=seed, **{key: b[key] - a[key] for _, key in METRICS}))
    per_label = frames[42, "cls"][["label_id", "label", "support"]].copy()
    for seed in ALL_SEEDS:
        for model in ("cls", "mean"):
            require(frames[seed, model][["label_id", "label", "support"]].equals(per_label[["label_id", "label", "support"]]), "Cross-seed label supports differ")
            per_label[f"{model}_seed{seed}_f1"] = frames[seed, model].f1
        per_label[f"mean_minus_cls_seed{seed}"] = frames[seed, "mean"].f1 - frames[seed, "cls"].f1
    for model in ("cls", "mean"):
        block = per_label[[f"{model}_seed{s}_f1" for s in ALL_SEEDS]]
        per_label[f"{model}_mean_f1"] = block.mean(axis=1)
        per_label[f"{model}_sample_sd"] = block.std(axis=1, ddof=1)
    block = per_label[[f"mean_minus_cls_seed{s}" for s in ALL_SEEDS]]
    per_label["mean_minus_cls_mean"] = block.mean(axis=1)
    per_label["mean_minus_cls_sample_sd"] = block.std(axis=1, ddof=1)
    per_label["mean_better_seeds"] = (block > 1e-12).sum(axis=1)
    per_label["mean_worse_seeds"] = (block < -1e-12).sum(axis=1)
    aggregated_frame = per_label.rename(columns={"cls_mean_f1": "cls_f1", "mean_mean_f1": "mean_f1"})
    aggregate_contribution = contribution(aggregated_frame, "cls_f1", "mean_f1")
    payload = dict(seeds=list(ALL_SEEDS), new_training_seeds=[123, 2026], seed42_reused=True,
                   formal_threshold=0.30, raw_metrics=rows, paired_deltas=paired,
                   aggregates=aggregates, diagnostics=diagnostics,
                   grief_relief_contributions=contributions, aggregate_contribution=aggregate_contribution,
                   provenance=provenance, seed42_reference=reference, source_sha256=check_sources())
    lines = ["# EXP5A_CLS_MEAN_MULTISEED_SUMMARY", "",
             "Three paired seeds: 42 (frozen reuse), 123 and 2026 (new independent raw-BERT training). Fixed threshold 0.30.", "",
             "## Test metrics: mean ± sample SD (%)", "", "| Metric | CLS | Mean pooling | Paired Mean - CLS (pp) | Positive / negative / zero seeds |",
             "|---|---:|---:|---:|---:|"]
    for label, key in METRICS:
        g = aggregates[key]
        cells = [f"{100*g[name]['mean']:.4f} ± {100*g[name]['sample_sd']:.4f}" for name in ("cls", "mean", "mean_minus_cls")]
        d = g["mean_minus_cls"]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {d['positive_seeds']} / {d['negative_seeds']} / {d['zero_seeds']} |")
    lines += ["", "## Per-seed formal test results (%)", "", "| Seed | Model | Micro-F1 | Macro-F1 | Rare-F1 | Weighted-F1 |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['model']} | " + " | ".join(f"{100*r[k]:.4f}" for k in ("micro_f1", "macro_f1", "rare_f1", "weighted_f1")) + " |")
    lines += ["", "## Paired deltas: Mean - CLS (pp)", "", "| Seed | Micro-F1 | Macro-F1 | Rare-F1 |", "|---|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['seed']} | " + " | ".join(f"{100*r[k]:+.4f}" for k in ("micro_f1", "macro_f1", "rare_f1")) + " |")
    lines += ["", "## Validation-selected checkpoint diagnostics", "", "| Seed | Model | Best epoch | Validation Macro-F1 (%) | Validation Micro-F1 (%) | Trainable params |", "|---|---|---:|---:|---:|---:|"]
    for d in diagnostics:
        lines.append(f"| {d['seed']} | {d['model']} | {d['best_epoch']} | {100*d['validation_macro_f1']:.4f} | {100*d['validation_micro_f1']:.4f} | {d['trainable_parameters']} |")
    lines += ["", "## All 28 labels: mean F1 (%) and paired differences", "", "| Label | Support | CLS mean±SD | Mean mean±SD | Mean - CLS (pp) | Mean better seeds |", "|---|---:|---:|---:|---:|---:|"]
    for r in per_label.itertuples():
        suffix = " (rare)" if r.label_id in RARE_IDS else ""
        lines.append(f"| {r.label}{suffix} | {r.support} | {100*r.cls_mean_f1:.4f} ± {100*r.cls_sample_sd:.4f} | {100*r.mean_mean_f1:.4f} ± {100*r.mean_sample_sd:.4f} | {100*r.mean_minus_cls_mean:+.4f} | {r.mean_better_seeds}/3 |")
    lines += ["", "## Grief + relief contributions (pp)", "", "| Seed | Macro delta | Grief+relief part | Other26 part | Rare delta | Grief+relief part | Other5 rare part |", "|---|---:|---:|---:|---:|---:|---:|"]
    contribution_rows = list(contributions.items()) + [("mean across seeds", aggregate_contribution)]
    for seed, d in contribution_rows:
        keys = ["macro_delta", "grief_relief_macro_contribution", "other26_macro_contribution", "rare_delta", "grief_relief_rare_contribution", "other5_rare_contribution"]
        lines.append(f"| {seed} | " + " | ".join(f"{100*d[k]:+.4f}" for k in keys) + " |")
    lines += ["", "Signed contributions are primary; fraction ratios can exceed100% or change sign when effects offset. Grief test support=6, relief=11; do not equate gains on these labels with general long-tail improvement.", "",
             "## Interpretation and scientific boundary", ""]
    for label, key in (("Macro-F1", "macro_f1"), ("Rare-F1", "rare_f1"), ("Micro-F1", "micro_f1")):
        d = aggregates[key]["mean_minus_cls"]
        lines.append(f"- {label}: paired average delta {100*d['mean']:+.4f} pp, sample SD {100*d['sample_sd']:.4f} pp; positive in {d['positive_seeds']}/3, negative in {d['negative_seeds']}/3.")
    macro = aggregates["macro_f1"]["mean_minus_cls"]
    rare = aggregates["rare_f1"]["mean_minus_cls"]
    if macro["positive_seeds"] == 3 and rare["positive_seeds"] == 3:
        interpretation = "Mean's Macro/Rare advantage has the same direction in all three tested seeds. This is repeated small-n evidence, not statistical significance or universal improvement."
    elif macro["negative_seeds"] == 3:
        interpretation = "CLS has higher Macro-F1 in all three tested seeds; the seed42 pooling hypothesis does not generalize to these tested runs as a Mean Macro advantage."
    else:
        interpretation = "Signs are mixed or tied across seeds/metrics; the seed42 direction should not be presented as a stable architecture advantage."
    lines += ["", interpretation, "",
             "- All models use fixed data/protocol/head/dropout; only representation and random seed differ. Each model independently starts from raw pretrained BERT, never from seed42/fine-tuned weights.",
             "- Every best checkpoint uses validation Macro-F1 at0.30; earliest ties. Test is evaluated once per new full model. Secondary thresholds are validation-only diagnostics.",
             "- Report averages per-seed metrics (sample SD ddof=1), not metrics of ensembled/pooled predictions. Paired-delta SD is computed directly across paired differences.",
             "- All seeds contribute regardless of which wins. No test-driven parameter/checkpoint/seed selection. Validation and test ranks may disagree; inspect the validation diagnostics rather than declaring a final model from test scores.",
             "- Only CLS and Mean receive new seeds. Query remains a seed42 pilot; no Query multi-seed mean or stable Query conclusion is claimed.",
             "- Three seeds provide limited repeatability evidence, not proof of equivalence, significance or mechanism. No automatic new method/seed or Exp5B.", ""]
    return payload, results, pd.DataFrame(paired), per_label, "\n".join(lines)


def compare(root, verify_only=False):
    root = Path(root)
    payload, results, paired, per_label, markdown = build(root)
    contents = {"EXP5A_CLS_MEAN_MULTISEED_SUMMARY.md": markdown,
                "MULTISEED_RAW_METRICS.csv": results.to_csv(index=False),
                "MULTISEED_PAIRED_DELTAS.csv": paired.to_csv(index=False),
                "MULTISEED_PER_LABEL.csv": per_label.to_csv(index=False)}
    json_name = "MULTISEED_COMPARISON.json"
    if verify_only:
        proof = read(root / "MULTISEED_SUCCESS.json")
        require(proof["status"] == "EXP5A_CLS_MEAN_MULTISEED_COMPLETE_VERIFIED" and proof["seeds"] == list(ALL_SEEDS) and proof["source_sha256"] == check_sources(), "Comparison identity differs")
        for name, digest in proof["artifacts_sha256"].items():
            require(sha(root / name) == digest, f"Comparison artifact changed: {name}")
        require(read(root / json_name) == payload, "Comparison JSON recomputation differs")
        for name, content in contents.items():
            require((root / name).read_text() == content, f"Comparison recomputation differs: {name}")
    else:
        require(not (root / "MULTISEED_SUCCESS.json").exists() and all(not (root / n).exists() for n in [*contents, json_name]), "Refusing comparison overwrite")
        write(root / json_name, payload)
        for name, content in contents.items():
            (root / name).write_text(content)
        evidence = [*contents, json_name, "preflight.json"] + [f"{m}_seed{s}/SUCCESS.json" for s in (123, 2026) for m in ("cls", "mean")]
        write(root / "MULTISEED_SUCCESS.json", dict(status="EXP5A_CLS_MEAN_MULTISEED_COMPLETE_VERIFIED",
              seeds=list(ALL_SEEDS), completed_at=now(), source_sha256=check_sources(),
              artifacts_sha256={n: sha(root / n) for n in evidence}))
        compare(root, verify_only=True)
    print("EXP5A_CLS_MEAN_MULTISEED_COMPLETE_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    compare(args.run_root, args.verify_only)
