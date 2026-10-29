"""Machine-read paired comparison; never retrains C or selects methods on test."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from control_common import (C_RUN, RARE_IDS, check_sources, checked_artifacts, load_all_verified,
                            now, read, require, sha, verify_c, verify_prediction_metrics, write)
from control_verify import verify_control
from exp5lib import exp1_reference_dir
from preflight import verify_reference

METRICS = [("Micro-F1", "micro_f1"), ("Macro-F1", "macro_f1"),
           ("Macro Precision", "macro_precision"), ("Macro Recall", "macro_recall"),
           ("Weighted-F1", "weighted_f1"), ("Rare-F1", "rare_f1")]


def contribution(frame, left, right):
    delta = frame[right] - frame[left]
    tiny = frame.label.isin(["grief", "relief"])
    rare = frame.label_id.isin(RARE_IDS)
    macro_delta = float(delta.mean())
    rare_delta = float(delta.loc[rare].mean())
    tiny_sum = float(delta.loc[tiny].sum())
    return dict(macro_delta=macro_delta, grief_relief_macro_contribution=tiny_sum / 28,
                other26_macro_contribution=float(delta.loc[~tiny].sum()) / 28,
                other26_mean_f1_delta=float(delta.loc[~tiny].mean()), rare_delta=rare_delta,
                grief_relief_rare_contribution=tiny_sum / 7,
                other5_rare_contribution=float(delta.loc[rare & ~tiny].sum()) / 7,
                macro_contribution_fraction=None if abs(macro_delta) < 1e-12 else (tiny_sum / 28) / macro_delta,
                rare_contribution_fraction=None if abs(rare_delta) < 1e-12 else (tiny_sum / 7) / rare_delta,
                improved_labels=int((delta > 1e-12).sum()), regressed_labels=int((delta < -1e-12).sum()))


def build(root):
    root = Path(root)
    c_reference = verify_c()
    datasets, _ = load_all_verified()
    roots = dict(cls=root / "cls_seed42", mean=root / "mean_seed42", query=C_RUN)
    values, frames, diagnostics = {}, {}, {}
    for name, path in roots.items():
        if name != "query":
            verify_control(path)
            require(read(path / "environment.json")["c_reference"] == c_reference, "Control C provenance changed")
        metrics, rare, frames[name] = verify_prediction_metrics(path, "test", datasets["test"])
        values[name] = dict(metrics, rare_f1=rare["mean_rare_label_f1"])
        selection = read(path / "checkpoint_selection.json")
        val = read(path / "validation_metrics.json")["fixed_0.30"]
        env = read(path / "environment.json")
        parameters = env["parameters_runtime"]
        count = parameters["exp5a_trainable_parameters"] if name == "query" else parameters["trainable_parameters"]
        diagnostics[name] = dict(best_epoch=selection["epoch"], validation_macro_f1=val["macro_f1"],
                                 validation_micro_f1=val["micro_f1"], trainable_parameters=count,
                                 source_success_sha256=sha(path / "SUCCESS.json"))
    require(sha(roots["cls"] / "sample_ids.json") == sha(roots["mean"] / "sample_ids.json"), "CLS/mean actual sample orders differ")
    reference = verify_reference()
    historical = dict(reference["formal_metrics"], rare_f1=reference["rare_f1_fixed_0.30"])
    frame = frames["cls"][["label_id", "label", "support", "f1"]].rename(columns={"f1": "cls_f1"})
    for name in ("mean", "query"):
        require(frames[name][["label_id", "label", "support"]].equals(frames["cls"][["label_id", "label", "support"]]), "Paired label supports differ")
        frame[f"{name}_f1"] = frames[name].f1
    frame["mean_minus_cls"] = frame.mean_f1 - frame.cls_f1
    frame["query_minus_mean"] = frame.query_f1 - frame.mean_f1
    frame["query_minus_cls"] = frame.query_f1 - frame.cls_f1
    pairs = [("Mean - CLS", "cls", "mean"), ("Query - Mean", "mean", "query"), ("Query - CLS", "cls", "query")]
    deltas, contributions = {}, {}
    for title, left, right in pairs:
        deltas[title] = {key: values[right][key] - values[left][key] for _, key in METRICS}
        contributions[title] = contribution(frame, f"{left}_f1", f"{right}_f1")
    payload = dict(seed=42, formal_threshold=0.30, metrics=values, historical_exp1_reference=historical,
                   deltas=deltas, grief_relief_contributions=contributions, diagnostics=diagnostics,
                   c_reference=c_reference, source_sha256=check_sources())
    lines = ["# EXP5A_CONTROL_SUMMARY", "", "Seed42 only; fixed threshold 0.30. No test-driven checkpoint selection.", "",
             "## Formal comparison", "", "| Metric | Historical Exp1 (reference only) | Same-env raw CLS | Masked mean | Frozen label query |",
             "|---|---:|---:|---:|---:|"]
    for label, key in METRICS:
        old = f"{historical[key]:.6f}" if key in historical else "N/A in verified artifact"
        lines.append(f"| {label} | {old} | {values['cls'][key]:.6f} | {values['mean'][key]:.6f} | {values['query'][key]:.6f} |")
    lines += ["", "Micro precision/recall and weighted precision/recall are also saved in each test_metrics.json and CONTROL_COMPARISON.json.", "",
              "## Paired deltas (percentage points)", "", "| Metric | Mean - CLS | Query - Mean | Query - CLS |", "|---|---:|---:|---:|"]
    for label, key in METRICS:
        lines.append(f"| {label} | " + " | ".join(f"{100*deltas[t][key]:+.4f}" for t, _, _ in pairs) + " |")
    lines += ["", "## Checkpoint and parameter diagnostics", "", "| Model | Best epoch | Validation Macro-F1 | Validation Micro-F1 | Trainable parameters |", "|---|---:|---:|---:|---:|"]
    for name, d in diagnostics.items():
        lines.append(f"| {name} | {d['best_epoch']} | {d['validation_macro_f1']:.6f} | {d['validation_micro_f1']:.6f} | {d['trainable_parameters']} |")
    lines += ["", "## All 28 labels", "", "| Label | Support | CLS F1 | Mean F1 | Query F1 | Mean - CLS | Query - Mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        suffix = " (rare)" if row.label_id in RARE_IDS else ""
        lines.append(f"| {row.label}{suffix} | {row.support} | {row.cls_f1:.6f} | {row.mean_f1:.6f} | {row.query_f1:.6f} | {row.mean_minus_cls:+.6f} | {row.query_minus_mean:+.6f} |")
    lines += ["", "## Grief + relief contribution (percentage points)", "", "| Contrast | Macro delta | Grief+relief part | Other26 part | Rare delta | Grief+relief part | Other5 rare part |", "|---|---:|---:|---:|---:|---:|---:|"]
    for title, d in contributions.items():
        keys = ["macro_delta", "grief_relief_macro_contribution", "other26_macro_contribution", "rare_delta", "grief_relief_rare_contribution", "other5_rare_contribution"]
        lines.append(f"| {title} | " + " | ".join(f"{100*d[k]:+.4f}" for k in keys) + " |")
        fraction = d["macro_contribution_fraction"]
        fraction_text = "undefined (zero net change)" if fraction is None else f"{100*fraction:.2f}%"
        lines += [f"", f"{title}: grief+relief account for {fraction_text} of net Macro change; {d['improved_labels']} labels improve and {d['regressed_labels']} regress."]
    lines += ["", "Contributions are signed; ratios can be negative or exceed 100% when other labels offset them. Raw signed contributions are primary. Grief support=6 and relief support=11; concentrated gains are not evidence of general long-tail improvement.", "",
              "## Interpretation and decision boundary", "",
              "No arbitrary post-hoc 'clearly better' or 'approximately equal' tolerance is used. This seed42 diagnostic supports numerical attribution clues, not statistical consistency, equivalence or causal proof.", ""]
    macro_mc, macro_qm = deltas["Mean - CLS"]["macro_f1"], deltas["Query - Mean"]["macro_f1"]
    lines += [f"Observed Macro-F1: Mean - CLS = {100*macro_mc:+.4f} pp; Query - Mean = {100*macro_qm:+.4f} pp.", "",
              "- If mean substantially beats CLS while query and mean remain practically close across metrics/labels, results are compatible with pooling explaining most gains. Do not claim statistical equivalence from one seed.",
              "- If query exceeds mean across important metrics and many labels, this suggests incremental query-training value; multi-seed evidence is needed before calling it consistent.",
              "- If all are close, no robust representation advantage is established; do not automatically start more seeds.", "",
              "## Matched implementation and limits", "",
              "- A uses raw last_hidden_state[:,0,:], not historical Exp1's learned tanh pooler; Exp1 is reference only.",
              "- B averages every unmasked token, including CLS/SEP, matching uniform attention in C. PAD is excluded.",
              "- A/B use the same classifier weights/bias and per-label independent dropout layout as frozen C; this is mathematically a D-to-28 linear head at evaluation. A shared [B,D] dropout would introduce an extra training difference, so is not used.",
              "- One discarded initialization draw (not a model parameter/module) matches C's initial classifier and CPU/CUDA RNG states, verified on the target environment. CLS/mean have no queries or token-attention modules.",
              "- Data loader, tokenization, shuffle seed, optimizer grouping, sample-normalized gradients, scheduler/warmup, validation metrics and threshold code are read-only reused from frozen C. Four epochs / seed42 only / BCE only.",
              "- C is reused, never retrained. All C hashes are verified and formal metrics recomputed on CPU from saved logits and canonical test labels; no new GPU test pass for C.",
              "- Best epoch is highest validation Macro-F1 at 0.30, earliest on ties. Each control evaluates test only once after checkpoint selection. Secondary threshold diagnostics are validation-selected and not the formal comparison.",
              "- Independent training changes backbone optimization trajectories; this comparison does not isolate an inference-only attention intervention. Historical C's training sample_ids prove dataset membership/order, not a stored per-epoch permutation trace.",
              "- Earlier inference diagnostic was 128 validation texts / 3584 label decisions, not 3584 examples. It does not prove queries have no training effect.",
              "- No prototype, ASL, focal/class-weight/contrastive loss, graph, hierarchy, hyperbolic geometry, label semantics or extra transformer. No Exp5B or extra seeds are started.", ""]
    return payload, frame, "\n".join(lines)


def compare(root, verify_only=False):
    root = Path(root)
    payload, frame, markdown = build(root)
    names = ["EXP5A_CONTROL_SUMMARY.md", "CONTROL_COMPARISON.json", "CONTROL_PER_LABEL_COMPARISON.csv"]
    if verify_only:
        proof = read(root / "CONTROL_SUCCESS.json")
        require(proof["status"] == "EXP5A_CONTROL_COMPLETE_VERIFIED" and proof["source_sha256"] == check_sources(), "Comparison success identity differs")
        for name, digest in proof["artifacts_sha256"].items():
            require(sha(root / name) == digest, f"Comparison artifact changed: {name}")
        require(read(root / names[1]) == payload, "Comparison JSON recomputation differs")
        require((root / names[0]).read_text() == markdown, "Summary recomputation differs")
        expected_csv = frame.to_csv(index=False)
        require((root / names[2]).read_text() == expected_csv, "Per-label comparison recomputation differs")
    else:
        require(not (root / "CONTROL_SUCCESS.json").exists() and all(not (root / n).exists() for n in names), "Refusing to overwrite comparison")
        write(root / names[1], payload)
        frame.to_csv(root / names[2], index=False)
        (root / names[0]).write_text(markdown)
        evidence = names + ["preflight.json", "cls_seed42/SUCCESS.json", "mean_seed42/SUCCESS.json"]
        write(root / "CONTROL_SUCCESS.json", dict(status="EXP5A_CONTROL_COMPLETE_VERIFIED", seed=42,
              completed_at=now(), source_sha256=check_sources(), artifacts_sha256={n: sha(root / n) for n in evidence}))
        compare(root, verify_only=True)
    print("=" * 50, "EXP5A_CONTROL_COMPLETE_VERIFIED", "CLS vs Mean Pooling vs Label Query", "Seed: 42", "=" * 50, sep="\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    compare(args.run_root, args.verify_only)
