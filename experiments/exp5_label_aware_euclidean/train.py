"""One-GPU FP32 Exp5A training with a strict smoke gate and single test pass."""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from data import Collator, LABEL_NAMES, RARE_IDS, RecordDataset, frequencies, load_all_verified, multi_hot
from exp5lib import (
    ASSET_MANIFEST, MODEL_DIR, REPO, capture, exp1_reference_dir, load_config,
    now, package_versions, read, require, require_runtime, sha, source_hashes, write,
)
from metrics import probabilities, rare_scores, score, tune_validation_threshold
from model import LabelAwareBert, make_optimizer, parameter_report


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def subset(records, count, seed):
    indices = np.random.RandomState(seed).permutation(len(records))[:count]
    return [records[int(index)] for index in indices]


def split_loader(records, tokenizer, config, batch_size, shuffle=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        RecordDataset(records), batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, collate_fn=Collator(tokenizer, config["max_length"]),
        num_workers=config["num_workers"], pin_memory=True,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, targets, ids = [], [], []
    loss_sum, count = 0.0, 0
    for batch in tqdm(loader, desc="Evaluation"):
        record_ids = batch.pop("record_ids")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(**batch)
        require(output.loss is not None and torch.isfinite(output.loss), "Non-finite evaluation loss")
        require(torch.isfinite(output.logits).all(), "Non-finite evaluation logits")
        size = batch["labels"].shape[0]
        loss_sum += float(output.loss) * size
        count += size
        logits.append(output.logits.cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())
        ids.extend(record_ids)
    return np.concatenate(logits), np.concatenate(targets), np.asarray(ids, dtype=str), loss_sum / count


def save_split(name, logits, targets, ids, selected, config, output_dir):
    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / f"{name}_predictions.npz", logits=logits, targets=targets, ids=ids)
    metrics, rare = {}, {}
    probs = probabilities(logits)
    for protocol, threshold in (("fixed_0.30", config["fixed_threshold"]), ("validation_selected", selected)):
        values, frame = score(targets, probs, threshold)
        metrics[protocol] = values
        rare[protocol] = rare_scores(frame)
        frame.to_csv(output_dir / f"{name}_{protocol}_per_label.csv", index=False)
        write(output_dir / f"{name}_{protocol}_metrics.json", values)
        if protocol == "fixed_0.30" and (name == "test" or config["mode"] == "smoke"):
            frame.to_csv(output_dir / "per_label_metrics.csv", index=False)
    write(output_dir / f"{name}_metrics.json", metrics)
    return metrics, rare


def load_checkpoint(path, config, device):
    model = LabelAwareBert.from_pretrained(str(MODEL_DIR), config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def attention_examples(model, tokenizer, validation_records, config, output_dir):
    rng = np.random.RandomState(config["seed"])
    chosen = [validation_records[int(index)] for index in rng.choice(len(validation_records), size=5, replace=False)]
    batch = Collator(tokenizer, config["max_length"])(chosen)
    record_ids = batch.pop("record_ids")
    batch = {key: value.to("cuda") for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        output = model(**batch, return_attention=True)
    probs = torch.sigmoid(output.logits).cpu().numpy()
    weights = output.attention_weights.cpu().numpy()
    input_ids = batch["input_ids"].cpu().numpy()
    masks = batch["attention_mask"].cpu().numpy()
    padded = np.broadcast_to(masks[:, None, :] == 0, weights.shape)
    require(np.max(weights[padded], initial=0.0) == 0.0, "PAD attention was nonzero")
    examples = []
    for row, record in enumerate(chosen):
        predicted = np.where(probs[row] >= config["fixed_threshold"])[0].tolist()
        focus = sorted(set(record["labels"] + predicted + [2, 14, 17]))
        valid_length = int(masks[row].sum())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[row, :valid_length].tolist())
        per_label = {}
        for label_id in focus:
            top = np.argsort(-weights[row, label_id, :valid_length])[:5]
            per_label[LABEL_NAMES[label_id]] = [
                {"token": tokens[int(position)], "weight": float(weights[row, label_id, int(position)])}
                for position in top
            ]
        examples.append(
            {
                "id": record_ids[row], "text": record["text"],
                "true_labels": [LABEL_NAMES[index] for index in record["labels"]],
                "predicted_labels_fixed_0.30": [LABEL_NAMES[index] for index in predicted],
                "label_attention": per_label,
            }
        )
    write(Path(output_dir) / "attention_examples.json", examples)
    lines = ["# Exp5A label-wise attention sanity examples", "", "These are diagnostics, not evaluation metrics.", ""]
    for index, example in enumerate(examples, 1):
        lines += [f"## Example {index}", "", f"- ID: `{example['id']}`", f"- Text: {example['text']}",
                  f"- True labels: {', '.join(example['true_labels'])}",
                  f"- Predicted labels at 0.30: {', '.join(example['predicted_labels_fixed_0.30']) or '(none)'}", ""]
        for label, items in example["label_attention"].items():
            lines.append(f"- {label}: " + ", ".join(f"`{item['token']}` ({item['weight']:.5f})" for item in items))
        lines.append("")
    (Path(output_dir) / "attention_examples.md").write_text("\n".join(lines), encoding="utf-8")


def compare_and_summarize(output_dir, metadata, config, test_metrics, test_rare):
    output_dir = Path(output_dir)
    reference = exp1_reference_dir()
    exp1 = read(reference / "test_metrics.json")["fixed_0.30"]
    exp1_rare = read(reference / "rare_label_metrics.json")["test"]["fixed_0.30"]["mean_rare_label_f1"]
    exp5 = test_metrics["fixed_0.30"]
    old = pd.read_csv(reference / "test_per_label_fixed_0.30.csv")
    new = pd.read_csv(output_dir / "test_fixed_0.30_per_label.csv")
    require(old[["label_id", "label", "support"]].sort_values("label_id").reset_index(drop=True).equals(
        new[["label_id", "label", "support"]].sort_values("label_id").reset_index(drop=True)
    ), "Exp1 and Exp5A per-label supports/order differ")
    comparison = old[["label_id", "label", "support", "f1"]].rename(columns={"f1": "exp1_f1"}).merge(
        new[["label_id", "f1"]].rename(columns={"f1": "exp5a_f1"}), on="label_id", validate="one_to_one"
    )
    comparison["delta"] = comparison["exp5a_f1"] - comparison["exp1_f1"]
    comparison.sort_values(["delta", "label_id"], ascending=[False, True]).to_csv(output_dir / "exp1_vs_exp5a_per_label.csv", index=False)
    metrics = [
        ("Micro-F1", "micro_f1", exp1.get("micro_f1")),
        ("Macro-F1", "macro_f1", exp1.get("macro_f1")),
        ("Macro Precision", "macro_precision", exp1.get("macro_precision")),
        ("Macro Recall", "macro_recall", exp1.get("macro_recall")),
        ("Weighted-F1", "weighted_f1", None),
        ("Rare-F1", "rare", exp1_rare),
    ]
    exp5_values = {**exp5, "rare": test_rare["fixed_0.30"]["mean_rare_label_f1"]}
    macro_delta = exp5["macro_f1"] - exp1["macro_f1"]
    answer = (
        "For this seed42 pilot, label-specific representations improved fixed-threshold Macro-F1 relative to the verified Exp1 reference."
        if macro_delta > 0 else
        "For this seed42 pilot, label-specific representations did not improve fixed-threshold Macro-F1 relative to the verified Exp1 reference."
    )
    lines = [
        "# EXP5A_SUMMARY", "",
        "## Research question", "",
        "Does replacing a single global `[CLS]` representation with label-specific representations obtained through learnable label queries improve fine-grained multi-label emotion recognition on GoEmotions under an otherwise matched BERT training setup?",
        "", answer, "",
        "> Scope: this is a single-seed Euclidean label-aware baseline. It is not a final model or a multi-seed conclusion.", "",
        "## Fixed-threshold 0.30 comparison", "",
        "| Metric | Exp1 Standard BERT | Exp5A Label-aware | Delta |", "|---|---:|---:|---:|",
    ]
    for label, key, old_value in metrics:
        new_value = exp5_values[key]
        if old_value is None:
            lines.append(f"| {label} | N/A in verified Exp1 artifact | {new_value:.6f} | N/A |")
        else:
            lines.append(f"| {label} | {old_value:.6f} | {new_value:.6f} | {new_value - old_value:+.6f} |")
    lines += ["", "The validation-selected global threshold is retained only as a secondary diagnostic; it is not the formal cross-experiment comparison.", "",
              "## Largest per-label changes", "", "### Top 10 improvements", "",
              "| Label | Support | Exp1 F1 | Exp5A F1 | Delta |", "|---|---:|---:|---:|---:|"]
    for row in comparison.nlargest(10, "delta").itertuples():
        lines.append(f"| {row.label} | {row.support} | {row.exp1_f1:.6f} | {row.exp5a_f1:.6f} | {row.delta:+.6f} |")
    lines += ["", "### Top 10 regressions", "", "| Label | Support | Exp1 F1 | Exp5A F1 | Delta |", "|---|---:|---:|---:|---:|"]
    for row in comparison.nsmallest(10, "delta").itertuples():
        lines.append(f"| {row.label} | {row.support} | {row.exp1_f1:.6f} | {row.exp5a_f1:.6f} | {row.delta:+.6f} |")
    lines += ["", "## Protocol and limits", "",
              f"- Best checkpoint: epoch {metadata['best_checkpoint']['epoch']}, selected only by validation Macro-F1 at fixed 0.30.",
              "- Test set was evaluated once after checkpoint and threshold selection were frozen.",
              "- Same official splits, 28-label order, max length 50, four epochs, batch 16, AdamW 5e-5, weight decay 0.01, linear warmup 0.1 and seed42 as Exp1.",
              "- The only intended model change is token-level label-wise attention with 28 learned queries and per-label linear heads.",
              "- No hierarchy, LGSA, graph, hyperbolic geometry, class weighting, resampling, contrastive/prototype loss or test-set tuning.",
              f"- Exp1 comparison reads three present artifacts whose SHA-256 values were verified against the reference manifest: {', '.join(metadata['exp1_reference']['verified_present_artifacts'])}.",
              f"- Historical Exp1 manifest note: {metadata['exp1_reference']['historical_archive_claim']}; live Slurm accounting was not re-queried in this Ubuntu session.",
              "- Suggested next step only if healthy: Exp5B Euclidean label prototypes. Do not execute it from this run.", ""]
    (output_dir / "EXP5A_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def verify_smoke_gate(smoke_dir, preflight, config):
    smoke_dir = Path(smoke_dir)
    success = read(smoke_dir / "SUCCESS.json")
    metadata = read(smoke_dir / "environment.json")
    smoke_config = read(smoke_dir / "config.json")
    require(success["status"] == "EXP5A_SMOKE_COMPLETE_VERIFIED", "Smoke success marker invalid")
    require(success["source_sha256"] == source_hashes() == preflight["source_sha256"], "Source changed after smoke")
    require(metadata["preflight_sha256"] == sha(Path(metadata["preflight_path"])), "Preflight proof changed")
    current_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    require(current_freeze == (smoke_dir / "pip-freeze.txt").read_text(), "Packages changed after smoke")
    for key, value in config.items():
        if key not in {"mode", "epochs"}:
            require(smoke_config.get(key) == value, f"Full config differs from smoke: {key}")


def run(phase, output_dir, preflight_path, smoke_dir=None):
    require_runtime(require_cuda=True)
    base_config = load_config()
    config = dict(base_config, mode=phase, epochs=1 if phase == "smoke" else base_config["epochs"])
    preflight = read(preflight_path)
    require(preflight["status"] == "EXP5A_PREFLIGHT_PASS", "Preflight did not pass")
    require(preflight["source_sha256"] == source_hashes(), "Source changed after preflight")
    if phase == "full":
        require(smoke_dir is not None, "Full run requires a verified smoke directory")
        verify_smoke_gate(smoke_dir, preflight, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    start = now()
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output_dir / "pip-freeze.txt").write_text(freeze)
    metadata = {
        "status": "running", "phase": phase, "seed": 42, "start_time": start, "end_time": None,
        "hostname": platform.node(), "python": platform.python_version(), "python_executable": sys.executable,
        "packages": package_versions(), "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "git_commit": capture(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_status": capture(["git", "-C", str(REPO), "status", "--porcelain"]),
        "source_sha256": source_hashes(), "preflight_path": str(Path(preflight_path).resolve()),
        "preflight_sha256": sha(preflight_path), "dataset_manifest_sha256": sha(ASSET_MANIFEST),
        "exp1_reference": preflight["exp1_reference"], "parameters": preflight["parameters"],
        "implementation_details": [
            "FP32, TF32 disabled, eager attention, deterministic algorithms required",
            "28 random learned label queries; scaled dot-product attention over BERT tokens; PAD mask before softmax",
            "Per-label independent linear weights; BCEWithLogitsLoss only",
            "No hierarchy, LGSA, graph, hyperbolic, prototype, contrastive, imbalance loss, resampling or augmentation",
        ],
    }
    write(output_dir / "config.json", config)
    write(output_dir / "environment.json", metadata)
    try:
        seed_everything(config["seed"])
        data, manifest = load_all_verified()
        train_records, validation_records = data["train"], data["validation"]
        if phase == "smoke":
            train_records = subset(train_records, config["smoke_train_samples"], config["seed"])
            validation_records = subset(validation_records, config["smoke_validation_samples"], config["seed"])
        metadata["official_split_sizes"] = {split: len(records) for split, records in data.items()}
        metadata["actual_split_sizes"] = {"train": len(train_records), "validation": len(validation_records), "test_evaluated": 0 if phase == "smoke" else len(data["test"])}
        metadata["model_asset"] = manifest["model"]
        write(output_dir / "sample_ids.json", {"train": [r["id"] for r in train_records], "validation": [r["id"] for r in validation_records]})
        counts = frequencies(data["train"])
        write(output_dir / "dataset_statistics.json", {"splits": metadata["official_split_sizes"], "num_labels": 28, "label_order": dict(enumerate(LABEL_NAMES)), "train_label_counts": counts.tolist(), "rare_ids": RARE_IDS})
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
        model = LabelAwareBert.from_pretrained(str(MODEL_DIR), config).to("cuda")
        metadata["parameters_runtime"] = parameter_report(model)
        train_loader = split_loader(train_records, tokenizer, config, config["per_device_train_batch_size"], shuffle=True)
        validation_loader = split_loader(validation_records, tokenizer, config, config["eval_batch_size"])
        optimizer = make_optimizer(model, config)
        steps_per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["epochs"]
        warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        metadata.update({"optimizer_steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps, "warmup_steps": warmup_steps})
        write(output_dir / "environment.json", metadata)
        history, best_f1, global_step = [], -1.0, 0
        checkpoint = output_dir / "checkpoint_best.pt"
        checkpoint_dir = output_dir / "checkpoint_tokenizer"
        validation_best = None
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated, epoch_examples, epoch_loss = 0, 0, 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            for batch_index, batch in enumerate(progress, 1):
                batch.pop("record_ids")
                batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                detailed = epoch == 1 and batch_index == 1
                output = model(**batch, return_attention=detailed)
                require(output.loss is not None and torch.isfinite(output.loss), "Non-finite training loss")
                require(torch.isfinite(output.logits).all(), "Non-finite training logits")
                if detailed:
                    pad = batch["attention_mask"][:, None, :] == 0
                    require(torch.equal(output.attention_weights.masked_select(pad), torch.zeros_like(output.attention_weights.masked_select(pad))), "PAD attention is nonzero")
                    expected = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, batch["labels"])
                    torch.testing.assert_close(output.loss, expected)
                    write(output_dir / "first_batch_shapes.json", {
                        "input_ids": list(batch["input_ids"].shape), "bert_hidden": list(output.hidden_states.shape),
                        "label_queries": list(model.label_queries.shape), "attention_scores": list(output.attention_weights.shape),
                        "attention_weights": list(output.attention_weights.shape), "label_specific_repr": list(output.label_representations.shape),
                        "logits": list(output.logits.shape), "targets": list(batch["labels"].shape), "pad_attention_max": 0.0,
                    })
                size = batch["labels"].shape[0]
                (output.loss * size).backward()
                if detailed:
                    require(model.label_queries.grad is not None, "label_queries received no gradient")
                    require(model.classifier_weight.grad is not None and model.classifier_bias.grad is not None, "classifier received no gradient")
                    require(any(parameter.grad is not None for parameter in model.encoder.parameters()), "BERT received no gradient")
                    require(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None), "Non-finite gradient")
                    metadata["first_batch_backward_verified"] = True
                accumulated += size
                epoch_examples += size
                epoch_loss += float(output.loss.detach()) * size
                if batch_index % config["gradient_accumulation_steps"] == 0 or batch_index == len(train_loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    global_step += 1
                progress.set_postfix(loss=f"{float(output.loss.detach()):.4f}", step=global_step)
            val_logits, val_targets, val_ids, val_loss = predict(model, validation_loader, "cuda")
            val_metrics, _ = score(val_targets, probabilities(val_logits), config["fixed_threshold"])
            row = {"epoch": epoch, "global_step": global_step, "training_loss": epoch_loss / epoch_examples,
                   "validation_loss": val_loss, "validation_macro_f1": val_metrics["macro_f1"],
                   "validation_micro_f1": val_metrics["micro_f1"], "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
            print(row, flush=True)
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                torch.save(model.state_dict(), checkpoint)
                tokenizer.save_pretrained(checkpoint_dir)
                validation_best = (val_logits.copy(), val_targets.copy(), val_ids.copy())
                write(output_dir / "checkpoint_selection.json", {"epoch": epoch, "global_step": global_step, "validation_macro_f1": best_f1, "threshold": 0.30})
        np.savez_compressed(output_dir / "validation_predictions.npz", logits=validation_best[0], targets=validation_best[1], ids=validation_best[2])
        metadata["best_checkpoint"] = read(output_dir / "checkpoint_selection.json")
        del model, optimizer, scheduler, output
        torch.cuda.empty_cache()
        best_model = load_checkpoint(checkpoint, config, "cuda")
        selected, search = tune_validation_threshold(validation_best[1], probabilities(validation_best[0]), config["threshold_candidates"])
        search.to_csv(output_dir / "threshold_search.csv", index=False)
        write(output_dir / "threshold_selection.json", {"selected_threshold": selected, "selection_split": "validation", "objective": "macro_f1", "tie_break": "lowest threshold", "formal_comparison_threshold": 0.30, "per_label_thresholds": False})
        validation_metrics, validation_rare = save_split("validation", validation_best[0], validation_best[1], validation_best[2], selected, config, output_dir)
        rare_report = {"definition": {"label_ids": RARE_IDS, "label_names": [LABEL_NAMES[i] for i in RARE_IDS]}, "validation": validation_rare}
        if phase == "full":
            test_loader = split_loader(data["test"], tokenizer, config, config["eval_batch_size"])
            test_logits, test_targets, test_ids, _ = predict(best_model, test_loader, "cuda")
            test_metrics, test_rare = save_split("test", test_logits, test_targets, test_ids, selected, config, output_dir)
            rare_report["test"] = test_rare
            attention_examples(best_model, tokenizer, data["validation"], config, output_dir)
            compare_and_summarize(output_dir, metadata, config, test_metrics, test_rare)
        else:
            reloaded, _, _, _ = predict(best_model, [next(iter(validation_loader))], "cuda")
            np.testing.assert_allclose(reloaded, validation_best[0][: len(reloaded)], rtol=1e-4, atol=1e-4)
            metadata["checkpoint_reload_verified"] = True
        write(output_dir / "rare_label_metrics.json", rare_report)
        require(source_hashes() == metadata["source_sha256"], "Source changed during training")
        metadata.update({"status": "completed", "end_time": now(), "completed_optimizer_steps": global_step})
        write(output_dir / "environment.json", metadata)
        from verify import verify_run
        verify_run(output_dir, require_success=False)
        status = "EXP5A_SMOKE_COMPLETE_VERIFIED" if phase == "smoke" else "EXPERIMENT_5A_COMPLETE_VERIFIED"
        write(output_dir / "SUCCESS.json", {"status": status, "experiment": "Label-aware Euclidean Baseline", "seed": 42,
              "completed_at": now(), "source_sha256": metadata["source_sha256"],
              "artifacts_sha256": {str(path.relative_to(output_dir)): sha(path) for path in output_dir.rglob("*") if path.is_file()}})
        print("=" * 50)
        print(status)
        print("Label-aware Euclidean Baseline")
        print("Seed: 42")
        print("=" * 50, flush=True)
    except BaseException as exc:
        metadata.update({"status": "failed", "end_time": now(), "error": repr(exc)})
        write(output_dir / "environment.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    run(args.phase, args.out, args.preflight, args.smoke_run)


if __name__ == "__main__":
    main()
