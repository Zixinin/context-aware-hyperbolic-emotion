"""Actual GPU smoke, four epochs, full validation reload, one formal test, CPU verification."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import traceback
from pathlib import Path

import numpy as np

from common import (CONFIG, HERE, check_package, check_reference, check_seal, find_snapshots,
                    now, protected_inventory, read, require, run_path, runtime,
                    seal, sha, snapshot_inventory, write)
from data27 import load_dataset, loader, save_audit, targets
from metrics27 import probabilities, save_predictions, score, write_csv


SUCCESS = "EXP6C0_ROBERTA27_MULTILABEL_BASELINE_VERIFIED"


def stage(root, name, **extra):
    write(root / "state.json", dict(stage=name, time=now(), **extra))
    print("STAGE", name, flush=True)


def identity(snapshot, model, tokenizer, loading_info):
    files = snapshot_inventory(snapshot)
    require(re.fullmatch(r"[0-9a-f]{40}", snapshot.name) is not None,
            "Snapshot identity must be an immutable 40-character HF commit; no unversioned directory")
    result = dict(snapshot_path=str(snapshot), model_revision=snapshot.name, tokenizer_revision=snapshot.name,
                checkpoint_sha256={k: v for k, v in files.items() if k in {"model.safetensors", "pytorch_model.bin"}},
                all_snapshot_files_sha256=files, model_config=model.encoder.config.to_dict(),
                tokenizer_class=type(tokenizer).__name__, tokenizer_vocab_size=tokenizer.vocab_size,
                tokenizer_special_tokens={"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
                tokenizer_init_kwargs=tokenizer.init_kwargs, backbone_loading_info=loading_info,
                total_parameters=sum(p.numel() for p in model.parameters()),
                classifier_parameters=model.classifier_weight.numel() + model.classifier_bias.numel())
    return json.loads(json.dumps(result, default=str))


def assert_plan(root):
    root = run_path(root)
    plan = read(root / "plan.json")
    require(plan["config"] == CONFIG, "Frozen config changed")
    require(plan["package_sha256"] == check_package(), "Source differs from smoke")
    require(plan["protocol_reference"] == check_reference(), "Historical protocol reference differs")
    current_runtime = runtime()
    stable = lambda env: {k: v for k, v in env.items() if k not in {"nvidia_smi"}}
    require(stable(plan["runtime"]) == stable(current_runtime), "Environment/git status differs from smoke; inspect before training")
    snapshot = Path(plan["snapshot"])
    require(plan["snapshot_files_sha256"] == snapshot_inventory(snapshot), "Pretrained snapshot changed")
    data, manifest = load_dataset()
    require(read(root / "dataset_manifest.json") == manifest, "Dataset changed since audit")
    require(read(root / "protected_before.json") == protected_inventory(), "Existing research files changed")
    return plan, data, manifest, snapshot


def smoke(root, snapshot_arg=None):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from model27 import gradients, load_model, optimizer_for, seed_everything, train_step
    root = run_path(root)
    require(not (root / "plan.json").exists(), "Smoke output already exists; use a fresh run")
    stage(root, "smoke_readonly_provenance")
    snapshots = find_snapshots()
    if snapshot_arg is None:
        require(len(snapshots) == 1, f"Need exactly one verified local RoBERTa snapshot or --snapshot; found: {snapshots}")
        snapshot = Path(snapshots[0])
    else:
        snapshot = Path(snapshot_arg).absolute()
        require(str(snapshot) in snapshots, "Snapshot must be a discovered canonical local RoBERTa snapshot")
    require(re.fullmatch(r"[0-9a-f]{40}", snapshot.name) is not None, "Unversioned snapshot refused")
    package, reference, env = check_package(), check_reference(), runtime()
    protected = protected_inventory()
    require(protected, "No existing files found for protection audit")
    write(root / "protected_before.json", protected)
    data, manifest = load_dataset()
    save_audit(root, manifest)
    write(root / "config.json", CONFIG)
    plan = dict(config=CONFIG, package_sha256=package, protocol_reference=reference, runtime=env,
                snapshot=str(snapshot), snapshot_files_sha256=snapshot_inventory(snapshot), created=now())
    write(root / "plan.json", plan)
    shutil.copyfile(HERE / "IMPLEMENTATION_AUDIT.md", root / "IMPLEMENTATION_AUDIT.md")
    shutil.copytree(HERE, root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    directory = root / "smoke"
    directory.mkdir(exist_ok=False)
    seed_everything()
    stage(root, "smoke_actual_cuda_forward_backward")
    model, tokenizer, loading = load_model(snapshot)
    provenance = identity(snapshot, model, tokenizer, loading)
    write(root / "model_identity.json", provenance)
    model = model.to("cuda")
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Expected FP32 model")
    optimizer, groups = optimizer_for(model)
    scheduler = get_linear_schedule_with_warmup(optimizer, 1, 2)
    # Guaranteed mixed cardinalities in actual canonical training examples.
    singles = [r for r in data["train"] if len(r["labels"]) == 1][:8]
    multis = [r for r in data["train"] if len(r["labels"]) > 1][:8]
    require(len(singles) == len(multis) == 8, "Cannot build smoke with real single/multi-label samples")
    smoke_rows = singles + multis
    batch = next(iter(loader(smoke_rows, tokenizer, shuffle=True)))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    report = train_step(model, batch, optimizer, scheduler, detailed=True)
    require(report["multi_examples"] == 8 and report["logits_shape"] == [16, 27], "Smoke multi-hot/shapes failed")
    require(optimizer.param_groups[0]["lr"] == 5e-5 and scheduler.last_epoch == 1, "Optimizer/scheduler smoke failed")
    # This checkpoint is smoke-only and cannot be mistaken for the formal best.
    torch.save(model.state_dict(), directory / "smoke_weights.pt")
    sample = data["validation"][:64]
    before = predict(model, loader(sample, tokenizer), "smoke validation")
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, tokenizer2, _ = load_model(snapshot)
    reloaded.load_state_dict(torch.load(directory / "smoke_weights.pt", map_location="cpu", weights_only=True), strict=True)
    reloaded.to("cuda")
    after = predict(reloaded, loader(sample, tokenizer2), "smoke checkpoint reload")
    reload_proof = compare_reload(before, after, manifest["names27"])
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    require(protected == protected_inventory(), "Existing research files changed during smoke")
    require(package == check_package(), "Package changed during smoke")
    require(plan["snapshot_files_sha256"] == snapshot_inventory(snapshot), "Pretrained snapshot changed during smoke")
    write(directory / "SMOKE_REPORT.json", dict(status="PASS", actual_gpu=True, shapes_and_gradients=report,
          optimizer_groups=groups, scheduler_steps=1, reload=reload_proof,
          dataset_splits={s: len(rows) for s, rows in data.items()}, formal_training=False,
          test_inference_calls=0, protected_file_count=len(protected), completed=now()))
    seal(directory, "SMOKE_VERIFIED.json", dict(status="EXP6C0_SMOKE_VERIFIED", package_sha256=package,
                                               plan_sha256=sha(root / "plan.json")))
    stage(root, "smoke_verified_waiting_for_formal_launch")
    print("EXP6C0_SMOKE_VERIFIED", root, flush=True)
    print("NEXT_AFTER_REVIEW: bash", HERE / "launch.sh", "train", root, flush=True)


def predict(model, batches, description):
    import torch
    model.eval()
    logs, gold, ids = [], [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(batches, 1):
            record_ids = batch.pop("ids")
            x = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            output = model(**x)
            require(bool(torch.isfinite(output["logits"]).all()) and bool(torch.isfinite(output["loss"])), "Nonfinite evaluation")
            logs.append(output["logits"].cpu().numpy())
            gold.append(x["labels"].cpu().numpy())
            ids.extend(record_ids)
            loss_sum += float(output["loss"]) * len(record_ids)
            n += len(record_ids)
            if i % 25 == 0:
                print(description, "batches", i, "examples", n, flush=True)
    require(n > 0, "Empty evaluation split")
    return np.concatenate(logs), np.concatenate(gold), np.asarray(ids, dtype=str), loss_sum / n


def compare_reload(before, after, names):
    a, b = before, after
    require(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), "Reload IDs/targets differ")
    np.testing.assert_allclose(a[0], b[0], atol=1e-6, rtol=1e-6)
    require(np.array_equal(probabilities(a[0]) >= 0.30, probabilities(b[0]) >= 0.30), "Reload threshold predictions differ")
    ma = score(a[1], probabilities(a[0]), names)[0]
    mb = score(b[1], probabilities(b[0]), names)[0]
    require(ma == mb, "Reload fixed-threshold metrics differ")
    return dict(num_examples=len(a[2]), ids_equal=True, targets_equal=True, threshold_predictions_equal=True,
                metrics_equal=True, max_abs_logit_error=float(np.max(np.abs(a[0] - b[0]))), atol=1e-6, rtol=1e-6)


def train(root):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from model27 import load_model, optimizer_for, seed_everything, train_step
    plan, data, manifest, snapshot = assert_plan(root)
    root = run_path(root)
    smoke_proof = check_seal(root / "smoke", "SMOKE_VERIFIED.json")
    require(smoke_proof["package_sha256"] == plan["package_sha256"] and
            smoke_proof["plan_sha256"] == sha(root / "plan.json"), "Smoke does not match formal plan")
    # Exclusive creation: a partial/failed formal run is never silently rerun.
    with (root / "FORMAL_TRAINING_CLAIM.json").open("x") as f:
        __import__("json").dump(dict(seed=42, epochs=4, started=now(), no_automatic_retry=True), f)
    env = dict(plan["runtime"], start_time=now(), status="training", completed_epochs=0,
               completed_optimizer_steps=0, formal_test_inference_calls=0)
    write(root / "environment.json", env)
    stage(root, "formal_seed42_fresh_initialization")
    seed_everything()
    model, tokenizer, loading = load_model(snapshot)
    require(identity(snapshot, model, tokenizer, loading) == read(root / "model_identity.json"), "Model/tokenizer differs from smoke")
    model.to("cuda")
    training = loader(data["train"], tokenizer, shuffle=True)
    validation = loader(data["validation"], tokenizer)
    optimizer, groups = optimizer_for(model)
    total_steps = len(training) * 4
    warmup = math.ceil(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    write(root / "optimizer_protocol.json", dict(parameter_groups=groups, optimizer="torch.optim.AdamW",
        betas=[0.9, 0.999], eps=1e-8, total_steps=total_steps, steps_per_epoch=len(training),
        warmup_steps=warmup, scheduler="linear", gradient_accumulation_steps=1,
        normalization="(mean_BCE*batch_size).backward; divide gradients by batch_size; clip_norm_1; AdamW; scheduler",
        sample_count=len(data["train"]), drop_last=False, deterministic=True, precision="FP32"))
    env.update(total_optimizer_steps=total_steps, warmup_steps=warmup, optimizer_steps_per_epoch=len(training))
    write(root / "environment.json", env)
    checkpoint = root / "best_checkpoint"
    checkpoint.mkdir(exist_ok=False)
    history, best, step = [], -1.0, 0
    with (root / "batch_log.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "batch", "step", "batch_size", "loss", "unclipped_grad_norm", "lr_used", "lr_after"])
        writer.writeheader()
        for epoch in range(1, 5):
            stage(root, f"training_epoch_{epoch}")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum, n = 0.0, 0
            for index, batch in enumerate(training, 1):
                lr_used = optimizer.param_groups[0]["lr"]
                details = train_step(model, batch, optimizer, scheduler, detailed=index == 1)
                if index == 1:
                    write(root / f"epoch{epoch}_first_batch.json", details)
                step += 1
                n += details["batch_size"]
                loss_sum += details["loss"] * details["batch_size"]
                writer.writerow(dict(epoch=epoch, batch=index, step=step, batch_size=details["batch_size"],
                    loss=details["loss"], unclipped_grad_norm=details["unclipped_grad_norm"],
                    lr_used=lr_used, lr_after=optimizer.param_groups[0]["lr"]))
                if index % 50 == 0 or index == len(training):
                    f.flush()
                    print(f"seed42 epoch={epoch}/4 batch={index}/{len(training)} step={step}/{total_steps} loss={details['loss']:.6f}", flush=True)
            require(n == len(data["train"]), "Training omitted/repeated examples")
            val = predict(model, validation, f"epoch{epoch} validation")
            values = score(val[1], probabilities(val[0]), manifest["names27"])[0]
            row = dict(epoch=epoch, global_step=step, training_examples=n, training_loss=loss_sum / n,
                       validation_loss=val[3], validation_macro_f1=values["macro_f1"],
                       validation_micro_f1=values["micro_f1"], learning_rate=optimizer.param_groups[0]["lr"])
            history.append(row)
            write_csv(root / "training_log.csv", history)
            write(root / f"epoch{epoch}_validation_metrics.json", values)
            print(row, flush=True)
            if values["macro_f1"] > best:
                best = values["macro_f1"]
                torch.save(model.state_dict(), checkpoint / "model.pt")
                tokenizer.save_pretrained(checkpoint / "tokenizer")
                model.encoder.config.save_pretrained(checkpoint / "encoder_config")
                save_predictions(root, "validation_selected", *val[:3], manifest["names27"])
                write(root / "checkpoint_selection.json", dict(epoch=epoch, global_step=step,
                    validation_macro_f1=best, threshold=0.30, selection_split="validation",
                    tie_break="earliest epoch via strict greater-than", rule=CONFIG["checkpoint_rule"],
                    checkpoint_sha256=sha(checkpoint / "model.pt")))
            env.update(completed_epochs=epoch, completed_optimizer_steps=step)
            write(root / "environment.json", env)
    require(step == total_steps, "Training did not finish all scheduled steps")
    del model, optimizer, scheduler, val
    gc.collect()
    torch.cuda.empty_cache()
    stage(root, "best_checkpoint_full_validation_reload")
    best_model, tokenizer, _ = load_model(snapshot)
    best_model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True), strict=True)
    best_model.to("cuda")
    # Use tokenizer serialized with the best checkpoint; verify its prediction parity.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True, use_fast=True)
    reloaded = predict(best_model, loader(data["validation"], tokenizer), "full validation reload")
    with np.load(root / "validation_selected_predictions.npz", allow_pickle=False) as z:
        previous = (z["logits"], z["targets"], z["ids"])
    reload_proof = compare_reload(previous, reloaded, manifest["names27"])
    require(reload_proof["num_examples"] == len(data["validation"]), "Reload must cover entire validation")
    write(root / "checkpoint_reload_verification.json", reload_proof)
    _, val_rare = save_predictions(root, "validation", *reloaded[:3], manifest["names27"])
    # Claim BEFORE entering test inference: failures never trigger an automatic second test pass.
    with (root / "FORMAL_TEST_CLAIM.json").open("x") as f:
        __import__("json").dump(dict(threshold=0.30, checkpoint_sha256=sha(checkpoint / "model.pt"),
                                   call=1, started=now(), validation_reload_passed=True), f)
    env.update(status="formal_test_started", formal_test_inference_calls=1)
    write(root / "environment.json", env)
    stage(root, "single_formal_test")
    tested = predict(best_model, loader(data["test"], tokenizer), "formal test (only call)")
    test_metrics, test_rare = save_predictions(root, "test", *tested[:3], manifest["names27"])
    write(root / "rare_label_metrics.json", dict(validation=val_rare, test=test_rare))
    del best_model
    gc.collect()
    torch.cuda.empty_cache()
    stage(root, "cpu_saved_prediction_and_immutable_input_verification")
    require(check_package() == plan["package_sha256"], "Package changed during training")
    require(protected_inventory() == read(root / "protected_before.json"), "Existing research files changed during training")
    require(snapshot_inventory(snapshot) == plan["snapshot_files_sha256"], "Pretrained snapshot changed during training")
    env.update(status="training_and_single_test_completed_verification_pending", end_time=now(),
               validation_reload_verified=True, existing_files_unchanged=True)
    write(root / "environment.json", env)
    verify(root, sealed=False)
    summarize(root, manifest, test_metrics)
    (root / "VERIFICATION_REPORT.md").write_text(
        "# Exp6C-0 verification\n\nPASS: canonical split/mapping SHA-256; 27-label filtering; multi-label retention; "
        "actual CUDA RoBERTa forward/backward and nonzero finite encoder/head gradients; optimizer/scheduler; "
        "four epochs and dynamically derived steps; validation-only earliest-tie selection; full validation checkpoint reload; "
        "one formal test claim/call; canonical-ID/gold alignment; all aggregate/per-label/rare metrics independently "
        "recomputed from saved predictions; package/model/data identities; existing research file hashes unchanged.\n\n"
        "No threshold search, extra seed, extra training, or supplementary neutral-only inference.\n"
        "Independent verification command reuses predictions only; it never loads a model or calls test inference.\n")
    env["status"] = "COMPLETE"
    write(root / "environment.json", env)
    stage(root, "COMPLETE")
    seal(root, "SUCCESS.json", dict(status=SUCCESS, completed=now(), seed=42, epochs=4,
                                    test_inference_calls=1, package_sha256=plan["package_sha256"]))
    check_seal(root, "SUCCESS.json")
    print("=" * 50, SUCCESS, "Backbone: roberta-base", "Task: 27-label multi-label emotion classification",
          "Loss: BCEWithLogitsLoss", "Seed: 42", "Status: COMPLETE", "=" * 50, sep="\n", flush=True)


def verify(root, sealed=True):
    """CPU-only repeatable result verification, never repeats inference."""
    root = run_path(root)
    if sealed:
        require(check_seal(root, "SUCCESS.json")["status"] == SUCCESS, "Missing success proof")
    plan = read(root / "plan.json")
    require(read(root / "config.json") == CONFIG == plan["config"], "Config mismatch")
    require(plan["package_sha256"] == check_package(), "Source changed")
    check_reference()
    data, manifest = load_dataset()
    require(manifest == read(root / "dataset_manifest.json") == read(root / "DATASET_MANIFEST.json"), "Dataset manifest mismatch")
    selection = read(root / "checkpoint_selection.json")
    with (root / "training_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    steps_per_epoch = math.ceil(len(data["train"]) / 16)
    require([int(r["epoch"]) for r in rows] == [1, 2, 3, 4], "Need four complete epochs")
    require([int(r["global_step"]) for r in rows] == [steps_per_epoch * i for i in range(1, 5)], "Step counts differ")
    require(all(int(r["training_examples"]) == len(data["train"]) for r in rows), "Wrong examples/epoch")
    require(all(np.isfinite(float(r[k])) for r in rows for k in ["training_loss", "validation_loss", "validation_macro_f1"]), "Nonfinite history")
    chosen = max(rows, key=lambda r: float(r["validation_macro_f1"]))
    require(selection["epoch"] == int(chosen["epoch"]) and selection["global_step"] == int(chosen["global_step"])
            and selection["validation_macro_f1"] == float(chosen["validation_macro_f1"])
            and selection["threshold"] == 0.30 and selection["selection_split"] == "validation", "Selection mismatch")
    require(selection["checkpoint_sha256"] == sha(root / "best_checkpoint/model.pt"), "Best checkpoint changed")
    opt = read(root / "optimizer_protocol.json")
    require(opt["total_steps"] == 4 * steps_per_epoch and opt["warmup_steps"] == math.ceil(4 * steps_per_epoch * 0.1), "Scheduler counts differ")
    with (root / "batch_log.csv").open() as f:
        batches = list(csv.DictReader(f))
    require(len(batches) == 4 * steps_per_epoch and [int(r["step"]) for r in batches] == list(range(1, 4 * steps_per_epoch + 1)), "Batch log incomplete")
    require(all(np.isfinite(float(r[k])) for r in batches for k in ["loss", "unclipped_grad_norm", "lr_used", "lr_after"]), "Nonfinite batch values")
    require(float(batches[-1]["lr_after"]) == 0, "Linear scheduler did not finish")
    for epoch in range(1, 5):
        gradients = read(root / f"epoch{epoch}_first_batch.json")["gradients"]
        require(all(g["norm"] > 0 and np.isfinite(g["norm"]) for g in gradients.values()), "Invalid encoder/head gradient")
    names = manifest["names27"]
    all_rare = read(root / "rare_label_metrics.json")
    for split in ["validation", "test"]:
        with np.load(root / f"{split}_predictions.npz", allow_pickle=False) as z:
            require(np.array_equal(z["ids"], [r["id"] for r in data[split]]), "Canonical prediction IDs differ")
            require(np.array_equal(z["targets"], targets(data[split])), "Canonical gold targets differ")
            require(np.array_equal(z["label_names"], names), "Prediction label order differs")
            probs = probabilities(z["logits"])
            require(np.array_equal(z["probabilities"], probs), "Saved probabilities differ")
            metrics, per_label, rare, pred = score(z["targets"], probs, names)
            require(np.array_equal(pred, z["predictions"]), "Saved binary predictions differ")
            require(np.array_equal(z["predicted_positive_counts"], pred.sum(1)) and
                    np.array_equal(z["true_positive_counts"], z["targets"].sum(1)), "Per-example prediction counts differ")
        require(metrics == read(root / f"{split}_metrics.json"), "Metric recomputation mismatch")
        require(rare == all_rare[split], "Rare-F1 recomputation mismatch")
        with (root / f"{split}_per_label_metrics.csv").open() as f:
            actual = list(csv.DictReader(f))
        require(len(actual) == 27, "Missing per-label metrics")
        for a, b in zip(actual, per_label):
            require(set(a) == set(b), "Per-label columns differ")
            for k, v in b.items():
                require(a[k] == v if isinstance(v, str) else abs(float(a[k]) - v) <= 1e-12, f"Per-label {k} differs")
        # Independent count-based cross-check of sklearn and all seven rare F1s.
        tp = np.asarray([r["tp"] for r in per_label], dtype=float)
        fp = np.asarray([r["fp"] for r in per_label], dtype=float)
        fn = np.asarray([r["fn"] for r in per_label], dtype=float)
        denom = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros(27), where=denom > 0)
        np.testing.assert_allclose(f1, [r["f1"] for r in per_label], atol=1e-15, rtol=0)
        require(abs(f1.mean() - metrics["macro_f1"]) < 1e-15, "Manual Macro-F1 differs")
        if split == "validation":
            require(metrics["macro_f1"] == selection["validation_macro_f1"], "Selected validation score differs")
    reload = read(root / "checkpoint_reload_verification.json")
    require(reload["num_examples"] == len(data["validation"]) and reload["threshold_predictions_equal"]
            and reload["metrics_equal"] and reload["ids_equal"] and reload["targets_equal"], "Full validation reload missing")
    test_claim = read(root / "FORMAL_TEST_CLAIM.json")
    require(test_claim["call"] == 1 and test_claim["threshold"] == 0.30 and test_claim["checkpoint_sha256"] == selection["checkpoint_sha256"], "Single test claim differs")
    env = read(root / "environment.json")
    require(env["completed_epochs"] == 4 and env["completed_optimizer_steps"] == 4 * steps_per_epoch and
            env["formal_test_inference_calls"] == 1 and env["existing_files_unchanged"], "Formal runtime proof incomplete")
    require(protected_inventory() == read(root / "protected_before.json"), "Protected old research files differ")
    require(snapshot_inventory(Path(plan["snapshot"])) == plan["snapshot_files_sha256"], "Pretrained snapshot differs")
    print("EXP6C0_SAVED_ARTIFACT_VERIFICATION_PASS", root, flush=True)


def summarize(root, manifest, metrics):
    env = read(root / "environment.json")
    selection = read(root / "checkpoint_selection.json")
    with (root / "test_per_label_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    ranked = sorted(rows, key=lambda r: (-float(r["f1"]), int(r["label_id"])))
    lines = ["# EXP6C0_SUMMARY", "", "RoBERTa-base + 27-label multi-label BCE, seed42. Fixed threshold 0.30.", "",
        "## 1. Data and retained label cardinality", "", "See DATASET_AUDIT.md for per-label support and the exact canonical IDs in DATASET_MANIFEST.json.", ""]
    for split, s in manifest["splits"].items():
        lines.append(f"- {split}: {s['retained_count']} retained; {s['single_emotion']} single-emotion; {s['multi_emotion']} multi-emotion; {s['removed_neutral_only']} neutral-only excluded.")
    lines += ["", "## 2. Training completion", "", f"Completed {env['completed_epochs']} epochs and {env['completed_optimizer_steps']} optimizer steps on {env['gpu']}.",
        "", "## 3. Checkpoint", "", f"Epoch {selection['epoch']}: validation Macro-F1 {selection['validation_macro_f1']:.8f}; full validation reload verified before the only formal test.",
        "", "## 4. Formal test metrics", "", "| Metric | Value |", "|---|---:|"]
    for k in ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "rare_f1"]:
        lines.append(f"| {k} | {metrics[k]:.8f} |")
    lines += ["", "## 5. Best and worst emotion labels", "", "Descriptive test results only; no tuning follows.", "",
        "| Group | Emotion | Support | F1 |", "|---|---|---:|---:|"]
    for group, subset in [("Best five", ranked[:5]), ("Worst five", ranked[-5:])]:
        for r in subset:
            lines.append(f"| {group} | {r['label']} | {r['support']} | {float(r['f1']):.8f} |")
    lines += ["", "## 6. Fixed seven rare emotions", "", "| Emotion | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for n in CONFIG["rare_labels"]:
        r = next(r for r in rows if r["label"] == n)
        lines.append(f"| {n} | {r['support']} | {float(r['precision']):.8f} | {float(r['recall']):.8f} | {float(r['f1']):.8f} |")
    lines += ["", "## 7. Differences from historical experiments", "",
        "Earlier Exp5A/Exp6B used BERT-base-cased, all 28 outputs and the complete official splits including neutral-only. "
        "A0 uses RoBERTa-base, 27 emotion outputs and official split filtering. Exp6B also added Formula A auxiliary loss; A0 uses unweighted mean BCE only. "
        "The four-epoch, batch16, seed42, AdamW/ceil-warmup/clipping and fixed0.30 validation-selection budget is retained. "
        "No 27-vs-28 Macro-F1 subtraction is a controlled improvement estimate. Original single-label HypEmo Weighted-F1 66.3% is not comparable.",
        "", "## 8. Suitability as the later matched baseline", "",
        "All verification gates passed. This is a reusable single-seed baseline, not a multi-seed statistical conclusion. "
        "Preserve package hashes, model snapshot, dataset IDs, logits, classifier path and exact training/evaluation protocol for A1/A2.",
        "", "## 9. Loss, predictions and gradients", "",
        "All logged batch losses and clipping norms are finite; nonzero finite encoder/head gradients were checked in smoke and each epoch's first batch. "
        "Every step used error_if_nonfinite gradient clipping. These checks establish numerical validity, not the absence of all modeling limitations.",
        f"Total predicted positives: {metrics['total_predicted_positives']}; gold positives: {metrics['total_gold_positives']}; "
        f"mean predictions/example: {metrics['mean_predictions_per_example']:.6f}; zero-prediction examples: {metrics['zero_prediction_examples']}; "
        f"zero-prediction labels: {', '.join(metrics['zero_prediction_labels']) or 'none'}. Per-label TP/FP/FN and prediction counts are archived.",
        "", "## 10. Next-stage classification-path contract", "",
        "A1/A2 must reuse model27.Roberta27.encoder and its complete first-token/dropout/classifier forward unchanged, "
        "the original initialization RNG convention, same snapshot/tokenizer, 27-label mapping/retained IDs, seed, loader, optimizer and validation rule. "
        "Any auxiliary component must be a separately controlled addition; do not replace the head with RobertaForSequenceClassification. "
        "The existing FYP independent-per-label dropout layout is documented in IMPLEMENTATION_AUDIT.md. "
        "Freeze baseline path initialization before any future auxiliary-module RNG draws. No A1/A2 code is executed in this experiment.",
        "", "Supplementary neutral-only diagnostic: skipped (optional). Threshold search: not performed. Stop after Exp6C-0."]
    (root / "EXP6C0_SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "train", "verify"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "smoke":
            smoke(args.run_root, args.snapshot)
        elif args.phase == "train":
            require(args.snapshot is None, "Formal run uses the immutable smoke plan snapshot")
            train(args.run_root)
        else:
            require(args.snapshot is None, "Verification reads the immutable plan")
            verify(args.run_root)
    except BaseException as e:
        # Never overwrite a completed proof or preexisting run on a rejected retry.
        if (args.phase != "verify" and args.run_root.exists()
                and not (args.run_root / "SUCCESS.json").exists()
                and not (args.run_root / "FAILURE.json").exists()):
            write(args.run_root / "FAILURE.json", dict(phase=args.phase, time=now(), reason=repr(e), traceback=traceback.format_exc()))
        raise
