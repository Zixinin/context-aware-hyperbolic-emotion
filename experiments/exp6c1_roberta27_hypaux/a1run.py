"""Explicit smoke/full A1 phases; exactly one seed and one formal test call."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from a1lib import (HERE, A0_RUN, CONFIG, RESULTS, EXP6B, a0, read, write, require, now, sha,
                   check_sources, baseline_evidence, protected_inventory, runtime, state_hash,
                   run_path, seal, check_seal, verify_upstream, load_reference_loss)
from a1model import (build, loss_components, parameter_groups, module_gradients,
                     diagnostic_values, train_step)
from data27 import loader, targets, save_audit
from model27 import load_model as load_a0_model, seed_everything, optimizer_for as a0_optimizer
from metrics27 import score, probabilities, save_predictions, write_csv
from run import compare_reload, identity as a0_model_identity

SUCCESS = "EXP6C1_ROBERTA27_HYPAUX_VERIFIED"


def stage(root, name, **extra):
    write(root / "state.json", dict(stage=name, time=now(), **extra))
    print("STAGE", name, flush=True)


def release():
    gc.collect()
    torch.cuda.empty_cache()


def initial_parity(snapshot, names, data, root):
    seed_everything()
    control, tokenizer, info = load_a0_model(snapshot)
    identity = a0_model_identity(snapshot, control, tokenizer, info)
    require(identity == read(A0_RUN / "model_identity.json"), "Loaded raw RoBERTa/tokenizer identity differs from A0")
    write(root / "MODEL_IDENTITY.json", identity)
    historical = torch.load(A0_RUN / "smoke/smoke_weights.pt", map_location="cpu", weights_only=True)
    current = control.state_dict()
    require(set(current) == set(historical), "A0 historical initialization state keys differ")
    for name, tensor in current.items():
        require(torch.equal(tensor, historical[name]), f"Fresh shared initialization differs from A0 historical smoke state: {name}")
    shared_hash = state_hash(current)
    del historical, current
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    control = control.to("cuda")
    sample = next(iter(loader(data["train"], tokenizer, shuffle=True)))
    sample_ids = sample.pop("ids")
    require(sample_ids == read(A0_RUN / "epoch1_first_batch.json")["ids"], "Initial DataLoader order differs from historical A0")
    x = {k: v.to("cuda") for k, v in sample.items()}
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        control.train()
        train_logits = control(**x)["logits"].cpu()
        control.eval()
        eval_logits = control(**x)["logits"].cpu()
    optimizer, shared_groups = a0_optimizer(control)
    del control, optimizer
    release()
    seed_everything()
    model, tokenizer, loading, identity = build(snapshot, names)
    require(state_hash(model.control.state_dict()) == shared_hash, "A1 shared tensor initialization differs")
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state()), "A1 post-constructor RNG differs from A0")
    model.to("cuda")
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.train()
        train_a1 = model(**x)["logits"].cpu()
        model.eval()
        eval_a1 = model(**x)["logits"].cpu()
    require(torch.equal(train_logits, train_a1) and torch.equal(eval_logits, eval_a1), "Initial A0/A1 classification logits are not exactly identical")
    optimizer, all_groups = parameter_groups(model)
    for old, new in zip(shared_groups, all_groups):
        shared = [n.removeprefix("control.") for n in new["parameter_names"] if n.startswith("control.")]
        require(shared == old["parameter_names"] and new["weight_decay"] == old["weight_decay"], "Shared optimizer groups differ from A0")
    proof = dict(shared_state_sha256=shared_hash, hyperbolic_initial_state_sha256=state_hash(model.hyper.state_dict()),
        raw_model_and_tokenizer_identity_exact_equal=True,
        historical_A0_smoke_weights_sha256=sha(A0_RUN / "smoke/smoke_weights.pt"),
        historical_anchor_note="A0 smoke used its first scheduler step at lr=0; all saved smoke tensors independently equal a freshly seeded raw A0 model. No trained A0 weights initialize A1.",
        tensors_compared=len(model.control.state_dict()), shared_parameters_exact_equal=True,
        initial_training_logits_exact_equal=True, initial_eval_logits_exact_equal=True,
        initial_train_logits_sha256=__import__("hashlib").sha256(train_logits.numpy().tobytes()).hexdigest(),
        max_abs_training_logit_error=float((train_logits - train_a1).abs().max()),
        max_abs_eval_logit_error=float((eval_logits - eval_a1).abs().max()),
        post_constructor_cpu_and_cuda_rng_exact_equal=True, shared_optimizer_groups_identical=True,
        first_canonical_training_batch_ids=sample_ids, hyperbolic_identity=identity,
        rng_scope="Construction is RNG-isolated. Original HNN dropout consumes extra RNG during training; later common dropout masks need not match A0.",
        weights_from_A0_best_checkpoint=False, weights_from_A1_smoke=False)
    write(root / "INITIALIZATION_PARITY.json", proof)
    del model, optimizer, x, sample
    release()
    return proof


def formula_and_gradient_case(model, batch):
    ids = batch.pop("ids")
    x = {k: v.to("cuda") for k, v in batch.items()}
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(**x)
    parts = out["components"]
    # The old function's auxiliary component is independent of its 28-label
    # distance-weighted BCE. Append a zero neutral target only for this parity test.
    reference = load_reference_loss()
    old = reference(torch.cat([out["logits"], torch.zeros(len(ids), 1, device="cuda")], 1),
                    torch.cat([x["labels"], torch.zeros(len(ids), 1, device="cuda")], 1),
                    out["z"], out["decoder_logits"], model.emotion_embeddings)
    torch.testing.assert_close(parts["auxiliary"], old["auxiliary"], rtol=0, atol=0)
    for variable in [out["z"], out["decoder_logits"]]:
        new_g = torch.autograd.grad(parts["auxiliary"], variable, retain_graph=True)[0]
        old_g = torch.autograd.grad(old["auxiliary"], variable, retain_graph=True)[0]
        torch.testing.assert_close(new_g, old_g, rtol=0, atol=0)
    params = list(model.named_parameters())
    gb = torch.autograd.grad(parts["classification"], [p for _, p in params], retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(.1 * parts["auxiliary"], [p for _, p in params], retain_graph=True, allow_unused=True)
    def norm(gs, prefix):
        selected = [g for (n, _), g in zip(params, gs) if n.startswith(prefix) and g is not None]
        require(all(bool(torch.isfinite(g).all()) for g in selected), "Nonfinite diagnostic component gradients")
        return sum(float(g.detach().double().square().sum()) for g in selected) ** .5
    bce_encoder, aux_encoder = norm(gb, "control.encoder."), norm(ga, "control.encoder.")
    require(bce_encoder > 0 and aux_encoder > 0, "Shared encoder lacks BCE/auxiliary gradient")
    require(norm(gb, "hyper.") == 0 and norm(ga, "control.classifier_") == 0, "Wrong component gradient path")
    require(norm(ga, "hyper.encoder.") > 0 and norm(ga, "hyper.decoder.") > 0, "Auxiliary components lack gradient")
    torch.testing.assert_close(parts["classification"], torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], x["labels"]), rtol=0, atol=0)
    out["loss"].backward()
    gradients = module_gradients(model)
    require(model.emotion_embeddings.grad is None, "Fixed label vectors are not fixed")
    values = diagnostic_values(out)
    model.zero_grad(set_to_none=True)
    return dict(ids=ids, positive_counts=parts["positive_count"].detach().cpu().tolist(),
                values=values, gradients=gradients, original_FormulaA_aux_value_and_gradients_exact_equal=True,
                BCE_unweighted_exact=True, labels_fixed=True,
                component_gradient_paths=dict(bce_shared_encoder_norm=bce_encoder,
                    weighted_aux_shared_encoder_norm=aux_encoder, weighted_aux_to_bce_encoder_gradient_norm=aux_encoder / bce_encoder,
                    bce_to_hyperbolic_norm=0, auxiliary_to_classifier_norm=0))


@torch.no_grad()
def predict(model, batches, description):
    model.eval()
    logs, gold, ids = [], [], []
    totals, count, max_ball_norm = {}, 0, 0.
    for index, batch in enumerate(batches, 1):
        record_ids = batch.pop("ids")
        x = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        output = model(**x)
        values = diagnostic_values(output)
        max_ball_norm = max(max_ball_norm, values["max_ball_norm"])
        size = len(record_ids)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + value * size
        logs.append(output["logits"].cpu().numpy())
        gold.append(x["labels"].cpu().numpy())
        ids.extend(record_ids)
        count += size
        if index % 25 == 0:
            print(description, "batches", index, "examples", count, flush=True)
    require(count > 0, "Empty evaluation split")
    diagnostics = {k: v / count for k, v in totals.items()}
    diagnostics["max_ball_norm"] = max_ball_norm
    diagnostics["mean_batch_weighted_auxiliary_to_bce"] = diagnostics["weighted_auxiliary_to_bce"]
    diagnostics["mean_batch_absolute_weighted_auxiliary_to_bce"] = diagnostics["absolute_weighted_auxiliary_to_bce"]
    diagnostics["weighted_auxiliary_to_bce"] = diagnostics["weighted_auxiliary"] / diagnostics["bce"]
    diagnostics["absolute_weighted_auxiliary_to_bce"] = abs(diagnostics["weighted_auxiliary"]) / diagnostics["bce"]
    require(all(not m.training for m in model.modules()), "Evaluation did not disable all dropout")
    return np.concatenate(logs), np.concatenate(gold), np.asarray(ids, dtype=str), diagnostics["loss"], diagnostics


def smoke(root):
    from transformers import get_linear_schedule_with_warmup
    root = run_path(root)
    stage(root, "verify_frozen_A0_and_FormulaA_foundation")
    source, env = check_sources(), runtime()
    data, manifest, baseline = baseline_evidence()
    upstream = verify_upstream()
    protected = protected_inventory()
    write(root / "protected_before.json", protected)
    old_plan = read(A0_RUN / "plan.json")
    snapshot = Path(old_plan["snapshot"])
    require(a0.snapshot_inventory(snapshot) == old_plan["snapshot_files_sha256"], "A0 RoBERTa snapshot changed")
    write(root / "config.json", CONFIG)
    save_audit(root, manifest)
    write(root / "A0_BASELINE_EVIDENCE.json", baseline)
    shutil.copytree(HERE, root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(HERE / "LOSS_SPECIFICATION.md", root / "LOSS_SPECIFICATION.md")
    plan = dict(config=CONFIG, source_sha256=source, runtime=env, baseline=baseline,
                snapshot=str(snapshot), snapshot_files_sha256=old_plan["snapshot_files_sha256"],
                upstream=upstream, created=now())
    write(root / "plan.json", plan)
    directory = root / "smoke"
    directory.mkdir(exist_ok=False)
    stage(root, "actual_shared_initialization_and_logits_parity")
    parity = initial_parity(snapshot, manifest["names27"], data, root)
    seed_everything()
    model, tokenizer, _, hyper_identity = build(snapshot, manifest["names27"])
    require(state_hash(model.hyper.state_dict()) == parity["hyperbolic_initial_state_sha256"], "Auxiliary initialization differs across fresh builds")
    model.to("cuda")
    np.save(root / "fixed_emotion_embeddings.npy", model.emotion_embeddings.detach().cpu().numpy())
    write(root / "hyperbolic_identity.json", hyper_identity)
    stage(root, "actual_multi_positive_loss_scale_and_gradient_paths")
    neutral = manifest["neutral_original_index"]
    cases = {
        "single_emotion": [r for r in data["train"] if len(r["labels"]) == 1][:16],
        "multiple_positive_emotions": [r for r in data["train"] if len(r["labels"]) > 1][:16],
        "original_neutral_plus_emotion_retained": [r for r in data["train"] if neutral in r["original_labels"]][:16]}
    reports = {}
    for name, rows in cases.items():
        require(len(rows) == 16, "Insufficient canonical smoke cases")
        reports[name] = formula_and_gradient_case(model, next(iter(loader(rows, tokenizer))))
        print("SMOKE_CASE", name, json.dumps(reports[name]["values"]), flush=True)
    write(directory / "LOSS_AND_GRADIENT_CASES.json", reports)
    del model
    release()
    stage(root, "two_actual_optimizer_steps_and_checkpoint_reload")
    seed_everything()
    model, tokenizer, _, _ = build(snapshot, manifest["names27"])
    require(state_hash(model.control.state_dict()) == parity["shared_state_sha256"], "Smoke shared initialization drift")
    require(state_hash(model.hyper.state_dict()) == parity["hyperbolic_initial_state_sha256"], "Smoke auxiliary initialization drift")
    model.to("cuda").train()
    optimizer, groups = parameter_groups(model)
    scheduler = get_linear_schedule_with_warmup(optimizer, 765, 7648)
    initial_hyp = state_hash(model.hyper.state_dict())
    iterations = []
    training = iter(loader(data["train"], tokenizer, shuffle=True))
    for i in range(2):
        lr_used = optimizer.param_groups[0]["lr"]
        values = train_step(model, next(training), optimizer, scheduler, detailed=True)
        iterations.append(dict(values, lr_used=lr_used, lr_after=optimizer.param_groups[0]["lr"]))
    require(iterations[0]["lr_used"] == 0 and iterations[1]["lr_used"] > 0 and state_hash(model.hyper.state_dict()) != initial_hyp, "Hyperbolic optimizer did not update")
    validation = predict(model, loader(data["validation"][:64], tokenizer), "smoke validation")
    torch.save(model.state_dict(), directory / "smoke_weights.pt")
    del model, optimizer, scheduler
    release()
    seed_everything()
    reloaded, tokenizer, _, _ = build(snapshot, manifest["names27"])
    reloaded.load_state_dict(torch.load(directory / "smoke_weights.pt", map_location="cpu", weights_only=True), strict=True)
    reloaded.to("cuda")
    measured = predict(reloaded, loader(data["validation"][:64], tokenizer), "smoke reload")
    reload_proof = compare_reload(validation, measured, manifest["names27"])
    for key in validation[4]:
        np.testing.assert_allclose(validation[4][key], measured[4][key], atol=1e-6, rtol=1e-6)
    del reloaded
    release()
    require(protected_inventory() == protected, "Frozen A0/earlier research files changed during smoke")
    require(check_sources() == source, "A1/A0 source changed during smoke")
    require(a0.snapshot_inventory(snapshot) == plan["snapshot_files_sha256"], "RoBERTa snapshot changed")
    write(directory / "SMOKE_REPORT.json", dict(status="PASS", actual_gpu=True, formal_epochs=0,
        test_inference_calls=0, initialization=parity, loss_gradient_cases=reports,
        optimizer_iterations=iterations, optimizer_groups=groups, reload=reload_proof,
        protected_file_count=len(protected), old_research_files_unchanged=True, completed=now()))
    seal(directory, "SMOKE_VERIFIED.json", status="EXP6C1_SMOKE_VERIFIED", plan_sha256=sha(root / "plan.json"), source_sha256=source)
    stage(root, "smoke_verified_formal_training_pending")
    print("EXP6C1_SMOKE_VERIFIED", root, flush=True)
    print("NEXT_AFTER_REVIEW: bash", HERE / "launch.sh", "train", root, flush=True)


def assert_plan(root):
    root = run_path(root)
    plan = read(root / "plan.json")
    require(plan["config"] == CONFIG and plan["source_sha256"] == check_sources(), "A1 config/source changed since smoke")
    current = runtime()
    for k in ["host", "python", "executable", "packages", "pip_freeze", "gpu", "cuda_runtime"]:
        require(current[k] == plan["runtime"][k], "Runtime changed since smoke")
    data, manifest, evidence = baseline_evidence()
    require(evidence == plan["baseline"], "A0 baseline changed since smoke")
    require(manifest == read(root / "dataset_manifest.json"), "Dataset changed since smoke")
    require(protected_inventory() == read(root / "protected_before.json"), "Old research files changed since smoke")
    require(verify_upstream() == plan["upstream"], "Upstream changed since smoke")
    snapshot = Path(plan["snapshot"])
    require(a0.snapshot_inventory(snapshot) == plan["snapshot_files_sha256"], "A0 snapshot changed")
    proof = check_seal(root / "smoke", "SMOKE_VERIFIED.json")
    require(proof["status"] == "EXP6C1_SMOKE_VERIFIED" and proof["source_sha256"] == plan["source_sha256"]
            and proof["plan_sha256"] == sha(root / "plan.json"), "Smoke proof does not match formal plan")
    require((root / "smoke_exit_code.txt").read_text().strip() == "0", "Smoke exit code is not zero")
    return data, manifest, plan, snapshot


def train(root):
    from transformers import get_linear_schedule_with_warmup, AutoTokenizer
    root = run_path(root)
    data, manifest, plan, snapshot = assert_plan(root)
    require(not (root / "FAILURE.json").exists(), "Prior failure requires review; no automatic retry")
    with (root / "FORMAL_TRAINING_CLAIM.json").open("x") as f:
        json.dump(dict(seed=42, epochs=4, started=now(), retries=False), f)
    env = dict(plan["runtime"], status="training", start_time=now(), completed_epochs=0,
               completed_optimizer_steps=0, formal_test_inference_calls=0)
    write(root / "environment.json", env)
    stage(root, "formal_seed42_fresh_A0_shared_initialization")
    seed_everything()
    model, tokenizer, _, hyper = build(snapshot, manifest["names27"])
    parity = read(root / "INITIALIZATION_PARITY.json")
    require(state_hash(model.control.state_dict()) == parity["shared_state_sha256"], "Formal shared state differs from A0/smoke")
    require(state_hash(model.hyper.state_dict()) == parity["hyperbolic_initial_state_sha256"], "Formal auxiliary state differs from verified smoke initialization")
    model.to("cuda")
    require(hyper == read(root / "hyperbolic_identity.json"), "Hyperbolic profile changed")
    require(np.array_equal(model.emotion_embeddings.detach().cpu(), np.load(root / "fixed_emotion_embeddings.npy", allow_pickle=False)), "Fixed labels changed")
    optimizer, groups = parameter_groups(model)
    training = loader(data["train"], tokenizer, shuffle=True)
    validation = loader(data["validation"], tokenizer)
    total_steps, warmup = len(training) * 4, math.ceil(len(training) * 4 * .1)
    require(total_steps == 7648 and warmup == 765, "A0/A1 filtered training budgets differ")
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    write(root / "optimizer_protocol.json", dict(groups=groups, total_steps=total_steps, warmup_steps=warmup,
        optimizer="unchanged A0 optimizer_for over all A1 trainable parameters", max_grad_norm=1.0,
        gradient_normalization="(mean_total_loss*batch_size).backward; grads/=batch_size; clip_all_1; AdamW; scheduler"))
    env.update(total_optimizer_steps=total_steps, warmup_steps=warmup)
    write(root / "environment.json", env)
    checkpoint = root / "best_checkpoint"
    checkpoint.mkdir(exist_ok=False)
    history, best, step = [], -1., 0
    fields = ["epoch", "batch", "step", "batch_size", "loss", "bce", "auxiliary", "weighted_auxiliary", "weighted_auxiliary_to_bce",
              "absolute_weighted_auxiliary_to_bce", "positive_ce_mean", "positive_dot_mean", "max_ball_norm", "unclipped_grad_norm", "lr_used", "lr_after"]
    with (root / "batch_log.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 5):
            stage(root, f"training_epoch_{epoch}")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums, count, order, max_norm = {}, 0, [], 0.
            for index, batch in enumerate(training, 1):
                lr = optimizer.param_groups[0]["lr"]
                row = train_step(model, batch, optimizer, scheduler, detailed=index == 1)
                if index == 1:
                    require(row["ids"] == read(A0_RUN / f"epoch{epoch}_first_batch.json")["ids"], "A0/A1 epoch-first batch order differs")
                    write(root / f"epoch{epoch}_first_batch.json", row)
                step += 1
                size = row["batch_size"]
                count += size
                order.extend(row["ids"])
                max_norm = max(max_norm, row["max_ball_norm"])
                for k in ["loss", "bce", "auxiliary", "weighted_auxiliary", "positive_ce_mean", "positive_dot_mean"]:
                    sums[k] = sums.get(k, 0.) + row[k] * size
                writer.writerow(dict(epoch=epoch, batch=index, step=step, lr_used=lr,
                                     lr_after=optimizer.param_groups[0]["lr"], **{k: row[k] for k in fields if k in row}))
                if index % 50 == 0 or index == len(training):
                    f.flush()
                    print(f"A1 seed42 epoch={epoch}/4 batch={index}/{len(training)} step={step}/{total_steps} BCE={row['bce']:.6f} aux={row['auxiliary']:.6f} total={row['loss']:.6f}", flush=True)
            require(count == 30587 and len(set(order)) == count and set(order) == {r["id"] for r in data["train"]}, "Training omitted/repeated examples")
            write(root / f"train_order_epoch{epoch}.json", order)
            val = predict(model, validation, f"epoch{epoch} validation")
            values = score(val[1], probabilities(val[0]), manifest["names27"])[0]
            entry = dict(epoch=epoch, global_step=step, training_examples=count,
                         **{"training_" + k: v / count for k, v in sums.items()},
                         max_training_ball_norm=max_norm, validation_loss=val[3],
                         validation_bce=val[4]["bce"], validation_auxiliary=val[4]["auxiliary"],
                         validation_macro_f1=values["macro_f1"], validation_micro_f1=values["micro_f1"],
                         learning_rate=optimizer.param_groups[0]["lr"])
            history.append(entry)
            write_csv(root / "training_log.csv", history)
            write(root / f"epoch{epoch}_validation_metrics.json", values)
            write(root / f"epoch{epoch}_validation_loss_diagnostics.json", val[4])
            print("EPOCH_RESULT", entry, flush=True)
            if values["macro_f1"] > best:
                best = values["macro_f1"]
                torch.save(model.state_dict(), checkpoint / "model.pt")
                tokenizer.save_pretrained(checkpoint / "tokenizer")
                model.control.encoder.config.save_pretrained(checkpoint / "encoder_config")
                save_predictions(root, "validation_selected", *val[:3], manifest["names27"])
                write(root / "validation_selected_loss_diagnostics.json", val[4])
                write(root / "checkpoint_selection.json", dict(epoch=epoch, global_step=step,
                    validation_macro_f1=best, threshold=.30, selection_split="validation",
                    rule=a0.CONFIG["checkpoint_rule"], tie_break="earliest via strict greater-than",
                    checkpoint_sha256=sha(checkpoint / "model.pt")))
            env.update(completed_epochs=epoch, completed_optimizer_steps=step)
            write(root / "environment.json", env)
    require(step == 7648 and scheduler.get_last_lr()[0] == 0, "Formal training did not complete scheduled budget")
    del model, optimizer, scheduler, val
    release()
    stage(root, "full_validation_reload_of_best_A1_checkpoint")
    seed_everything()
    model, tokenizer, _, _ = build(snapshot, manifest["names27"])
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True), strict=True)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True, use_fast=True)
    reloaded = predict(model, loader(data["validation"], tokenizer), "full validation reload")
    with np.load(root / "validation_selected_predictions.npz", allow_pickle=False) as z:
        previous = z["logits"], z["targets"], z["ids"]
    reload_proof = compare_reload(previous, reloaded, manifest["names27"])
    require(reload_proof["num_examples"] == 3834, "Full validation reload missing")
    selected_diagnostics = read(root / "validation_selected_loss_diagnostics.json")
    for k, value in reloaded[4].items():
        np.testing.assert_allclose(value, selected_diagnostics[k], atol=1e-6, rtol=1e-6)
    write(root / "checkpoint_reload_verification.json", reload_proof)
    _, val_rare = save_predictions(root, "validation", *reloaded[:3], manifest["names27"])
    write(root / "validation_loss_diagnostics.json", reloaded[4])
    with (root / "FORMAL_TEST_CLAIM.json").open("x") as f:
        json.dump(dict(call=1, started=now(), threshold=.30, checkpoint_sha256=sha(checkpoint / "model.pt")), f)
    env.update(status="formal_test_started", formal_test_inference_calls=1)
    write(root / "environment.json", env)
    stage(root, "single_formal_A1_test")
    tested = predict(model, loader(data["test"], tokenizer), "formal test (one call)")
    test_metrics, test_rare = save_predictions(root, "test", *tested[:3], manifest["names27"])
    write(root / "test_loss_diagnostics.json", tested[4])
    write(root / "rare_label_metrics.json", dict(validation=val_rare, test=test_rare))
    del model
    release()
    require(check_sources() == plan["source_sha256"], "A1/A0 source changed")
    require(protected_inventory() == read(root / "protected_before.json"), "A0 or earlier research artifacts changed")
    require(a0.snapshot_inventory(snapshot) == plan["snapshot_files_sha256"], "Raw snapshot changed")
    env.update(status="training_and_single_test_completed", end_time=now(), old_research_files_unchanged=True)
    write(root / "environment.json", env)
    stage(root, "CPU_saved_prediction_verification_and_matched_comparison")
    verify(root, sealed=False)
    summarize(root, manifest, test_metrics)
    (root / "VERIFICATION_REPORT.md").write_text(
        "# Exp6C-1 verification\n\nPASS: frozen A0 code/checkpoint/results; identical canonical IDs/27 labels/snapshot/tokenizer; "
        "historical A0/fresh shared initial tensors and initial train/eval logits; Formula A aux value/gradient equivalence; "
        "ordinary unweighted BCE; ball/finite/nonzero gradients; optimizer/scheduler and four epochs; "
        "validation-only earliest-tie selection; entire validation reload; single formal Test; "
        "saved-prediction aggregate/per-label/Rare-F1 recomputation and paired A0 comparison.\n\n"
        "No extra seeds, distance weighting, new hierarchy, ASL, Graph or test tuning.\n")
    env["status"] = "COMPLETE"
    write(root / "environment.json", env)
    stage(root, "COMPLETE")
    seal(root, "SUCCESS.json", status=SUCCESS, seed=42, epochs=4, test_inference_calls=1, source_sha256=plan["source_sha256"], completed=now())
    check_seal(root, "SUCCESS.json")
    print("=" * 50, SUCCESS, "Backbone: roberta-base", "Task: 27-label multi-label", "Loss: BCE + 0.1 * FormulaA_aux", "Seed: 42", "Status: COMPLETE", "=" * 50, sep="\n", flush=True)


def verify(root, sealed=True):
    root = run_path(root)
    if sealed:
        require(check_seal(root, "SUCCESS.json")["status"] == SUCCESS, "A1 success proof missing")
    plan = read(root / "plan.json")
    require(plan["source_sha256"] == check_sources() and read(root / "config.json") == CONFIG, "A1 source/config differs")
    data, manifest, baseline = baseline_evidence()
    require(baseline == plan["baseline"] and manifest == read(root / "dataset_manifest.json"), "A0/data changed")
    require(protected_inventory() == read(root / "protected_before.json"), "Protected research files differ")
    require(a0.snapshot_inventory(Path(plan["snapshot"])) == plan["snapshot_files_sha256"], "Snapshot differs")
    env = read(root / "environment.json")
    require(env["completed_epochs"] == 4 and env["completed_optimizer_steps"] == 7648 and env["formal_test_inference_calls"] == 1, "A1 formal runtime incomplete")
    with (root / "training_log.csv").open() as f:
        history = list(csv.DictReader(f))
    require([int(r["epoch"]) for r in history] == [1, 2, 3, 4] and [int(r["global_step"]) for r in history] == [1912, 3824, 5736, 7648], "A1 training history incomplete")
    require(all(int(r["training_examples"]) == 30587 for r in history), "A1 train counts differ")
    selection = read(root / "checkpoint_selection.json")
    chosen = max(history, key=lambda r: float(r["validation_macro_f1"]))
    require(selection["epoch"] == int(chosen["epoch"]) and selection["global_step"] == int(chosen["global_step"])
            and selection["validation_macro_f1"] == float(chosen["validation_macro_f1"]) and selection["threshold"] == .30
            and selection["selection_split"] == "validation", "Wrong A1 selected checkpoint")
    require(selection["checkpoint_sha256"] == sha(root / "best_checkpoint/model.pt"), "A1 checkpoint changed")
    opt = read(root / "optimizer_protocol.json")
    require(opt["total_steps"] == 7648 and opt["warmup_steps"] == 765 and opt["max_grad_norm"] == 1, "A1 optimizer protocol differs")
    with (root / "batch_log.csv").open() as f:
        batches = list(csv.DictReader(f))
    require(len(batches) == 7648 and [int(r["step"]) for r in batches] == list(range(1, 7649)), "A1 step log incomplete")
    for row in batches:
        require(all(np.isfinite(float(row[k])) for k in row), "Nonfinite batch log")
        require(float(row["max_ball_norm"]) < 1 and float(row["bce"]) > 0, "Invalid ball/BCE in batch log")
        require(abs(float(row["loss"]) - float(row["bce"]) - .1 * float(row["auxiliary"])) < 2e-6, "Total loss is not BCE+0.1aux")
    require(float(batches[-1]["lr_after"]) == 0, "Scheduler did not finish")
    for epoch in range(1, 5):
        order = read(root / f"train_order_epoch{epoch}.json")
        require(len(order) == len(set(order)) == 30587 and set(order) == {r["id"] for r in data["train"]}, "Bad training order")
        report = read(root / f"epoch{epoch}_first_batch.json")
        require(set(report["gradients"]) == {"shared_encoder", "classifier", "hyperbolic_projection", "auxiliary_decoder"}, "Missing gradient group")
        require(all(g["norm"] > 0 and np.isfinite(g["norm"]) for g in report["gradients"].values()), "Invalid gradients")
    for split in ["validation", "test"]:
        with np.load(root / f"{split}_predictions.npz", allow_pickle=False) as z:
            require(np.array_equal(z["ids"], [r["id"] for r in data[split]]) and np.array_equal(z["targets"], targets(data[split])), "Prediction canonical IDs/gold differ")
            require(np.array_equal(z["label_names"], manifest["names27"]), "Not the same 27 label order")
            probs = probabilities(z["logits"])
            values, rows, rare, pred = score(z["targets"], probs, manifest["names27"])
            require(np.array_equal(z["probabilities"], probs) and np.array_equal(z["predictions"], pred), "Saved probability/prediction mismatch")
            require(np.array_equal(z["predicted_positive_counts"], pred.sum(1)) and np.array_equal(z["true_positive_counts"], z["targets"].sum(1)), "Saved positive counts differ")
        require(values == read(root / f"{split}_metrics.json") and rare == read(root / "rare_label_metrics.json")[split], "Metrics/rare recomputation differ")
        with (root / f"{split}_per_label_metrics.csv").open() as f:
            recorded = list(csv.DictReader(f))
        require(len(recorded) == 27, "Expected 27 per-label rows")
        for a, b in zip(recorded, rows):
            require(set(a) == set(b), "Per-label schema differs")
            for k, value in b.items():
                require(a[k] == value if isinstance(value, str) else abs(float(a[k]) - value) < 1e-12, "Per-label metric differs")
        manual = [2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"]) if 2 * r["tp"] + r["fp"] + r["fn"] else 0 for r in rows]
        np.testing.assert_allclose(manual, [r["f1"] for r in rows], rtol=0, atol=1e-15)
        if split == "validation":
            require(values["macro_f1"] == selection["validation_macro_f1"], "Reloaded validation selection score differs")
        diagnostics = read(root / f"{split}_loss_diagnostics.json")
        require(all(np.isfinite(v) for v in diagnostics.values()) and abs(diagnostics["loss"] - diagnostics["bce"] - .1 * diagnostics["auxiliary"]) < 2e-6, "Evaluation loss decomposition differs")
    reload = read(root / "checkpoint_reload_verification.json")
    require(reload["num_examples"] == 3834 and all(reload[k] for k in ["ids_equal", "targets_equal", "metrics_equal", "threshold_predictions_equal"]), "Full A1 validation reload missing")
    claim = read(root / "FORMAL_TEST_CLAIM.json")
    require(claim["call"] == 1 and claim["threshold"] == .30 and claim["checkpoint_sha256"] == selection["checkpoint_sha256"], "Single formal test claim invalid")
    print("EXP6C1_SAVED_ARTIFACT_VERIFICATION_PASS", root, flush=True)


def summarize(root, manifest, metrics):
    baseline = read(A0_RUN / "test_metrics.json")
    selection = read(root / "checkpoint_selection.json")
    comparison = [dict(metric=k, A0=baseline[k], A1=metrics[k], delta=metrics[k] - baseline[k],
                       delta_percentage_points=100 * (metrics[k] - baseline[k])) for k in
                  ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "rare_f1"]]
    write_csv(root / "A0_vs_A1_metrics.csv", comparison)
    def csv_rows(path):
        with path.open() as f:
            return list(csv.DictReader(f))
    old, new = csv_rows(A0_RUN / "test_per_label_metrics.csv"), csv_rows(root / "test_per_label_metrics.csv")
    rows = []
    for a, b in zip(old, new):
        require(a["label"] == b["label"] and a["support"] == b["support"], "A0/A1 label/support mismatch")
        rows.append(dict(label_id=int(b["label_id"]), label=b["label"], support=int(b["support"]),
                         **{"A0_" + k: float(a[k]) for k in ["precision", "recall", "f1"]},
                         **{"A1_" + k: float(b[k]) for k in ["precision", "recall", "f1"]},
                         **{"A0_" + k: int(a[k]) for k in ["tp", "fp", "fn"]},
                         **{"A1_" + k: int(b[k]) for k in ["tp", "fp", "fn"]},
                         delta_f1=float(b["f1"]) - float(a["f1"])))
    require(len(rows) == 27, "Comparison must contain exactly 27 emotions")
    write_csv(root / "A0_vs_A1_27_label_metrics.csv", rows)
    with np.load(A0_RUN / "test_predictions.npz", allow_pickle=False) as a, np.load(root / "test_predictions.npz", allow_pickle=False) as b:
        require(np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["targets"], b["targets"]), "Paired test ID/gold mismatch")
        counts = dict(examples=len(a["ids"]), examples_with_any_prediction_change=int((a["predictions"] != b["predictions"]).any(1).sum()),
                      A0_predicted_positives=int(a["predictions"].sum()), A1_predicted_positives=int(b["predictions"].sum()))
    write(root / "paired_prediction_diagnostics.json", counts)
    lines = ["# EXP6C1_SUMMARY", "", "A1: RoBERTa27 + ordinary BCE + 0.1 x validated Formula A auxiliary-only objective.", "",
        "The sole planned method addition is the original HNN/tangent decoder/fixed 27 label vectors and its multi-positive auxiliary supervision.",
        "A0 code, checkpoint and results are read-only; no A0 retraining or test inference was performed.", "",
        "## Matched protocol", "", "Same canonical 30587/3834/3821 IDs, 27-label order, raw RoBERTa snapshot/tokenizer, exact A0 forward path, "
        "seed42, max_length50, batch16, four epochs/7648 steps/765 ceil warmup, AdamW5e-5/WD0.01, clipping1, FP32/eager, fixed0.30 and validation Macro-F1/earliest tie.",
        "Shared initial tensors and initial train/eval classification logits match A0 exactly; see INITIALIZATION_PARITY.json. "
        "No A0 trained checkpoint or A1 smoke weights initialize formal training.", "",
        f"Best A1 epoch: {selection['epoch']}; validation Macro-F1={selection['validation_macro_f1']:.8f}. Full 3834-example reload passed before the only formal test.",
        "", "## Fixed0.30 Test comparison", "", "| Metric | A0 | A1 | Delta (percentage points) |", "|---|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['metric']} | {r['A0']*100:.4f}% | {r['A1']*100:.4f}% | {r['delta_percentage_points']:+.4f} |")
    delta = metrics["macro_f1"] - baseline["macro_f1"]
    lines += ["", "At this single seed, A1 " + ("increased" if delta > 0 else "decreased" if delta < 0 else "did not change") +
              f" matched test Macro-F1 by {delta*100:+.4f} percentage points. This does not establish a multi-seed or geometry-specific causal conclusion.",
              "", "## All 27 emotion labels", "", "| Emotion | Support | A0 F1 | A1 Precision | A1 Recall | A1 F1 | Delta pp |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['support']} | {r['A0_f1']*100:.2f}% | {r['A1_precision']*100:.2f}% | {r['A1_recall']*100:.2f}% | {r['A1_f1']*100:.2f}% | {r['delta_f1']*100:+.2f} |")
    lines += ["", "Rare-F1 uses the same seven named emotions as A0. Detailed rare TP/FP/FN/precision/recall/F1 appear in rare_label_metrics.json.",
              "", "## Auxiliary scale and gradient diagnostics", "", "See LOSS_SPECIFICATION.md for the exact q-positive mean/batch mean formula and paths. "
              "No distance weights, positive class weights, sample weights or alternative formula are used. "
              "LOSS_AND_GRADIENT_CASES.json in smoke verifies single/multiple/neutral-coincident retained examples, Formula A value and gradient equivalence, "
              "unweighted BCE, fixed label vectors, sphere-interior values, and BCE/auxiliary-to-shared-encoder gradients.", "",
              "Training epoch means and each batch's BCE, raw/weighted auxiliary, signed/absolute ratios, CE/dot decomposition, ball norm and clipping norm "
              "are archived in training_log.csv and batch_log.csv. First-batch gradients for all four module groups are saved each epoch.", "",
              "| Evaluation split | BCE | Raw auxiliary | Weighted auxiliary | Total |", "|---|---:|---:|---:|---:|"]
    for split in ["validation", "test"]:
        d = read(root / f"{split}_loss_diagnostics.json")
        lines.append(f"| {split} | {d['bce']:.8f} | {d['auxiliary']:.8f} | {d['weighted_auxiliary']:.8f} | {d['loss']:.8f} |")
    lines += ["", "All numerical checks passed; this is not a claim that the auxiliary dominates or benefits every class. "
              "Raw auxiliary values may be negative due to the dot term; they are not clamped, normalized or tuned. "
              "Original HNN dropout consumes additional RNG during training, so subsequent shared dropout masks need not match A0. "
              "This compares addition of the complete auxiliary branch/supervision; isolating hyperbolic geometry itself would require a separately authorized control.",
              "", "## Completion boundary", "", "No extra seed, distance weighting, ASL, graph, new hierarchy or test tuning. Stop after Exp6C-1. "
              "All four epochs, validation-only selection, complete validation reload, one test and saved-prediction verification passed."]
    (root / "EXP6C1_SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "train", "verify"])
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.phase == "smoke":
            smoke(args.run_root)
        elif args.phase == "train":
            train(args.run_root)
        else:
            verify(args.run_root)
    except BaseException as e:
        if (args.phase != "verify" and args.run_root.exists() and not (args.run_root / "SUCCESS.json").exists()
                and not (args.run_root / "FAILURE.json").exists()):
            write(args.run_root / "FAILURE.json", dict(phase=args.phase, time=now(), reason=repr(e), traceback=traceback.format_exc()))
        raise
