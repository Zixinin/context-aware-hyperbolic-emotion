"""Only seeds123/2026: frozen seed42 training loop, independent new results."""
from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from multi_common import (ASSET_MANIFEST, MODEL_DIR, REPO, check_sources, config_for, freeze,
                            load_all_verified, now, package_versions, read, require,
                            require_runtime, seal, sha, verify_seed42, write)
from control_model import ControlBert, parameter_report
from data import LABEL_NAMES, RARE_IDS, frequencies
from model import make_optimizer
from train import predict, save_split, seed_everything, split_loader, subset
from metrics import probabilities, score, tune_validation_threshold


def run(mode, representation, root, seed):
    require_runtime(True)
    require(os.environ.get("PYTHONHASHSEED") == str(seed), "Use launch.sh to set the actual seed's Python hash state")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Deterministic CuBLAS setting missing")
    root = Path(root).resolve()
    require(root.parent == (REPO / "results/exp5a_controls_multiseed").resolve() and root.name.startswith("run_"), "Wrong results root")
    source = check_sources()
    preflight = read(root / "preflight.json")
    require(preflight["status"] == "EXP5A_MULTISEED_PREFLIGHT_PASS", "Preflight missing")
    require(preflight["source_sha256"] == source, "Source changed after audit")
    require(preflight["pip_freeze"] == freeze(), "Environment changed after audit")
    require(preflight["dataset_manifest_sha256"] == sha(ASSET_MANIFEST), "Asset manifest changed")
    require(preflight["seed42_reference"] == verify_seed42(), "Frozen seed42 reference changed after audit")
    config = config_for(mode, representation, seed)
    if mode == "full":
        from multi_verify import verify_control
        smoke = root / f"smoke_{representation}_seed{seed}"
        verify_control(smoke)
        smoke_cfg = read(smoke / "config.json")
        require({k: v for k, v in config.items() if k not in {"mode", "epochs"}} ==
                {k: v for k, v in smoke_cfg.items() if k not in {"mode", "epochs"}}, "Smoke config differs")
        require((smoke / "pip-freeze.txt").read_text() == freeze(), "Environment changed since smoke")
    output_dir = root / (f"smoke_{representation}_seed{seed}" if mode == "smoke" else f"{representation}_seed{seed}")
    output_dir.mkdir(exist_ok=False)
    (output_dir / "pip-freeze.txt").write_text(freeze())
    metadata = dict(status="running", phase=mode, representation=representation, seed=seed,
                    start_time=now(), hostname=platform.node(), python_executable=sys.executable,
                    python=platform.python_version(), packages=package_versions(),
                    gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                    source_sha256=source, preflight_path=str(root / "preflight.json"),
                    preflight_sha256=sha(root / "preflight.json"), seed42_reference=preflight["seed42_reference"],
                    dataset_manifest_sha256=sha(ASSET_MANIFEST),
                    implementation_details=["raw CLS or PAD-masked mean; CLS/SEP included in mean",
                    "no pooler; FP32 deterministic eager; independent-per-label dropout matches C",
                    "CLS/Mean head and initial RNG match seedwise frozen initialization convention; BCE only"])
    write(output_dir / "config.json", config)
    write(output_dir / "environment.json", metadata)
    try:
        seed_everything(seed)
        datasets, _ = load_all_verified()
        training, validation = datasets["train"], datasets["validation"]
        if mode == "smoke":
            training = subset(training, config["smoke_train_samples"], seed)
            validation = subset(validation, config["smoke_validation_samples"], seed)
        write(output_dir / "sample_ids.json", {"train": [r["id"] for r in training], "validation": [r["id"] for r in validation]})
        write(output_dir / "dataset_statistics.json", dict(splits={k: len(v) for k, v in datasets.items()},
              num_labels=28, label_order=dict(enumerate(LABEL_NAMES)), train_label_counts=frequencies(datasets["train"]).tolist(), rare_ids=RARE_IDS))
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
        model = ControlBert.from_pretrained(str(MODEL_DIR), config).to("cuda")
        metadata["parameters_runtime"] = parameter_report(model)
        require(metadata["parameters_runtime"] == preflight["parameters"][representation], "Runtime parameter count differs")
        training_loader = split_loader(training, tokenizer, config, 16, shuffle=True)
        validation_loader = split_loader(validation, tokenizer, config, 64)
        optimizer = make_optimizer(model, config)
        # Frozen protocol: accumulation=1, ceil warmup, sample-normalized gradients.
        require(config["gradient_accumulation_steps"] == 1, "Unexpected gradient accumulation")
        total_steps = len(training_loader) * config["epochs"]
        metadata.update(total_optimizer_steps=total_steps, optimizer_steps_per_epoch=len(training_loader),
                        warmup_steps=math.ceil(total_steps * config["warmup_ratio"]))
        scheduler = get_linear_schedule_with_warmup(optimizer, metadata["warmup_steps"], total_steps)
        best_f1, step, history, validation_best = -1.0, 0, [], None
        checkpoint = output_dir / "checkpoint_best.pt"
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss, examples = 0.0, 0
            for index, batch in enumerate(tqdm(training_loader, desc=f"{representation} Epoch {epoch}/{config['epochs']}"), 1):
                batch.pop("record_ids")
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
                detailed = epoch == 1 and index == 1
                output = model(**batch, return_attention=detailed)
                require(torch.isfinite(output.loss) and torch.isfinite(output.logits).all(), "Non-finite training output")
                if detailed:
                    torch.testing.assert_close(output.loss, torch.nn.functional.binary_cross_entropy_with_logits(output.logits, batch["labels"]))
                    write(output_dir / "first_batch_shapes.json", dict(input_ids=list(batch["input_ids"].shape),
                          bert_hidden=list(output.hidden_states.shape), pooled=list(output.pooled.shape),
                          logits=list(output.logits.shape), targets=list(batch["labels"].shape),
                          query_parameters=0, representation=representation))
                size = len(batch["labels"])
                (output.loss * size).backward()
                if detailed:
                    require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), "Missing/non-finite first gradients")
                    metadata["first_batch_backward_verified"] = True
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(size)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                examples += size
                epoch_loss += float(output.loss.detach()) * size
            val_logits, val_gold, val_ids, val_loss = predict(model, validation_loader, "cuda")
            values, _ = score(val_gold, probabilities(val_logits), 0.30)
            row = dict(epoch=epoch, global_step=step, training_loss=epoch_loss / examples,
                       validation_loss=val_loss, validation_macro_f1=values["macro_f1"],
                       validation_micro_f1=values["micro_f1"], learning_rate=scheduler.get_last_lr()[0])
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
            print(row, flush=True)
            if values["macro_f1"] > best_f1:
                best_f1 = values["macro_f1"]
                torch.save(model.state_dict(), checkpoint)
                tokenizer.save_pretrained(output_dir / "checkpoint_tokenizer")
                validation_best = (val_logits.copy(), val_gold.copy(), val_ids.copy())
                write(output_dir / "checkpoint_selection.json", dict(epoch=epoch, global_step=step,
                      validation_macro_f1=best_f1, threshold=0.30))
        metadata["best_checkpoint"] = read(output_dir / "checkpoint_selection.json")
        del model, optimizer, scheduler, output
        torch.cuda.empty_cache()
        best_model = ControlBert.from_pretrained(str(MODEL_DIR), config)
        best_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        best_model = best_model.to("cuda")
        # Numerical checkpoint reload gate on validation only for both smoke/full.
        reloaded, _, _, _ = predict(best_model, [next(iter(validation_loader))], "cuda")
        np.testing.assert_allclose(reloaded, validation_best[0][:len(reloaded)], rtol=1e-4, atol=1e-4)
        metadata["checkpoint_reload_verified"] = True
        selected, search = tune_validation_threshold(validation_best[1], probabilities(validation_best[0]), config["threshold_candidates"])
        search.to_csv(output_dir / "threshold_search.csv", index=False)
        write(output_dir / "threshold_selection.json", dict(selected_threshold=selected, selection_split="validation",
              objective="macro_f1", tie_break="lowest threshold", formal_comparison_threshold=0.30, per_label_thresholds=False))
        _, val_rare = save_split("validation", *validation_best, selected, config, output_dir)
        rare = dict(definition=dict(label_ids=RARE_IDS, label_names=[LABEL_NAMES[i] for i in RARE_IDS]), validation=val_rare)
        if mode == "full":
            test_loader = split_loader(datasets["test"], tokenizer, config, 64)
            test_logits, test_gold, test_ids, _ = predict(best_model, test_loader, "cuda")
            _, test_rare = save_split("test", test_logits, test_gold, test_ids, selected, config, output_dir)
            rare["test"] = test_rare
        write(output_dir / "rare_label_metrics.json", rare)
        require(check_sources() == source, "Source changed during training")
        require(verify_seed42() == preflight["seed42_reference"], "Frozen seed42 reference changed during training")
        metadata.update(status="completed", end_time=now(), completed_optimizer_steps=step)
        write(output_dir / "environment.json", metadata)
        from multi_verify import verify_control
        verify_control(output_dir, require_success=False)
        status = "EXP5A_MULTISEED_SMOKE_VERIFIED" if mode == "smoke" else "EXP5A_MULTISEED_MODEL_VERIFIED"
        seal(output_dir, status, seed=seed, representation=representation, mode=mode)
        print(status, representation, output_dir, flush=True)
    except BaseException as exc:
        metadata.update(status="failed", end_time=now(), error=repr(exc))
        write(output_dir / "environment.json", metadata)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--representation", choices=("cls", "mean"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(123, 2026), required=True)
    args = parser.parse_args()
    run(args.phase, args.representation, args.run_root, args.seed)
