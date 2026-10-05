"""One-GPU FP32 BERT reference; real runs require a Linux Slurm GPU allocation."""
import argparse
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from .common import ROOT, capture, load_config, package_versions, read_json, require_slurm_gpu, sha256, source_hashes, utc_now, write_json
from .data import MultiLabelCollator, analyze, load_prepared, tokenize_dataset
from .evaluate import predict, report_split, select_threshold
from .metrics import probabilities, score
from .model import load_model, make_optimizer
from .visualize import training_figures


def run(config, output_dir):
    require_slurm_gpu()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This protocol supports one GPU only.")
    output_dir = Path(output_dir)
    # Atomic directory claim avoids concurrent jobs overwriting each other's results.
    output_dir.mkdir(parents=True, exist_ok=False)
    start = utc_now()
    metadata = {"status": "running", "start_time": start, "end_time": None, "hostname": platform.node(),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu_model": torch.cuda.get_device_name(0),
                "number_of_gpus": torch.cuda.device_count(), "cuda_module": os.environ.get("CUDA_MODULE"),
                "torch_cuda_runtime": torch.version.cuda, "python": platform.python_version(), "python_executable": sys.executable,
                "packages": package_versions(), "git_commit": capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                "git_status": capture(["git", "-C", str(ROOT), "status", "--porcelain"]),
                "source_sha256": source_hashes(), "config": config, "nvidia_smi": capture(["nvidia-smi"]),
                "implementation_details": ["FP32; TF32 disabled; eager attention; deterministic algorithms required",
                    "Linear warmup/decay; AdamW excludes bias and LayerNorm weights from weight decay",
                    "Keep final partial batch, normalize accumulated gradients by actual number of examples",
                    "Epoch-end model selection at fixed 0.30; first epoch wins exact F1 ties",
                    "Validation threshold ties choose lowest threshold; >= decision rule; no forced label",
                    "No early stopping; no AMP; no resume; standard pretrained BERT pooler/head"]}
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "environment.json", metadata)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output_dir / "pip-freeze.txt").write_text(freeze)
        metadata["pip_freeze_sha256"] = sha256(output_dir / "pip-freeze.txt")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        set_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset, manifest = load_prepared()
        metadata["assets_manifest"] = manifest
        metadata["assets_manifest_sha256"] = sha256(ROOT / "assets/manifest.json")
        metadata["dataset_split_sizes"] = {s: len(dataset[s]) for s in dataset}
        frequencies, _, rare = analyze(dataset, output_dir / "dataset_analysis")
        figures = output_dir / "figures"
        figures.mkdir(exist_ok=True)
        shutil.copy2(output_dir / "dataset_analysis/label_frequency.png", figures / "label_frequency.png")
        tokenizer, model = load_model(manifest["model_snapshot"])
        model.to("cuda")
        train_data, validation_data = dataset["train"], dataset["validation"]
        if config["mode"] == "smoke":
            train_data = train_data.shuffle(seed=config["seed"]).select(range(config["smoke_train_samples"]))
            validation_data = validation_data.shuffle(seed=config["seed"]).select(range(config["smoke_validation_samples"]))
        metadata["actual_split_sizes"] = {"train": len(train_data), "validation": len(validation_data),
                                           "test_evaluated": 0 if config["mode"] == "smoke" else len(dataset["test"])}
        write_json(output_dir / "sample_ids.json", {"train": list(train_data["id"]), "validation": list(validation_data["id"])})
        collator = MultiLabelCollator(tokenizer)
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = DataLoader(tokenize_dataset(train_data, tokenizer, config["max_length"]),
                                  batch_size=config["per_device_train_batch_size"], shuffle=True, generator=generator,
                                  collate_fn=collator, num_workers=config["num_workers"], pin_memory=True)
        def evaluation_loader(split):
            return DataLoader(tokenize_dataset(split, tokenizer, config["max_length"]),
                              batch_size=config["eval_batch_size"], shuffle=False, collate_fn=collator,
                              num_workers=config["num_workers"], pin_memory=True)
        validation_loader = evaluation_loader(validation_data)
        optimizer = make_optimizer(model, config)
        steps_per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["epochs"]
        warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        metadata.update({"optimizer_steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
                         "warmup_steps": warmup_steps, "nominal_effective_batch_size": 16,
                         "last_optimizer_batch_examples": len(train_data) % 16 or 16})
        write_json(output_dir / "environment.json", metadata)
        history, best_f1, global_step = [], -1.0, 0
        checkpoint = output_dir / "checkpoints/best"
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_examples, epoch_examples, epoch_loss = 0, 0, 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            for batch_index, batch in enumerate(progress, 1):
                batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                if batch["labels"].dtype != torch.float32 or batch["labels"].shape[1] != 28:
                    raise RuntimeError("Expected float32 (batch, 28) multi-hot targets.")
                output = model(**batch)
                loss = output.loss  # BertForSequenceClassification selects BCEWithLogitsLoss.
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                size = batch["labels"].shape[0]
                if global_step == 0 and batch_index == 1:
                    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, batch["labels"])
                    torch.testing.assert_close(loss, expected_loss)
                    metadata["bce_loss_verified"] = True
                (loss * size).backward()
                accumulated_examples += size
                epoch_examples += size
                epoch_loss += float(loss.detach()) * size
                if batch_index % config["gradient_accumulation_steps"] == 0 or batch_index == len(train_loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_examples)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_examples = 0
                    global_step += 1
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}", step=global_step)
            val_logits, val_targets, val_loss = predict(model, validation_loader, "cuda")
            val_metrics, _ = score(val_targets, probabilities(val_logits), config["fixed_threshold"])
            row = {"epoch": epoch, "global_step": global_step, "training_loss": epoch_loss / epoch_examples,
                   "validation_loss": val_loss, "validation_macro_f1": val_metrics["macro_f1"],
                   "validation_micro_f1": val_metrics["micro_f1"], "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
            print(row, flush=True)
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                model.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                write_json(checkpoint / "selection.json", {"epoch": epoch, "global_step": global_step,
                           "validation_macro_f1": best_f1, "threshold": config["fixed_threshold"]})
                np.savez_compressed(output_dir / "validation_predictions.npz", logits=val_logits, targets=val_targets,
                                    ids=np.asarray(validation_data["id"], dtype=str))
        metadata["best_checkpoint"] = read_json(checkpoint / "selection.json")
        training_figures(pd.DataFrame(history), figures)
        # Reload the serialized best checkpoint to exercise checkpoint integrity.
        del model, optimizer, scheduler, output, loss
        torch.cuda.empty_cache()
        _, best_model = load_model(str(checkpoint))
        best_model.to("cuda")
        saved_validation = np.load(output_dir / "validation_predictions.npz", allow_pickle=False)
        val_logits, val_targets = saved_validation["logits"], saved_validation["targets"]
        selected = select_threshold(val_logits, val_targets, config, output_dir)
        _, val_rare = report_split("validation", val_logits, val_targets, selected, config, output_dir, rare, frequencies)
        rare_report = {"definition": rare, "validation": val_rare}
        if config["mode"] == "full":
            # The only test forward pass in the entire experiment. Threshold is already frozen on disk.
            test_logits, test_targets, _ = predict(best_model, evaluation_loader(dataset["test"]), "cuda")
            np.savez_compressed(output_dir / "test_predictions.npz", logits=test_logits, targets=test_targets,
                                ids=np.asarray(dataset["test"]["id"], dtype=str))
            _, test_rare = report_split("test", test_logits, test_targets, selected, config, output_dir, rare, frequencies)
            rare_report["test"] = test_rare
        else:
            # A small validation-only reload check; no test-set use in smoke.
            reloaded, _, _ = predict(best_model, [next(iter(validation_loader))], "cuda")
            np.testing.assert_allclose(reloaded, val_logits[:len(reloaded)], rtol=1e-4, atol=1e-4)
            metadata["checkpoint_reload_verified"] = True
        write_json(output_dir / "rare_label_metrics.json", rare_report)
        if source_hashes() != metadata["source_sha256"]:
            raise RuntimeError("Project source changed during the run; cannot certify reproducibility.")
        metadata.update({"status": "completed", "end_time": utc_now(), "completed_optimizer_steps": global_step})
        write_json(output_dir / "environment.json", metadata)
        from scripts.summarize_results import summarize
        summarize(output_dir)
        # Verify actual artifacts before creating a success marker.
        from scripts.verify_run import verify_artifacts
        verify_artifacts(output_dir, require_success=False)
        write_json(output_dir / "SUCCESS.json", {"mode": config["mode"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
                   "completed_at": utc_now(), "source_sha256": metadata["source_sha256"],
                   "artifacts_sha256": {str(p.relative_to(output_dir)): sha256(p) for p in output_dir.rglob("*") if p.is_file()}})
        print(f"SUCCESS: {output_dir}", flush=True)
    except BaseException as exc:
        metadata.update({"status": "failed", "end_time": utc_now(), "error": repr(exc)})
        write_json(output_dir / "environment.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default=str(ROOT / "config/baseline_bert.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, choices=(8, 16))
    parser.add_argument("--gradient-accumulation-steps", type=int, choices=(1, 2))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    require_slurm_gpu()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] != 16:
        raise ValueError("Choose 16x1 or 8x2; effective batch size must remain 16.")
    config["mode"] = args.mode
    if args.mode == "smoke":
        config["epochs"] = 1
    default_output = ROOT / "outputs" / ("smoke" if args.mode == "smoke" else "bert_baseline") / f"seed{config['seed']}"
    if args.mode == "smoke":
        default_output = default_output / f"job{os.environ['SLURM_JOB_ID']}"
    output = args.output_dir or default_output
    if args.mode == "full":
        from scripts.verify_run import verify_smoke_gate
        if not os.environ.get("SMOKE_RUN"):
            raise RuntimeError("Full run requires SMOKE_RUN pointing to a verified successful smoke job.")
        verify_smoke_gate(Path(os.environ["SMOKE_RUN"]), config)
    run(config, output)


if __name__ == "__main__":
    main()
