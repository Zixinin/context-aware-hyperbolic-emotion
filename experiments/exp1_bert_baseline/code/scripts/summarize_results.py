"""Write the dissertation summary from measured outputs, never estimated scores."""
import argparse
import csv
from pathlib import Path

from src.common import read_json


def summarize(run_dir):
    run_dir = Path(run_dir)
    config, meta = read_json(run_dir / "config.json"), read_json(run_dir / "environment.json")
    split = "test" if config["mode"] == "full" else "validation"
    metrics = read_json(run_dir / f"{split}_metrics.json")
    rare = read_json(run_dir / "rare_label_metrics.json")
    with (run_dir / "per_label_metrics.csv").open() as stream:
        labels = list(csv.DictReader(stream))
    lines = ["# GoEmotions BERT baseline 实验记录", "",
             "正式测试集结果。" if config["mode"] == "full" else "SMOKE 工程检查：以下仅为验证子集结果，不能作为科研结论。", "",
             f"- 官方划分：{meta['dataset_split_sizes']}", f"- 本次实际样本：{meta['actual_split_sizes']}",
             f"- 数据来源：{meta['assets_manifest']['data_source']}",
             f"- Python {meta['python']}；PyTorch {meta['packages']['torch']}；CUDA runtime {meta['torch_cuda_runtime']}",
             f"- GPU：{meta['gpu_model']}；Slurm Job ID：{meta['slurm_job_id']}",
             f"- 模型：{config['model_name']}；seed：{config['seed']}",
             f"- 训练：{config['epochs']} epochs；LR={config['learning_rate']}；batch={config['per_device_train_batch_size']}；"
             f"accumulation={config['gradient_accumulation_steps']}；effective batch=16；max length={config['max_length']}",
             f"- 最佳 checkpoint：epoch {meta['best_checkpoint']['epoch']}，验证集固定 0.30 Macro-F1={meta['best_checkpoint']['validation_macro_f1']:.6f}",
             f"- 开始：{meta['start_time']}；结束：{meta['end_time']}", "",
             "| Protocol | Threshold | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol, values in metrics.items():
        numbers = [values[k] for k in ("threshold", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")]
        lines.append("| " + protocol + " | " + " | ".join(f"{v:.6f}" for v in numbers) + " |")
    lines += ["", "稀有标签由完整训练集频次最低的 7 类定义：" + ", ".join(rare["definition"]["label_names"]), ""]
    for protocol, values in rare[split].items():
        lines.append(f"- {protocol}：rare mean F1={values['mean_rare_label_f1']:.6f}；non-rare mean F1={values['mean_non_rare_label_f1']:.6f}")
    lines += ["", "固定 0.30 阈值下：", "", "- Top 5：" + ", ".join(f"{r['label']} ({float(r['f1']):.4f})" for r in reversed(labels[-5:])),
              "- Bottom 5：" + ", ".join(f"{r['label']} ({float(r['f1']):.4f})" for r in labels[:5]), "", "实现细节与边界：", ""]
    lines += [f"- {detail}" for detail in meta["implementation_details"]]
    if meta["assets_manifest"]["dataset"].get("deviation"):
        lines.append("- " + meta["assets_manifest"]["dataset"]["deviation"])
    lines += ["- 本次不生成可选 PCA。", "- 单 seed 42，不代表跨种子平均表现。" if config["seed"] == 42 else f"- 单 seed {config['seed']}。",
              "- 完成证据还需核对 sacct COMPLETED / 0:0 和 Slurm 日志。", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    summarize(parser.parse_args().run_dir)
