"""Unmodified BERT pooled output + default dropout/linear classification head."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABEL_NAMES


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, num_labels=28,
        problem_type="multi_label_classification", attn_implementation="eager",
        id2label=dict(enumerate(LABEL_NAMES)), label2id={name: i for i, name in enumerate(LABEL_NAMES)})
    return tokenizer, model


def make_optimizer(model, config):
    import torch
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(s in n for s in no_decay)],
         "weight_decay": config["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(s in n for s in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config["learning_rate"],
                            betas=(config["adam_beta1"], config["adam_beta2"]), eps=config["adam_epsilon"])
