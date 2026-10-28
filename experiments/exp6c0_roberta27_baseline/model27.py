"""Frozen A0 classification path for later A1/A2 reuse; BCE only."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from common import CONFIG, require


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Roberta27(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.classifier_weight = nn.Parameter(torch.empty(27, hidden))
        self.classifier_bias = nn.Parameter(torch.zeros(27))
        self.dropout = nn.Dropout(0.1)
        # Preserve the established FYP CLS initialization convention (no learned query).
        discarded_draw = torch.empty(27, hidden)
        nn.init.normal_(discarded_draw, mean=0, std=0.02)
        nn.init.normal_(self.classifier_weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        require(bool(torch.all(attention_mask[:, 0] == 1)), "First token is masked")
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              return_dict=True).last_hidden_state
        h = hidden[:, 0, :]
        # Exactly the existing FYP CLS dropout/classifier layout. In eval this is
        # F.linear(h, classifier_weight, classifier_bias), with no dense/tanh/pooler.
        expanded = h[:, None, :].expand(-1, 27, -1)
        logits = (self.dropout(expanded) * self.classifier_weight[None]).sum(-1) + self.classifier_bias[None]
        loss = None
        if labels is not None:
            require(labels.shape == logits.shape and labels.dtype == torch.float32, "Need float32 [B,27] targets")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        return dict(logits=logits, loss=loss, hidden_shape=list(hidden.shape), pooled_shape=list(h.shape))


def load_model(snapshot):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    encoder, info = AutoModel.from_pretrained(str(snapshot), local_files_only=True,
        use_safetensors=(snapshot / "model.safetensors").is_file(),
        attn_implementation="eager", add_pooling_layer=False, output_loading_info=True)
    require(not info.get("missing_keys") and not info.get("mismatched_keys") and not info.get("error_msgs"),
            f"Backbone did not load completely: {info}")
    cfg = encoder.config
    require(cfg.model_type == "roberta" and cfg.hidden_size == 768 and cfg.num_hidden_layers == 12
            and cfg.num_attention_heads == 12 and cfg.vocab_size == 50265, "Not roberta-base")
    require(cfg.hidden_dropout_prob == 0.1 and cfg.attention_probs_dropout_prob == 0.1, "Backbone dropout changed")
    require(tokenizer.bos_token_id == 0 and tokenizer.eos_token_id == 2 and tokenizer.pad_token_id == 1, "Wrong RoBERTa tokenizer")
    probe = tokenizer("A tokenizer identity check", add_special_tokens=True)["input_ids"]
    require(probe[0] == tokenizer.bos_token_id and probe[-1] == tokenizer.eos_token_id, "First token must be <s>")
    return Roberta27(encoder), tokenizer, info


def optimizer_for(model):
    decay, clean, names = [], [], [[], []]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = int(name.endswith("bias") or "bias" in name or "LayerNorm.weight" in name)
        (clean if group else decay).append(p)
        names[group].append(name)
    optimizer = torch.optim.AdamW([
        dict(params=decay, weight_decay=0.01), dict(params=clean, weight_decay=0.0)],
        lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
    require(sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters())), "Optimizer misses parameters")
    return optimizer, [dict(weight_decay=g["weight_decay"], parameter_names=n) for g, n in zip(optimizer.param_groups, names)]


def gradients(model):
    result = {}
    for group, params in [("encoder", list(model.encoder.parameters())),
                           ("classifier", [model.classifier_weight, model.classifier_bias])]:
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params), f"Missing/nonfinite {group} gradient")
        total = sum(float(p.grad.detach().double().square().sum()) for p in params)
        require(total > 0 and np.isfinite(total), f"Zero/invalid {group} gradient")
        result[group] = dict(norm=total ** 0.5, tensors=len(params),
                             nonzero_tensors=sum(bool(torch.count_nonzero(p.grad)) for p in params))
    return result


def train_step(model, batch, optimizer, scheduler, detailed=False):
    ids = batch.pop("ids")
    x = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    out = model(**x)
    require(bool(torch.isfinite(out["loss"])) and bool(torch.isfinite(out["logits"]).all()), "Nonfinite training forward")
    n = len(ids)
    (out["loss"] * n).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(n)
    report = gradients(model) if detailed else None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=float(out["loss"].detach()), batch_size=n, unclipped_grad_norm=float(norm),
                gradients=report, logits_shape=list(out["logits"].shape), targets_shape=list(x["labels"].shape),
                hidden_shape=out["hidden_shape"], pooled_shape=out["pooled_shape"],
                multi_examples=int((x["labels"].sum(1) > 1).sum()), ids=ids)
