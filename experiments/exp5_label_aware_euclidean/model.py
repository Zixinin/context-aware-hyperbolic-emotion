"""BERT token encoder plus 28 learned Euclidean label queries and per-label heads."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LabelAwareOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    attention_weights: torch.Tensor | None = None
    label_representations: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class LabelAwareBert(nn.Module):
    def __init__(self, encoder, num_labels=28, dropout=0.1, initializer_std=0.02):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.hidden_size = int(encoder.config.hidden_size)
        self.label_queries = nn.Parameter(torch.empty(num_labels, self.hidden_size))
        self.classifier_weight = nn.Parameter(torch.empty(num_labels, self.hidden_size))
        self.classifier_bias = nn.Parameter(torch.zeros(num_labels))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.label_queries, mean=0.0, std=initializer_std)
        nn.init.normal_(self.classifier_weight, mean=0.0, std=initializer_std)

    @classmethod
    def from_pretrained(cls, path, config):
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            path,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
            add_pooling_layer=False,
        )
        return cls(
            encoder,
            num_labels=config["num_labels"],
            dropout=config["dropout"],
            initializer_std=config["query_initializer_std"],
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, return_attention=False):
        if attention_mask.ndim != 2 or not torch.all(attention_mask.sum(dim=1) > 0):
            raise ValueError("Every sequence must contain at least one unmasked token")
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = encoded.last_hidden_state
        scores = torch.einsum("nd,bld->bnl", self.label_queries, hidden) / math.sqrt(self.hidden_size)
        valid = attention_mask[:, None, :].to(torch.bool)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        label_representations = torch.bmm(weights, hidden)
        logits = (self.dropout(label_representations) * self.classifier_weight[None, :, :]).sum(-1)
        logits = logits + self.classifier_bias[None, :]
        loss = None
        if labels is not None:
            if labels.dtype != torch.float32 or labels.shape != logits.shape:
                raise ValueError("Expected float32 targets with shape [B, 28]")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return LabelAwareOutput(
            loss=loss,
            logits=logits,
            attention_weights=weights if return_attention else None,
            label_representations=label_representations if return_attention else None,
            hidden_states=hidden if return_attention else None,
        )


def make_optimizer(model, config):
    no_decay = ("bias", "LayerNorm.weight")
    decay, clean = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (clean if name.endswith("bias") or any(token in name for token in no_decay) else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]}, {"params": clean, "weight_decay": 0.0}],
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
    )


def parameter_report(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    label_module = model.label_queries.numel() + model.classifier_weight.numel() + model.classifier_bias.numel()
    return {
        "exp5a_total_parameters": total,
        "exp5a_trainable_parameters": trainable,
        "label_aware_module_parameters": label_module,
        "label_queries_parameters": model.label_queries.numel(),
        "per_label_classifier_parameters": model.classifier_weight.numel() + model.classifier_bias.numel(),
    }
