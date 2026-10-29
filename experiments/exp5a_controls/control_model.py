"""Pure CLS/mean representations; no pooler, query parameter or attention module."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ControlOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    pooled: torch.Tensor | None = None


class ControlBert(nn.Module):
    def __init__(self, encoder, representation, num_labels=28, dropout=0.1, initializer_std=0.02):
        super().__init__()
        if representation not in {"cls", "mean"}:
            raise ValueError("Only cls and mean controls are authorized")
        self.encoder = encoder
        self.representation = representation
        self.num_labels = num_labels
        self.hidden_size = int(encoder.config.hidden_size)
        self.classifier_weight = nn.Parameter(torch.empty(num_labels, self.hidden_size))
        self.classifier_bias = nn.Parameter(torch.zeros(num_labels))
        self.dropout = nn.Dropout(dropout)
        # Match frozen C's RNG consumption before head init. Not stored/trainable,
        # not used in forward, and not a query module.
        discarded_draw = torch.empty(num_labels, self.hidden_size)
        nn.init.normal_(discarded_draw, mean=0.0, std=initializer_std)
        nn.init.normal_(self.classifier_weight, mean=0.0, std=initializer_std)

    @classmethod
    def from_pretrained(cls, path, config):
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                            attn_implementation="eager", add_pooling_layer=False)
        return cls(encoder, config["representation"], config["num_labels"], config["dropout"],
                   config["query_initializer_std"])

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, return_attention=False):
        if attention_mask.ndim != 2 or not torch.all(attention_mask.sum(-1) > 0):
            raise ValueError("Each sequence needs an unmasked token")
        if not torch.all(attention_mask[:, 0] == 1):
            raise ValueError("CLS must be unmasked at position zero")
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=True).last_hidden_state
        if self.representation == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        # The head is algebraically Linear(D -> 28). Expand only to match frozen
        # C's independent-per-label dropout masks, not to introduce token attention.
        representations = pooled[:, None, :].expand(-1, self.num_labels, -1)
        logits = (self.dropout(representations) * self.classifier_weight[None, :, :]).sum(-1)
        logits = logits + self.classifier_bias[None, :]
        loss = None
        if labels is not None:
            if labels.dtype != torch.float32 or labels.shape != logits.shape:
                raise ValueError("Expected float32 targets [B, 28]")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return ControlOutput(loss, logits, hidden if return_attention else None,
                             pooled if return_attention else None)


def parameter_report(model):
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "classifier_parameters": model.classifier_weight.numel() + model.classifier_bias.numel(),
            "query_parameters": 0}
