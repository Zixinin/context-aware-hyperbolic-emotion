"""CPU-safe synthetic tests for shape, mask, gradients, loss and source boundaries."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from exp5lib import HERE, load_config, source_hashes, write
from metrics import probabilities, score, tune_validation_threshold
from model import LabelAwareBert


class DummyEncoder(nn.Module):
    def __init__(self, vocab=31, hidden=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_dict=True):
        del attention_mask, token_type_ids, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_model():
    torch.manual_seed(42)
    model = LabelAwareBert(DummyEncoder(), num_labels=28, dropout=0.0, initializer_std=0.02)
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    labels = torch.zeros(2, 28, dtype=torch.float32)
    labels[0, [2, 17]] = 1
    labels[1, [14]] = 1
    output = model(input_ids, attention_mask, labels=labels, return_attention=True)
    assert output.hidden_states.shape == (2, 5, 12)
    assert model.label_queries.shape == (28, 12)
    assert output.attention_weights.shape == (2, 28, 5)
    assert output.label_representations.shape == (2, 28, 12)
    assert output.logits.shape == labels.shape
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.attention_weights).all()
    assert torch.equal(output.attention_weights[0, :, 3:], torch.zeros(28, 2))
    torch.testing.assert_close(output.attention_weights.sum(-1), torch.ones(2, 28))
    expected = nn.functional.binary_cross_entropy_with_logits(output.logits, labels)
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert model.label_queries.grad is not None and torch.isfinite(model.label_queries.grad).all()
    assert model.classifier_weight.grad is not None and torch.isfinite(model.classifier_weight.grad).all()
    assert model.classifier_bias.grad is not None and torch.isfinite(model.classifier_bias.grad).all()
    assert model.encoder.embedding.weight.grad is not None


def test_metrics():
    targets = np.zeros((3, 28), dtype=np.float32)
    targets[0, 0] = targets[1, 1] = targets[2, 2] = 1
    logits = np.full((3, 28), -8.0)
    logits[0, 0] = logits[1, 1] = logits[2, 2] = 8.0
    values, frame = score(targets, probabilities(logits), 0.30)
    assert values["micro_f1"] == 1.0
    assert len(frame) == 28
    selected, search = tune_validation_threshold(targets, probabilities(logits), [0.3, 0.5])
    assert selected == 0.3 and len(search) == 2


def test_source_boundary():
    forbidden = {"lgsa", "hierarchy", "graph", "gat", "gcn", "hyperbolic"}
    imports = []
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    bad = sorted(name for name in imports if any(token in name.lower().split(".") for token in forbidden))
    assert not bad, f"Exp5A imports forbidden model modules: {bad}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_config()
    test_model()
    test_metrics()
    test_source_boundary()
    report = {
        "status": "EXP5A_UNIT_TESTS_PASS",
        "tests": ["shape", "attention_mask", "BCE", "backward_gradients", "finite_values", "metrics", "source_boundary"],
        "source_sha256": source_hashes(),
    }
    write(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
