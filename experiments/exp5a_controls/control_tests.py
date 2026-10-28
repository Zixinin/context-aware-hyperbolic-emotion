"""Synthetic CPU tests only. No Mac pretrained-model training."""
from __future__ import annotations

import argparse
import ast
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

from control_common import HERE, check_sources, checked_artifacts, seal, write, RARE_IDS
from control_model import ControlBert
from unit_tests import DummyEncoder, test_metrics
from model import LabelAwareBert
import torch


def tests():
    torch.manual_seed(9)
    encoder = DummyEncoder()
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    targets = torch.zeros(2, 28, dtype=torch.float32)
    targets[0, [2, 17]] = 1
    targets[1, 14] = 1
    rngs, heads = [], []
    torch.manual_seed(42)
    query = LabelAwareBert(copy.deepcopy(encoder), dropout=0.1)
    query_rng = torch.get_rng_state()
    for representation in ("cls", "mean"):
        torch.manual_seed(42)
        model = ControlBert(copy.deepcopy(encoder), representation, dropout=0.1)
        rngs.append(torch.get_rng_state())
        heads.append(model.classifier_weight.detach().clone())
        assert torch.equal(heads[-1], query.classifier_weight)
        assert torch.equal(rngs[-1], query_rng)
        assert not hasattr(model, "label_queries")
        assert not any("query" in name for name, _ in model.named_parameters())
        model.eval()
        output = model(ids, mask, labels=targets, return_attention=True)
        hidden = model.encoder.embedding(ids)
        expected = hidden[:, 0] if representation == "cls" else torch.stack([hidden[0, :3].mean(0), hidden[1].mean(0)])
        torch.testing.assert_close(output.pooled, expected)
        torch.testing.assert_close(output.logits, torch.nn.functional.linear(expected, model.classifier_weight, model.classifier_bias))
        altered = ids.clone()
        altered[0, 3:] = torch.tensor([12, 13])
        torch.testing.assert_close(model(altered, mask).logits, output.logits)
        torch.testing.assert_close(output.loss, torch.nn.functional.binary_cross_entropy_with_logits(output.logits, targets))
        output.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        state = copy.deepcopy(model.state_dict())
        reload = ControlBert(copy.deepcopy(encoder), representation)
        reload.load_state_dict(state, strict=True)
        reload.eval()
        torch.testing.assert_close(reload(ids, mask).logits, output.logits)
        try:
            model(ids, torch.zeros_like(mask))
            raise AssertionError("Empty masks accepted")
        except ValueError:
            pass
        # Mean is equivalent to zero-query uniform attention including CLS/SEP.
        if representation == "mean":
            query.eval()
            with torch.no_grad():
                query.label_queries.zero_()
            torch.testing.assert_close(query(ids, mask).logits, output.logits)
        # Training dropout must consume the same RNG and masks as C.
        model.train()
        torch.manual_seed(77)
        actual = model(ids, mask, return_attention=True)
        after = torch.get_rng_state()
        torch.manual_seed(77)
        query.dropout.train()
        direct = query.dropout(actual.pooled[:, None, :].expand(-1, 28, -1))
        direct = (direct * model.classifier_weight[None]).sum(-1) + model.classifier_bias
        torch.testing.assert_close(actual.logits, direct)
        assert torch.equal(after, torch.get_rng_state())
    test_metrics()
    test_proofs_and_report()
    test_verifier_fixture()
    forbidden = {"lgsa", "hierarchy", "graph", "gat", "gcn", "hyperbolic"}
    for path in HERE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(forbidden.intersection(name.lower().split(".")) for name in names)
    return ["cls_formula", "masked_mean_formula", "PAD_invariance", "linear_head_equivalence",
            "no_query_parameters", "BCE", "finite_gradients", "checkpoint_reload",
            "empty_mask_rejected", "C_head_and_RNG_parity", "C_dropout_parity",
            "uniform_query_equivalence", "metrics", "source_boundary",
            "artifact_tamper_rejection", "report_machine_metrics", "signed_rare_contribution",
            "verifier_synthetic_smoke", "metric_tamper_rejection"]


def test_proofs_and_report():
    import pandas as pd
    import numpy as np
    import control_compare as compare
    from data import LABEL_NAMES
    with tempfile.TemporaryDirectory(prefix="exp5a_control_synthetic_") as directory:
        root = Path(directory)
        proof_root = root / "proof"
        proof_root.mkdir()
        write(proof_root / "artifact.json", {"value": 1})
        seal(proof_root, "SYNTHETIC_ONLY")
        checked_artifacts(proof_root)
        write(proof_root / "artifact.json", {"value": 2})
        try:
            checked_artifacts(proof_root)
            raise AssertionError("Artifact tampering accepted")
        except RuntimeError:
            pass
        base = pd.DataFrame(dict(label_id=range(28), label=LABEL_NAMES, support=[20]*28,
                                 precision=[0.4]*28, recall=[0.4]*28, f1=[0.4]*28))
        base.loc[16, "support"], base.loc[23, "support"] = 6, 11
        frames = {}
        values = {}
        query_dir = root / "query"
        c_reference = {"synthetic": True}
        for name, folder in (("cls", root / "cls_seed42"), ("mean", root / "mean_seed42"), ("query", query_dir)):
            folder.mkdir()
            frame = base.copy()
            if name != "cls":
                frame.loc[[16, 23], "f1"] += 0.1
            if name == "query":
                frame.loc[1, "f1"] += 0.02
            frames[str(folder)] = frame
            v = {key: 0.4 for _, key in compare.METRICS if key != "rare_f1"}
            v["macro_f1"] = float(frame.f1.mean())
            v.update(threshold=0.30, num_examples=5427)
            values[str(folder)] = v
            write(folder / "SUCCESS.json", {"status": "SYNTHETIC_ONLY"})
            write(folder / "sample_ids.json", {"synthetic": ["a", "b"]})
            write(folder / "checkpoint_selection.json", {"epoch": 2})
            write(folder / "validation_metrics.json", {"fixed_0.30": v})
            parameters = {"exp5a_trainable_parameters": 107762716} if name == "query" else {"trainable_parameters": 107741212}
            write(folder / "environment.json", {"parameters_runtime": parameters, "c_reference": c_reference})
        def measurements(folder, split, records):
            frame = frames[str(folder)]
            return values[str(folder)], {"mean_rare_label_f1": float(frame.loc[frame.label_id.isin(RARE_IDS), "f1"].mean())}, frame
        with patch.object(compare, "C_RUN", query_dir), patch.object(compare, "verify_c", return_value=c_reference), \
             patch.object(compare, "load_all_verified", return_value=({"test": []}, {})), \
             patch.object(compare, "verify_control", return_value={}), \
             patch.object(compare, "verify_prediction_metrics", side_effect=measurements), \
             patch.object(compare, "verify_reference", return_value={"formal_metrics": values[str(root / "cls_seed42")], "rare_f1_fixed_0.30": 0.4}):
            payload, table, markdown = compare.build(root)
        assert len(table) == 28 and "N/A in verified artifact" not in markdown
        d = payload["grief_relief_contributions"]["Mean - CLS"]
        np.testing.assert_allclose(d["macro_delta"], 0.2/28, atol=1e-12)
        np.testing.assert_allclose(d["grief_relief_macro_contribution"], 0.2/28, atol=1e-12)
        np.testing.assert_allclose(d["grief_relief_rare_contribution"], 0.2/7, atol=1e-12)
        assert d["improved_labels"] == 2 and d["regressed_labels"] == 0
        np.testing.assert_allclose(payload["deltas"]["Query - Mean"]["macro_f1"], 0.02/28, atol=1e-12)
        assert "Historical Exp1 (reference only)" in markdown and "107741212" in markdown


def test_verifier_fixture():
    """Exercise verifier end-to-end on explicitly synthetic CPU-only artifacts."""
    import numpy as np
    import pandas as pd
    import control_verify as verifier
    from control_common import config_for, freeze, read, sha
    from data import LABEL_NAMES, multi_hot
    from train import subset, save_split
    from metrics import probabilities, tune_validation_threshold
    cfg = config_for("smoke", "mean")
    datasets = {split: [{"id": f"{split}_{i}", "text": "synthetic", "labels": [i % 28]}
                       for i in range(count)] for split, count in (("train", 1000), ("validation", 500), ("test", 1))}
    training = subset(datasets["train"], 1000, 42)
    validation = subset(datasets["validation"], 500, 42)
    gold = np.stack([multi_hot(r["labels"]) for r in validation])
    logits = np.full_like(gold, -5.0)
    logits[gold == 1] = 5.0
    ids = np.asarray([r["id"] for r in validation])
    selected, grid = tune_validation_threshold(gold, probabilities(logits), cfg["threshold_candidates"])
    with tempfile.TemporaryDirectory(prefix="exp5a_control_verifier_fixture_") as directory:
        root = Path(directory)
        package_freeze = freeze()
        source = check_sources()
        params = {"trainable_parameters": 107741212}
        # Keep all fixture data under the temporary root, including its preflight.
        preflight = root / "synthetic_preflight.json"
        write(preflight, {"status": "EXP5A_CONTROL_PREFLIGHT_PASS", "source_sha256": source, "parameters": {"mean": params}, "c_reference": {},
                          "packages": {}, "pip_freeze": package_freeze})
        write(root / "config.json", cfg)
        write(root / "environment.json", {"status": "completed", "source_sha256": source,
              "hostname": "D3020-A3", "gpu": "synthetic RTX 4090 fixture (not GPU evidence)",
              "first_batch_backward_verified": True, "checkpoint_reload_verified": True,
              "preflight_path": str(preflight), "preflight_sha256": sha(preflight),
              "parameters_runtime": params, "c_reference": {}, "packages": {},
              "completed_optimizer_steps": 63, "total_optimizer_steps": 63})
        (root / "pip-freeze.txt").write_text(package_freeze)
        (root / "checkpoint_best.pt").write_bytes(b"SYNTHETIC_NOT_A_TRAINED_MODEL")
        write(root / "checkpoint_tokenizer/tokenizer_config.json", {"synthetic": True})
        # A controlled frequency fixture is necessary because synthetic class
        # counts do not share the real dataset's rare-frequency ordering.
        fixture_counts = gold.sum(0).astype(int).tolist()
        write(root / "dataset_statistics.json", dict(splits={k: len(v) for k, v in datasets.items()},
              num_labels=28, label_order=dict(enumerate(LABEL_NAMES)), train_label_counts=fixture_counts, rare_ids=RARE_IDS))
        write(root / "sample_ids.json", {"train": [r["id"] for r in training], "validation": [r["id"] for r in validation]})
        write(root / "first_batch_shapes.json", {"logits": [16, 28], "targets": [16, 28], "pooled": [16, 768], "query_parameters": 0})
        val, rare = save_split("validation", logits, gold, ids, selected, cfg, root)
        write(root / "rare_label_metrics.json", {"definition": {"label_ids": RARE_IDS}, "validation": rare})
        values = val["fixed_0.30"]
        write(root / "checkpoint_selection.json", {"epoch": 1, "global_step": 63, "validation_macro_f1": values["macro_f1"], "threshold": 0.30})
        pd.DataFrame([dict(epoch=1, global_step=63, training_loss=0.2, validation_loss=0.1,
              validation_macro_f1=values["macro_f1"], validation_micro_f1=values["micro_f1"], learning_rate=0.0)]).to_csv(root / "training_log.csv", index=False)
        grid.to_csv(root / "threshold_search.csv", index=False)
        write(root / "threshold_selection.json", dict(selected_threshold=selected, selection_split="validation",
              objective="macro_f1", tie_break="lowest threshold", formal_comparison_threshold=0.30, per_label_thresholds=False))
        seal(root, "EXP5A_CONTROL_SMOKE_VERIFIED", representation="mean", mode="smoke")
        with patch.object(verifier, "load_all_verified", return_value=(datasets, {})), \
             patch.object(verifier, "frequencies", return_value=np.asarray(fixture_counts)):
            assert verifier.verify_control(root)["status"] == "EXP5A_CONTROL_ARTIFACT_CHECK_PASS"
            changed = read(root / "validation_metrics.json")
            changed["fixed_0.30"]["macro_f1"] = 0.0
            write(root / "validation_metrics.json", changed)
            try:
                verifier.verify_control(root)
                raise AssertionError("Tampered metric accepted")
            except RuntimeError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = dict(status="EXP5A_CONTROL_UNIT_TESTS_PASS", tests=tests(), source_sha256=check_sources())
    write(args.out, result)
    print(result["status"], result["tests"], flush=True)
