"""Synthetic CPU tests: seed plumbing, unchanged loop, paired stats/report and tamper gates."""
from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from multi_common import (AUTHORIZED, ALL_SEEDS, HERE, check_sources, config_for, model_directory,
                          checked_artifacts, seal, write, read, sha)
from control_model import ControlBert
from control_tests import tests as frozen_tests
from unit_tests import DummyEncoder
from model import LabelAwareBert
from train import split_loader, subset


def test_seed_plumbing():
    torch.manual_seed(1)
    encoder = DummyEncoder()
    heads = []
    for seed in AUTHORIZED:
        for mode in ("smoke", "full"):
            for representation in ("cls", "mean"):
                cfg = config_for(mode, representation, seed)
                assert cfg["seed"] == seed and cfg["authorized_seeds"] == [123, 2026]
                assert cfg["epochs"] == (1 if mode == "smoke" else 4)
                assert f"seed{seed}" in model_directory("/synthetic", mode, representation, seed).name
        torch.manual_seed(seed)
        query = LabelAwareBert(copy.deepcopy(encoder))
        expected = query.classifier_weight.detach().clone()
        state = torch.get_rng_state()
        for representation in ("cls", "mean"):
            torch.manual_seed(seed)
            model = ControlBert(copy.deepcopy(encoder), representation)
            assert torch.equal(expected, model.classifier_weight)
            assert torch.equal(state, torch.get_rng_state())
        heads.append(expected)
    assert not torch.equal(heads[0], heads[1])
    for seed in (42, 0, 124, -1):
        try:
            config_for("full", "cls", seed)
            raise AssertionError("Unauthorized new seed accepted")
        except RuntimeError:
            pass
    # Loader uses config seed, not a hidden seed42. Original Collator is unused
    # here because sampler order can be examined independently of tokenization.
    records = [{"id": str(i)} for i in range(60)]
    orders = []
    for seed in AUTHORIZED:
        cfg = config_for("full", "cls", seed)
        a = list(split_loader(records, None, cfg, 16, shuffle=True).sampler)
        b = list(split_loader(records, None, cfg, 16, shuffle=True).sampler)
        assert a == b and sorted(a) == list(range(60))
        assert subset(records, 10, seed) == subset(records, 10, seed)
        orders.append(a)
    assert orders[0] != orders[1]


def test_loop_identity():
    # The epoch/gradient/optimizer/validation/checkpoint block is byte-identical
    # to the frozen seed42 loop. No optimizer/head/training changes accompany seeds.
    original = (HERE.parent / "exp5a_controls/control_train.py").read_text()
    new = (HERE / "multi_train.py").read_text()
    start = "        for epoch in range(1, config[\"epochs\"] + 1):"
    end = "        metadata[\"best_checkpoint\"]"
    assert original.split(start, 1)[1].split(end, 1)[0] == new.split(start, 1)[1].split(end, 1)[0]
    module = ast.parse(new)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"seed_everything", "subset"}
                   and any(isinstance(a, ast.Constant) and a.value == 42 for a in n.args) for n in ast.walk(module))


def test_verifier_seed_fixtures():
    """Execute the frozen synthetic smoke fixture through each actual new verifier."""
    import inspect
    import control_tests as old_tests
    from multi_common import RARE_IDS
    source = inspect.getsource(old_tests.test_verifier_fixture)
    replacements = [
        ('def test_verifier_fixture():', 'def test_verifier_fixture(seed, mode):'),
        ('import control_verify as verifier', 'import multi_verify as verifier'),
        ('from control_common import config_for, freeze, read, sha', 'from multi_common import config_for, freeze, read, sha'),
        ('config_for("smoke", "mean")', 'config_for(mode, "mean", seed)'),
        ('subset(datasets["train"], 1000, 42)', 'datasets["train"] if mode == "full" else subset(datasets["train"], 1000, seed)'),
        ('subset(datasets["validation"], 500, 42)', 'datasets["validation"] if mode == "full" else subset(datasets["validation"], 500, seed)'),
        ('EXP5A_CONTROL_PREFLIGHT_PASS', 'EXP5A_MULTISEED_PREFLIGHT_PASS'),
        ('c_reference', 'seed42_reference'),
        ('EXP5A_CONTROL_SMOKE_VERIFIED', 'EXP5A_MULTISEED_SMOKE_VERIFIED'),
        ('representation="mean", mode="smoke")', 'seed=seed, representation="mean", mode=mode)'),
        ('EXP5A_CONTROL_ARTIFACT_CHECK_PASS', 'EXP5A_MULTISEED_ARTIFACT_CHECK_PASS'),
    ]
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    full_changes = [
        ('"completed_optimizer_steps": 63, "total_optimizer_steps": 63', '"completed_optimizer_steps": (10856 if mode == "full" else 63), "total_optimizer_steps": (10856 if mode == "full" else 63)'),
        ('"global_step": 63, "validation_macro_f1"', '"global_step": (2714 if mode == "full" else 63), "validation_macro_f1"'),
        ('pd.DataFrame([dict(epoch=1, global_step=63,', 'pd.DataFrame([dict(epoch=i, global_step=(2714*i if mode == "full" else 63),'),
        ('learning_rate=0.0)]).to_csv', 'learning_rate=0.0) for i in range(1, cfg["epochs"]+1)]).to_csv'),
        ('seal(root, "EXP5A_MULTISEED_SMOKE_VERIFIED",', 'seal(root, "EXP5A_MULTISEED_MODEL_VERIFIED" if mode == "full" else "EXP5A_MULTISEED_SMOKE_VERIFIED",'),
    ]
    for old, new in full_changes:
        assert old in source, old
        source = source.replace(old, new)
    anchor = '        write(root / "rare_label_metrics.json", {"definition": {"label_ids": RARE_IDS}, "validation": rare})'
    full_test = '''        rare_report = {"definition": {"label_ids": RARE_IDS}, "validation": rare}
        if mode == "full":
            test_gold = np.stack([multi_hot(r["labels"]) for r in datasets["test"]])
            test_logits = np.where(test_gold == 1, 5.0, -5.0).astype(np.float32)
            test_ids = np.asarray([r["id"] for r in datasets["test"]])
            _, test_rare = save_split("test", test_logits, test_gold, test_ids, selected, cfg, root)
            rare_report["test"] = test_rare
        write(root / "rare_label_metrics.json", rare_report)'''
    assert anchor in source
    source = source.replace(anchor, full_test)
    namespace = dict(old_tests.__dict__)
    namespace.update(check_sources=check_sources, seal=seal, write=write, RARE_IDS=RARE_IDS)
    exec(compile(source, "<synthetic_multiseed_verifier_fixture>", "exec"), namespace)
    for seed in AUTHORIZED:
        for mode in ("smoke", "full"):
            namespace["test_verifier_fixture"](seed, mode)


def test_report():
    import multi_compare as compare
    from data import LABEL_NAMES, RARE_IDS
    a = compare.stats([1.0, 2.0, 3.0])
    assert a["mean"] == 2.0 and a["sample_sd"] == 1.0
    d = compare.stats([0.02, -0.01, 0.0])
    assert (d["positive_seeds"], d["negative_seeds"], d["zero_seeds"]) == (1, 1, 1)
    with tempfile.TemporaryDirectory(prefix="exp5a_multi_report_fixture_") as directory:
        root = Path(directory)
        old_root = root / "frozen42"
        old_root.mkdir()
        reference = {"synthetic_only": True}
        frames, metrics = {}, {}
        for seed in ALL_SEEDS:
            for model in ("cls", "mean"):
                run = (old_root if seed == 42 else root) / f"{model}_seed{seed}"
                run.mkdir()
                frame = pd.DataFrame(dict(label_id=range(28), label=LABEL_NAMES, support=[20]*28,
                                          precision=[0.4]*28, recall=[0.4]*28, f1=[0.4]*28))
                frame.loc[16, "support"], frame.loc[23, "support"] = 6, 11
                shift = {42: 0.02, 123: -0.01, 2026: 0.0}[seed] if model == "mean" else 0.0
                frame.f1 += shift
                frames[str(run)] = frame
                metrics[str(run)] = {key: 0.4+shift for _, key in compare.METRICS if key != "rare_f1"}
                write(run / "config.json", dict(mode="full", representation=model, seed=seed))
                write(run / "environment.json", dict(seed42_reference=reference, parameters_runtime={"trainable_parameters": 107741212}))
                write(run / "checkpoint_selection.json", {"epoch": 4})
                write(run / "sample_ids.json", {"synthetic": True})
                write(run / "SUCCESS.json", {"synthetic": True})
        write(root / "preflight.json", {"synthetic": True})
        def measured(run, split, records):
            f = frames[str(run)]
            return metrics[str(run)], {"mean_rare_label_f1": float(f.loc[f.label_id.isin(RARE_IDS), "f1"].mean())}, f
        with patch.object(compare, "SEED42_ROOT", old_root), patch.object(compare, "verify_seed42", return_value=reference), \
             patch.object(compare, "load_all_verified", return_value=({"test": [], "validation": []}, {})), \
             patch.object(compare, "verify_control", return_value={}), \
             patch.object(compare, "verify_seed42_model", return_value={}), \
             patch.object(compare, "verify_prediction_metrics", side_effect=measured):
            payload, raw, paired, labels, markdown = compare.build(root)
            assert len(raw) == 6 and len(paired) == 3 and len(labels) == 28
            np.testing.assert_allclose(payload["aggregates"]["macro_f1"]["mean_minus_cls"]["mean"], 0.01/3, atol=1e-12)
            assert payload["aggregates"]["macro_f1"]["mean_minus_cls"]["positive_seeds"] == 1
            assert "Signs are mixed" in markdown and "Query remains a seed42 pilot" in markdown
            with contextlib.redirect_stdout(io.StringIO()):
                compare.compare(root)
                compare.compare(root, verify_only=True)
            changed = read(root / "MULTISEED_COMPARISON.json")
            changed["aggregates"]["macro_f1"]["mean_minus_cls"]["mean"] = 9.0
            write(root / "MULTISEED_COMPARISON.json", changed)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    compare.compare(root, verify_only=True)
                raise AssertionError("Tampered comparison accepted")
            except RuntimeError:
                pass


def tests():
    old = frozen_tests()
    test_seed_plumbing()
    test_loop_identity()
    test_verifier_seed_fixtures()
    test_report()
    with tempfile.TemporaryDirectory(prefix="exp5a_multi_seal_") as directory:
        root = Path(directory)
        write(root / "artifact.json", {"value": 1})
        seal(root, "SYNTHETIC_ONLY", seed=123)
        assert checked_artifacts(root)["seed"] == 123
    return old + ["new_seed_authorization", "seedwise_head_RNG_pairing", "seeded_shuffle_and_subset",
                  "frozen_training_loop_identity", "sample_SD_ddof1", "paired_delta_direction_counts",
                  "three_seed_report_recomputation", "comparison_tamper_rejection", "new_seed_seal_identity",
                  "seed123_smoke_full_verifier_fixtures", "seed2026_smoke_full_verifier_fixtures"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="EXP5A_MULTISEED_UNIT_TESTS_PASS", tests=tests(), source_sha256=check_sources())
    write(args.out, result)
    print(result["status"], result["tests"], flush=True)
