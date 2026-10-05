"""Lightweight synthetic CPU logic only; never load or train BERT here."""
import unittest

import numpy as np

from src.common import load_config, require_slurm_gpu
from src.data import LABEL_NAMES, MultiLabelCollator, multi_hot
from src.metrics import probabilities, rare_label_scores, score, tune_validation_threshold


class LogicTests(unittest.TestCase):
    def test_multi_hot_preserves_multiple_labels(self):
        vector = multi_hot([9, 25])
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (28,))
        self.assertEqual(np.flatnonzero(vector).tolist(), [9, 25])

    def test_invalid_labels_rejected(self):
        for labels in ([], [28], [-1], [1, 1], [1.5]):
            with self.assertRaises(ValueError):
                multi_hot(labels)

    def test_sigmoid_stability(self):
        result = probabilities(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(result, [0, 0.5, 1])
        with self.assertRaises(ValueError):
            probabilities([np.nan])

    def test_micro_macro_use_all_28_labels(self):
        target = np.zeros((2, 28))
        target[0, [0, 1]] = 1
        target[1, 0] = 1
        probs = np.zeros_like(target)
        probs[0, 0] = 0.3  # equality is a positive decision
        probs[1, [0, 1]] = 0.5
        metrics, frame = score(target, probs, 0.3)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 28)
        self.assertEqual(len(frame), 28)
        self.assertTrue(frame["f1"].is_monotonic_increasing)
        self.assertEqual(int(frame["support"].sum()), 3)

    def test_zero_predictions_do_not_force_argmax(self):
        targets = np.stack([multi_hot([0]), multi_hot([1])])
        values, _ = score(targets, np.full((2, 28), 0.01), 0.30)
        self.assertEqual(values["micro_precision"], 0.0)
        self.assertEqual(values["micro_recall"], 0.0)

    def test_thresholds_use_validation_and_lowest_tie(self):
        targets = np.stack([multi_hot([0]), multi_hot([0])])
        probs = np.zeros((2, 28))
        probs[:, 0] = 0.2
        selected, table = tune_validation_threshold(targets, probs, [0.3, 0.15, 0.1])
        self.assertEqual(selected, 0.1)
        self.assertEqual(table["threshold"].tolist(), [0.1, 0.15, 0.3])

    def test_rare_averages_use_label_ids_after_sort(self):
        target = np.stack([multi_hot([0]), multi_hot([1])])
        _, frame = score(target, target, 0.3)
        values = rare_label_scores(frame, [0])
        self.assertEqual(values["mean_rare_label_f1"], 1)
        self.assertAlmostEqual(values["mean_non_rare_label_f1"], 1 / 27)

    def test_dynamic_collation_keeps_float32(self):
        import torch
        class DummyTokenizer:
            def pad(self, records, padding, return_tensors):
                length = max(len(row["input_ids"]) for row in records)
                return {"input_ids": torch.tensor([row["input_ids"] + [0] * (length - len(row["input_ids"])) for row in records])}
        batch = MultiLabelCollator(DummyTokenizer())([
            {"input_ids": [101, 102], "labels": multi_hot([9, 25]).tolist()},
            {"input_ids": [101, 100, 102], "labels": multi_hot([27]).tolist()},
        ])
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["labels"].shape, (2, 28))
        self.assertEqual(batch["labels"].dtype, torch.float32)

    def test_config_protocol(self):
        config = load_config()
        self.assertEqual(config["per_device_train_batch_size"] * config["gradient_accumulation_steps"], 16)
        self.assertEqual(config["threshold_candidates"], [round(i * 0.05, 2) for i in range(1, 19)])
        self.assertEqual(len(LABEL_NAMES), 28)

    def test_local_training_guard(self):
        import platform
        if platform.system() != "Linux":
            with self.assertRaises(RuntimeError):
                require_slurm_gpu()


if __name__ == "__main__":
    unittest.main()
