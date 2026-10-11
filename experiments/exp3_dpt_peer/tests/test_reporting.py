"""Tests for scientific eligibility and matched evaluation semantics."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reporting import aggregate_siblings, frozen_pairs, DIRECTIONS, validation_curve
from protocol import source_audit


class ReportingTests(unittest.TestCase):
    def test_validation_curve_accepts_native_and_normalized_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            expected = [(i, i / 10) for i in range(1, 5)]
            for epoch, value in expected:
                (directory / f'validation_epoch_{epoch}.json').write_text(json.dumps({'macro_f1': value}))
            self.assertEqual(validation_curve(directory), expected)
            path = directory / 'validation_metrics_by_epoch.csv'
            path.write_text('epoch,validation_macro_f1\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n')
            self.assertEqual(validation_curve(directory), expected)
            path.write_text('epoch,macro_f1\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n')
            self.assertEqual(validation_curve(directory), expected)
            path.write_text('epoch,validation_macro_f1\n1,0.1\n2,0.2\n3,0.3\n')
            with self.assertRaises(RuntimeError):
                validation_curve(directory)

    def test_aggregate_excludes_all_co_gt_and_neutral(self):
        parents = [30] * 28
        parents[2] = parents[3] = 29
        parents[27] = 31
        y = np.zeros((3, 28), dtype=bool)
        y[0, 2] = True
        y[1, [2, 3]] = True
        y[2, 27] = True
        z = np.zeros((3, 28))
        z[0, 2], z[0, 3] = .2, .8
        result = aggregate_siblings(y, z, parents)
        self.assertEqual(result['eligible_gt_instances'], 1)
        self.assertLess(result['mean_margin'], 0)
        self.assertEqual(result['inversion_rate'], 1)
        self.assertEqual(result['conditional_fp_rate'], 1)

    def test_inversion_uses_logits_despite_sigmoid_saturation(self):
        parents = list(range(28))
        parents[1] = parents[0]
        y = np.zeros((1, 28), dtype=bool)
        y[0, 0] = True
        z = np.zeros((1, 28))
        z[0, 0], z[0, 1] = 90, 100
        result = aggregate_siblings(y, z, parents)
        self.assertEqual(result['inversion_rate'], 1)

    def test_all_twelve_pairs_are_frozen_and_co_gt_filtered(self):
        hierarchy = json.loads((ROOT.parents[1] / 'experiments/exp2_1_multiseed/frozen/hierarchy.json').read_text())
        labels = hierarchy['labels']
        names = [r['fine_label'] for r in labels]
        parents = [r['coarse_id'] for r in labels]
        y = np.eye(28, dtype=bool)
        y = np.concatenate([y, y[:1]])
        y[-1, names.index('approval')] = True
        z = np.zeros((29, 28))
        rows = frozen_pairs(y, z, z, names, parents)
        self.assertEqual([(r['true_label'], r['sibling_negative']) for r in rows], DIRECTIONS)
        selected = next(r for r in rows if r['true_label'] == 'admiration' and r['sibling_negative'] == 'approval')
        self.assertEqual(selected['N'], 1)
        self.assertEqual(selected['co_gt_excluded'], 1)
        self.assertTrue(all(r['delta_margin'] == r['delta_inversion_rate'] == r['delta_conditional_fp_rate'] == 0 for r in rows))


if __name__ == '__main__':
    unittest.main()
