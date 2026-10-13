"""CPU synthetic tests only. These are NOT Ubuntu scientific results."""
import ast
import csv
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import analysis as a
import probe as p
import run as runner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LABELS = sorted(json.loads((REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions/goemotions_adaptation/hierarchy.json').read_text())['labels'],
                key=lambda r: r['fine_id'])


class ArrayTests(unittest.TestCase):
    def test_metrics_match_frozen_original_evaluator(self):
        import sys
        adapter = REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions/goemotions_adaptation'
        sys.path.insert(0, str(adapter))
        spec = importlib.util.spec_from_file_location('frozen_reference_evaluator', adapter / 'evaluation.py')
        evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
        rng = np.random.RandomState(23)
        gold, logits = (rng.rand(100, 28) < .08).astype(int), rng.randn(100, 28).astype(np.float32)
        scores = a.probability(logits)
        original, original_rows = evaluator.metrics(gold, scores, .30, a.RARE)
        computed = a.overall(gold, scores)
        for key in ('micro_precision', 'micro_recall', 'micro_f1', 'macro_f1', 'macro_precision', 'macro_recall'):
            self.assertEqual(computed[key], original[key])
        rows = a.per_label(gold, logits, logits, LABELS, np.arange(28))
        for ours, reference in zip(rows, original_rows):
            for key in ('tp', 'fp', 'fn', 'prediction_count', 'precision', 'recall', 'f1'):
                self.assertEqual(ours['exp2_'+key], reference[key])

    def test_counts_and_empty_denominators(self):
        gold = np.array([[1, 0], [0, 1], [1, 0]])
        scores = np.array([[.8, .4], [.1, .2], [.2, .1]])
        result = a.counts(gold, scores)
        np.testing.assert_array_equal(result['tp'], [1, 0])
        np.testing.assert_array_equal(result['fp'], [0, 1])
        np.testing.assert_array_equal(result['fn'], [1, 1])
        np.testing.assert_allclose(result['f1'], [2/3, 0])
        m = a.overall(gold, scores)
        self.assertAlmostEqual(m['micro_f1'], .4)
        self.assertAlmostEqual(m['average_predicted_label_count'], 2/3)

    def test_spearman_ties_constant_and_missing(self):
        self.assertAlmostEqual(a.spearman([1, 1, 2, 3], [4, 4, 2, 1])['rho'], -1)
        self.assertIsNone(a.spearman([1, 1, 1], [2, 3, 4])['rho'])
        self.assertEqual(a.spearman([None, 2, 3], [1, 2, 3])['n'], 2)
        from scipy.stats import spearmanr
        x, y = [1, 5, 5, 3, 7], [2, 2, 7, 8, 1]
        self.assertAlmostEqual(a.spearman(x, y)['rho'], float(spearmanr(x, y).statistic))

    def test_co_gt_excluded_singleton_and_logit_inversion(self):
        y = np.array([[1, 1, 0, 0], [0, 0, 0, 1]])
        z = np.array([[90., 100., 91., 200.], [0., 0., 0., 1.]])
        rows = a.sibling_per_label(y, z, [28, 28, 28, 31])
        self.assertEqual(rows[0]['eligible_gt_instances'], 1)
        self.assertEqual(rows[0]['inversion_rate'], 1)
        self.assertEqual(rows[1]['inversion_rate'], 0)
        self.assertIsNone(rows[3]['margin'])
        self.assertEqual(int(a.eligible_mask(y, [28, 28, 28, 31]).sum()), 2)

    def test_all_family_co_gt_is_ineligible(self):
        y = np.array([[1, 1, 0]])
        self.assertFalse(a.eligible_mask(y, [28, 28, 31]).any())

    def test_all_labels_branches_and_gains(self):
        y = np.eye(28, dtype=int)
        z0 = np.full((28, 28), -5., dtype=np.float32)
        z1 = z0.copy(); z1[np.diag_indices(28)] = 2.
        rows = a.per_label(y, z0, z1, LABELS, np.arange(1, 29))
        self.assertEqual(len(rows), 28)
        self.assertEqual(a.gain_summary(rows)['all']['improved'], 28)
        self.assertEqual(a.gain_summary(rows)['neutral']['labels'], 1)
        br = a.branches(rows)
        self.assertEqual(sum(r['fine_labels'] for r in br), 28)
        self.assertEqual(sum(r['delta_total_tp'] for r in br), 28)
        self.assertTrue(all(r['delta_mean_f1'] == 1 for r in br))

    def test_score_shift_missing_predicted_positive_is_not_zero(self):
        y = np.eye(28, dtype=int)
        z = np.full((28, 28), -5., dtype=np.float32)
        rows = a.score_shift(y, z, z, LABELS)
        self.assertIsNone(rows[0]['exp2_predicted_positive_score'])
        self.assertEqual(rows[0]['delta_separation'], 0.)

    def test_activity_subsets_and_fp_context(self):
        y = np.zeros((3, 28), dtype=int); y[0, 0] = 1; y[1, [0, 1]] = 1; y[2, 27] = 1
        z0 = np.full(y.shape, -5., dtype=np.float32); z1 = z0.copy()
        z1[0, 1] = 2.; z1[1, [0, 1]] = 2.; z1[2, 0] = 2.
        result = a.activity(y, z0, z1)
        self.assertEqual(result['exp3']['total_predicted_positives'], 4)
        self.assertEqual(result['exp3']['tp'], 2)
        self.assertEqual(result['exp3']['fp'], 2)
        self.assertEqual(sum(r['examples'] for r in result['exp3']['count_distribution'].values()), 3)
        parents = [r['coarse_id'] for r in LABELS]
        groups = a.example_groups(y, z0, z1, parents)
        self.assertEqual(next(r['examples'] for r in groups if r['subset'] == 'single_gt' and r['model'] == 'exp3'), 2)
        ec = a.error_context(y, z1, parents)
        self.assertEqual(ec['total_fp'], 2)
        self.assertEqual(ec['sibling_linked_fp'], 1)
        self.assertEqual(ec['fn_with_predicted_fp_sibling'], 1)

    def test_prediction_identity_never_silently_aligns(self):
        x = {'gold': np.eye(2), 'ids': np.array(['a', 'b'])}
        with self.assertRaisesRegex(RuntimeError, 'IDs'):
            a.matched(x, {'gold': x['gold'], 'ids': np.array(['b', 'a'])})

    def test_gradient_fractions_zero_norm_and_sd(self):
        rows = [{'scope': 'x', 'cosine': c} for c in [-.4, -.01, .01, .4, None]]
        result = a.gradient_summary(rows)['x']
        self.assertEqual(result['valid_cosines'], 4)
        self.assertEqual(result['fraction_negative'], .5)
        self.assertEqual(result['fraction_near_zero_abs_le_0.05'], .5)
        self.assertEqual(result['fraction_aligned_gt_0.05'], .25)
        self.assertAlmostEqual(result['sd_sample'], float(np.std([-.4, -.01, .01, .4], ddof=1)))


class ProbeTests(unittest.TestCase):
    def test_actual_kernel_loss_not_selection_count_and_co_gt(self):
        import sys
        package = REPO / 'experiments/exp3_dpt_peer'
        sys.path.insert(0, str(package))
        from peer_contrastive import PeerHardNegativeContrastiveLoss
        loss_fn = PeerHardNegativeContrastiveLoss([28, 28, 28, 31], .1, 1)
        anchors = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        keys = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], requires_grad=True)
        scores = torch.tensor([[0., 9., 8., 0.], [8., 0., 9., 0.]])
        gold = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 0]])
        loss, stats = loss_fn(anchors, keys, scores, gold)
        terms = p.term_values(loss_fn, anchors, keys, scores, gold)
        self.assertEqual(len(terms), stats['number_of_peer_terms'])
        self.assertEqual([t['negative_id'] for t in terms], [2, 2, 2])
        self.assertAlmostEqual(float(loss), np.mean([t['raw_loss'] for t in terms]), places=6)
        self.assertNotEqual(terms[0]['raw_loss'], terms[1]['raw_loss'])

    def test_raw_vs_weighted_loss_contribution(self):
        terms = [dict(true_label='admiration', negative_label='approval', parent='Positive',
                      raw_loss=1., weighted_batch_contribution=.5),
                 dict(true_label='anger', negative_label='annoyance', parent='Negative',
                      raw_loss=3., weighted_batch_contribution=.75),
                 dict(true_label='confusion', negative_label='curiosity', parent='Ambiguous',
                      raw_loss=1., weighted_batch_contribution=.25)]
        true, parents, negatives, pairs, summary = p.aggregate_terms(terms, LABELS, np.arange(28), 2, .5)
        self.assertEqual(len(true), 28)
        self.assertEqual(negatives[0]['label'], 'annoyance')
        negative_parent = next(r for r in parents if r['label'] == 'Negative')
        self.assertAlmostEqual(negative_parent['raw_loss_share'], .6)
        self.assertAlmostEqual(negative_parent['weighted_batch_loss_share'], .5)
        self.assertAlmostEqual(summary['weighted_mean_peer_loss'], .75)
        neutral = next(r for r in true if r['label'] == 'neutral')
        self.assertIsNone(neutral['raw_loss_mean'])

    def test_streaming_cosine_excludes_unshared(self):
        named = [('bert.encoder.layer.0.x', torch.nn.Parameter(torch.ones(2))),
                 ('multiclass_bias', torch.nn.Parameter(torch.ones(2))),
                 ('cls.predictions.transform.x', torch.nn.Parameter(torch.ones(2)))]
        x = [torch.tensor([1., 2.]), torch.ones(2), torch.tensor([3., 0.])]
        y = [torch.tensor([-1., -2.]), None, torch.tensor([0., 3.])]
        rows, excluded, included = p.compare_gradients(named, x, y, 0, .5)
        full = rows[0]
        self.assertAlmostEqual(full['cosine'], -5/14)
        self.assertAlmostEqual(full['peer_gradient_norm_weighted'], .5*math.sqrt(14))
        self.assertEqual(len(included), 2)
        self.assertEqual(excluded[0]['parameter'], 'multiclass_bias')
        self.assertAlmostEqual(next(r['cosine'] for r in rows if r['scope'] == 'BERT_encoder'), -1.)
        self.assertIsNone(next(r['cosine'] for r in rows if r['scope'] == 'GAT_path'))

    def test_records_term_eligibility(self):
        records = [{'labels': [0, 1]}, {'labels': [27]}]
        parents = {i: r['coarse_id'] for i, r in enumerate(LABELS)}
        self.assertEqual(p.eligible_mask_for_records(records, parents), [2, 0])


class SafetyTests(unittest.TestCase):
    def test_no_optimizer_or_backward_or_checkpoint_write(self):
        for filename in ('run.py', 'analysis.py', 'probe.py', 'report.py'):
            tree = ast.parse((HERE / filename).read_text())
            calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertFalse(any(isinstance(f, ast.Attribute) and f.attr in
                                 ('step', 'backward', 'zero_grad', 'save', 'save_pretrained') for f in calls), filename)
            self.assertNotIn('torch.optim', (HERE / filename).read_text())

    def test_wrong_host_rejected_without_forward(self):
        with patch('platform.system', return_value='Darwin'), self.assertRaisesRegex(RuntimeError, 'ONLY'):
            runner.audit(REPO)

    def test_snapshot_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.json'
            source.write_text('{"a":1}')
            before = runner.snapshot([root])
            source.write_text('{"a":2}')
            self.assertNotEqual(before, runner.snapshot([root]))

    def test_exclusive_writes_and_json_no_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a.jsonwrite(root / 'x.json', {'value': None})
            with self.assertRaises(FileExistsError):
                a.jsonwrite(root / 'x.json', {'value': 2})
            with self.assertRaises(ValueError):
                a.jsonwrite(root / 'nan.json', {'value': float('nan')})
            a.csvwrite(root / 'x.csv', [{'x': 1, 'missing': None}])
            with (root / 'x.csv').open() as stream:
                self.assertEqual(list(csv.DictReader(stream))[0], {'x': '1', 'missing': ''})


class RenderingTests(unittest.TestCase):
    def test_synthetic_report_and_all_required_figures(self):
        """All generated data here are synthetic; no real experiment is replayed."""
        from report import figures, write_report
        y = np.eye(28, dtype=int)
        z0 = np.full(y.shape, -1., dtype=np.float32); z0[np.diag_indices(28)] = 2.
        z1 = z0.copy(); z1[0, 0] = -2.; z1[1, 2] = 1.; z1[16, 16] = 3.
        rows = a.per_label(y, z0, z1, LABELS, np.arange(1, 29)*10)
        br, shifted, active = a.branches(rows), a.score_shift(y, z0, z1, LABELS), a.activity(y, z0, z1)
        terms = [dict(true_label='admiration', negative_label='approval', parent='Positive', raw_loss=1.,
                      weighted_batch_contribution=.5),
                 dict(true_label='anger', negative_label='annoyance', parent='Negative', raw_loss=2.,
                      weighted_batch_contribution=1.),
                 dict(true_label='confusion', negative_label='curiosity', parent='Ambiguous', raw_loss=.5,
                      weighted_batch_contribution=.25)]
        true, parents, neg, pairs, actual = p.aggregate_terms(terms, LABELS, np.arange(1, 29)*10, 3, .5)
        gradient_rows = [{'batch_index': b, 'scope': scope, 'cosine': (b-5)*.04} for b in range(10) for scope in
                         ['full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform']]
        gradient = a.gradient_summary(gradient_rows)
        rare = a.rare_details(y, z0, z1, LABELS, rows, [], pairs)
        validation = {'epoch3': active['exp2'], 'epoch4': active['exp3'],
            'labels_margin_improved_f1_decreased': [], 'cosine_margin_improved_f1_decreased': [],
            'existing_replay_comparison': [{'metric': 'mean_probability_margin', 'delta_epoch4_minus_epoch3': .01},
                                          {'metric': 'macro_f1', 'delta_epoch4_minus_epoch3': -.001}]}
        summary = {'main_metrics': active, 'gain_loss': a.gain_summary(rows), 'gradient_scopes': gradient,
                   'validation_epoch3_4': validation, 'protected_file_count': 0, 'branches': br,
                   'peer_loss_snapshot': actual, 'sibling_correlations': a.sibling_correlations(rows),
                   'error_context': {'exp2': a.error_context(y, z1, [r['coarse_id'] for r in LABELS])},
                   'grief_relief': rare}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'MPLCONFIGDIR': temp+'/mpl'}):
            out = Path(temp)
            a.csvwrite(out / 'gradient_conflict_batches.csv', gradient_rows)
            figures(out, rows, br, shifted, true, neg, active, gradient,
                    [r for r in rows if r['structurally_peer_eligible']])
            self.assertEqual(len(list((out / 'figures').glob('*.png'))), 10)
            self.assertTrue(all(p.stat().st_size > 1000 for p in (out / 'figures').glob('*.png')))
            if os.environ.get('FYP_DIAG_TEST_RENDER_DIR'):
                shutil.copytree(out / 'figures', Path(os.environ['FYP_DIAG_TEST_RENDER_DIR']))
            choice = write_report(out, summary, rows, parents, true, neg, rows)
            self.assertEqual(choice, 'STOP_DPT_BRANCH')
            report = (out / 'EXP3B_DIAGNOSTIC_SUMMARY.md').read_text()
            for q in range(1, 11):
                self.assertIn('## Q'+str(q)+' ', report)
            validation['existing_replay_comparison'][0]['delta_epoch4_minus_epoch3'] = -.01
            other = out / 'different_scope'; other.mkdir()
            self.assertEqual(write_report(other, summary, rows, parents, true, neg, rows), 'INSUFFICIENT_EVIDENCE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
