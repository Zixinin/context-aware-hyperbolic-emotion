"""Synthetic CPU tests only; never read or execute scientific runs."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('complete_analysis', Path(__file__).with_name('complete_analysis.py'))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def test_normalized_and_native_metrics_both_reconcile(self):
        measured = dict(micro_precision=.5, micro_recall=.6, micro_f1=.55, macro_f1=.4)
        for alias in ('rare_f1', 'rare_mean_f1'):
            saved = dict(measured, **{alias: .35})
            recovery.reconcile(measured, saved, .35)
            with self.assertRaises(RuntimeError):
                recovery.reconcile(measured, saved, .36)
            saved['micro_f1'] += .01
            with self.assertRaises(RuntimeError):
                recovery.reconcile(measured, saved, .35)

    def test_both_aliases_must_agree(self):
        self.assertEqual(recovery.rare_metric({'rare_f1': .5, 'rare_mean_f1': .5}), .5)
        with self.assertRaises(RuntimeError):
            recovery.rare_metric({'rare_f1': .5, 'rare_mean_f1': .51})

    def test_missing_invalid_or_nonfinite_alias_rejected(self):
        for value in ({}, {'rare_f1': float('nan')}, {'rare_mean_f1': float('inf')}, {'rare_f1': 50}):
            with self.assertRaises(RuntimeError):
                recovery.rare_metric(value)

    def test_metrics_do_not_mutate_saved_or_measured_dict(self):
        measured = dict(micro_precision=.5, micro_recall=.6, micro_f1=.55, macro_f1=.4)
        saved = dict(measured, rare_f1=.35)
        original = json.dumps(saved, sort_keys=True)
        recovery.reconcile(measured, saved, .35)
        self.assertEqual(original, json.dumps(saved, sort_keys=True))
        self.assertNotIn('rare_mean_f1', saved)

    def test_equivalence_rejects_changed_counts_keys_nulls_and_values(self):
        record = [{'eligible_terms': 4, 'raw_loss_sum': 1.234, 'raw_loss_mean': None, 'label': 'grief'}]
        recovery.equivalent(record, json.loads(json.dumps(record)))
        for changed in (
            [{'eligible_terms': 5, 'raw_loss_sum': 1.234, 'raw_loss_mean': None, 'label': 'grief'}],
            [{'eligible_terms': 4., 'raw_loss_sum': 1.234, 'raw_loss_mean': None, 'label': 'grief'}],
            [{'eligible_terms': 4, 'raw_loss_sum': 1.235, 'raw_loss_mean': None, 'label': 'grief'}],
            [{'eligible_terms': 4, 'raw_loss_sum': 1.234, 'raw_loss_mean': 0., 'label': 'grief'}],
            [{'eligible_terms': 4, 'raw_loss_sum': 1.234, 'raw_loss_mean': None}],
        ):
            with self.assertRaises(RuntimeError):
                recovery.equivalent(record, changed)

    def test_csv_roundtrip_preserves_ids_json_zero_and_none(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'table.csv'
            row = {'sample_id': '00042', 'label': 'neutral', 'eligible_terms': 0,
                   'raw_loss_mean': None, 'raw_loss_sum': 0., 'negative_selection_counts': '{}'}
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            recovery.equivalent([row], recovery.csvread(path))

    def test_nonfinite_csv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.csv'
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['raw_loss_sum'])
                writer.writeheader(); writer.writerow({'raw_loss_sum': 'nan'})
            with self.assertRaises(RuntimeError):
                recovery.csvread(path)

    def test_partial_reuse_includes_integrity_and_both_gradient_tables(self):
        self.assertEqual(len(set(recovery.GPU_FILES)), 12)
        for filename in ('model_state_integrity.json', 'gradient_conflict_batches.csv',
                         'gradient_conflict_by_layer.csv', 'peer_loss_terms_snapshot.csv'):
            self.assertIn(filename, recovery.GPU_FILES)
        self.assertNotIn('FAILURE.json', recovery.GPU_FILES)
        self.assertNotIn('SUCCESS.json', recovery.GPU_FILES)

    def test_full_saved_probe_reconciliation_cpu_synthetic_only(self):
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp3b_diagnostics'))
        from analysis import csvwrite, gradient_summary, jsonwrite
        from probe import aggregate_terms
        repo = Path(__file__).resolve().parents[2]
        labels = sorted(json.loads((repo / 'experiments/exp2_lgsa/LGSA-GoEmotions/goemotions_adaptation/hierarchy.json').read_text())['labels'],
                        key=lambda r: r['fine_id'])
        train = [{'id': 'toy'+str(i), 'labels': [0 if i < 36884 else 27]} for i in range(43410)]
        frequency = np.array([sum(i in row['labels'] for row in train) for i in range(28)])
        order = np.random.RandomState(42).permutation(len(train)).tolist()
        ordered = [train[i] for i in order]
        terms, batches = [], []
        for b in range(2714):
            selected = ordered[b*16:(b+1)*16]
            n = sum(row['labels'] == [0] for row in selected)
            for row in selected:
                if row['labels'] == [0]:
                    terms.append({'batch_index': b, 'sample_id': row['id'], 'true_label': labels[0]['fine_label'],
                        'negative_label': labels[1]['fine_label'], 'parent': labels[0]['coarse_label'],
                        'eligible_terms_in_batch': n, 'raw_loss': math.log(2), 'cosine_margin': 0.,
                        'weighted_batch_contribution': .5*math.log(2)/n})
            batches.append({'batch_index': b, 'examples': len(selected), 'eligible_terms': n,
                'actual_peer_loss': math.log(2) if n else 0., 'weighted_peer_loss': .5*math.log(2) if n else 0.})
        true_rows, parent_rows, negative_rows, pairs, actual = aggregate_terms(terms, labels, frequency, 2714, .5)
        actual['train_examples'] = 43410
        scopes = ['full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform']
        def grad(b, scope):
            return {'batch_index': b, 'scope': scope, 'shared_parameter_tensors': 1,
                'original_gradient_norm': 1., 'peer_gradient_norm_unweighted': .4,
                'peer_gradient_norm_weighted': .2, 'cosine': .2, 'weighted_peer_to_original_norm_ratio': .2,
                'original_lgsa_loss': 3., 'peer_loss': batches[b]['actual_peer_loss'], 'hga_alpha': 1.}
        gradients = [grad(b, scope) for b in range(10) for scope in scopes]
        layers = [grad(b, 'BERT_layer_'+str(k)) for b in range(10) for k in range(12)]
        with tempfile.TemporaryDirectory() as temp:
            partial = Path(temp) / 'partial'; partial.mkdir()
            run = Path(temp) / 'run'; run.mkdir()
            baseline = Path(temp) / 'baseline'; baseline.mkdir()
            for path in (run / 'test_fixed_0.30_metrics.json', run / 'validation_epoch_3.json', run / 'validation_epoch_4.json'):
                jsonwrite(path, {'rare_mean_f1': .5})
            jsonwrite(baseline / 'test_fixed_0.30_metrics.json', {'rare_f1': .5})
            for filename, rows in [('peer_loss_terms_snapshot.csv', terms), ('peer_loss_batches_snapshot.csv', batches),
                    ('peer_loss_by_true_label.csv', true_rows), ('peer_loss_by_parent.csv', parent_rows),
                    ('hard_negative_loss_contribution.csv', negative_rows), ('peer_loss_by_pair_snapshot.csv', pairs),
                    ('gradient_conflict_batches.csv', gradients), ('gradient_conflict_by_layer.csv', layers)]:
                csvwrite(partial / filename, rows)
            jsonwrite(partial / 'fixed_batch_protocol.json', {
                'batch_size': 16, 'gradient_batch_indices': list(range(10)),
                'train_ids_in_forward_order': [r['id'] for r in ordered],
                'gradient_sample_ids': [[r['id'] for r in ordered[b*16:(b+1)*16]] for b in range(10)]})
            jsonwrite(partial / 'model_state_integrity.json', {
                'all_parameters_and_buffers_exactly_unchanged': True, 'frozen_best_and_final_keys_equal_initial': True,
                'optimizer_instances_created': 0, 'optimizer_steps': 0, 'before_sha256': 'a'*64, 'after_sha256': 'a'*64})
            jsonwrite(partial / 'peer_loss_contribution_summary.json', actual)
            jsonwrite(partial / 'gradient_conflict_summary.json', {
                'scopes': gradient_summary(gradients), 'checkpoint_epoch': 3, 'frozen_raw_prompt_and_keys': True,
                'shared_parameter_names': ['toy.parameter'],
                'excluded_by_batch': [{'batch_index': b, 'excluded_unshared': []} for b in range(10)]})
            context = dict(train=train, labels=labels, frequency=frequency, run=run, baseline=baseline)
            result = recovery.validate_records(partial, context)
            self.assertEqual(result[0]['terms'], 36884)
            self.assertEqual(result[-1]['full_shared']['valid_cosines'], 10)
            # Deliberately use a co-GT/self negative: rejected before aggregates.
            terms[0]['negative_label'] = terms[0]['true_label']
            bad = partial / 'peer_loss_terms_snapshot.csv'
            with bad.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(terms[0]))
                writer.writeheader(); writer.writerows(terms)
            with self.assertRaises(RuntimeError):
                recovery.validate_records(partial, context)


if __name__ == '__main__':
    unittest.main(verbosity=2)
