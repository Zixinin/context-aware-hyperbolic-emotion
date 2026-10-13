"""CPU extension regression tests. Never train or touch existing result files."""
import argparse
import ast
import json
import platform
import tempfile
import unittest
import contextlib
import io
import shutil
import csv
from unittest.mock import patch
from pathlib import Path
import numpy as np
from m4lib import *
from diagnostics import (ratio, ranks, correlation, mean_std, harmonic, attribution,
    paths, paired_labels, group_rows, loss_trace, render_summary)
from seed_extension import (derive_main, function_tree, paired_baseline, seed_environment,
    pilot_driver)
from hierarchy_adapter import load_hierarchy

REPOSITORY = None
TREE = None

class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = load_hierarchy(TREE)
        cls.cfg = read(REPOSITORY / 'experiments/exp2_1_multiseed/frozen/lgsa.json')

    def test_pilot_hash_pin(self):
        validate_sources()
        self.assertEqual(len(package_hashes()), 15)

    def test_seed42_orchestration_ast_identity(self):
        self.assertEqual(ast.dump(derive_main(42)[0]), ast.dump(function_tree(pilot_driver.main)))

    def test_seed_123_2026_derivation_reverses_to_exact_pilot(self):
        for seed in (123, 2026):
            with self.subTest(seed=seed):
                tree, counts = derive_main(seed)
                self.assertEqual(counts['metadata_seed'], 2)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Dict):
                        for k, v in zip(node.keys, node.values):
                            if isinstance(k, ast.Constant) and k.value == 'seed' and isinstance(v, ast.Constant) and v.value == seed:
                                v.value = 42
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        node.value = node.value.replace(f'/seed{seed}', '/seed42').replace(f'hierarchy seed{seed}', 'hierarchy seed42')
                        if node.value == 'EXP4B_SEED_FULL_COMPLETE_VERIFIED':
                            node.value = 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED'
                self.assertEqual(ast.dump(tree), ast.dump(function_tree(pilot_driver.main)))

    def test_environment_uses_actual_hash_seed(self):
        for seed in SEEDS:
            fn = seed_environment(seed)
            self.assertIn(str(seed), fn.__code__.co_consts)
            if seed != 42:
                self.assertNotIn('42', fn.__code__.co_consts)

    def test_unsupported_seed_rejected(self):
        with self.assertRaises(RuntimeError):
            derive_main(7)

    def test_only_seed_changes_formal_config(self):
        original = dict(self.cfg, total_max_length=68)
        for seed in (123, 2026):
            candidate = dict(original, seed=seed)
            self.assertEqual([k for k in original if original[k] != candidate[k]], ['seed'])

    def test_original_loader_shuffles_by_requested_seed_and_epoch(self):
        import torch
        import random
        source = REPOSITORY / 'experiments/exp2_lgsa/LGSA-GoEmotions/goemotions_adaptation'
        from unit_tests import upstream_symbols
        # Explicitly load the frozen pilot test helper, never this extension's test file.
        scope = upstream_symbols(source / 'data_goemotions.py', {'TensorRows'}, {})
        state = {'seed': 42}
        scope.update(torch=torch, random=random, config=lambda: state)
        upstream_symbols(source / 'run.py', {'loader'}, scope)
        sequences = []
        for seed in SEEDS:
            state['seed'] = seed
            data = {'input_ids': torch.arange(40)}
            order = [int(x) for batch in scope['loader'](data, 8, epoch=2, shuffle=True) for x in batch['input_ids']]
            expected = list(range(40)); random.Random(seed + 2).shuffle(expected)
            self.assertEqual(order, expected)
            sequences.append(order)
        self.assertEqual(len({tuple(order) for order in sequences}), 3)

    def test_matched_baseline_seed123_and2026_full_provenance(self):
        import seed_extension as extension
        from preflight import DATASET
        from b4lib import signature
        for seed in (123, 2026):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temp:
                root, baselines = Path(temp) / 'runtime', Path(temp) / 'baselines'
                run = baselines / f'seed{seed}'
                (root / 'goemotions_adaptation').mkdir(parents=True)
                run.mkdir(parents=True)
                write(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json', self.cfg)
                cfg = dict(self.cfg, seed=seed)
                expected = read(REPOSITORY / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
                signatures = {}
                for split in ('validation', 'test'):
                    rs = [{'id': split + str(i), 'text': 'Synthetic only', 'labels': [0]} for i in range(SIZES[split])]
                    signatures[split] = signature(rs)
                    y = np.zeros((len(rs), 28), dtype=np.int64); y[:, 0] = 1
                    z = np.zeros_like(y, dtype=np.float32)
                    np.savez_compressed(run / (split + '_predictions.npz'), gold=y, logits=z, ids=np.array([r['id'] for r in rs]))
                write(run / 'test_fixed_0.30_metrics.json', metrics(y, probabilities(z), .30)[0])
                meta = {'status': 'COMPLETED', 'model': 'lgsa', 'seed': seed, 'phase': 'full',
                    'slurm_state': 'COMPLETED', 'slurm_exit_code': '0:0', 'native_success_sha256': 'synthetic',
                    'config': cfg, 'legacy_source_sha256': expected, 'lgsa_patch_commit': '209f83762dae1b354ba40e68b6952482ecf96208',
                    'hierarchy_sha256': 'synthetic_tree', 'raw_model_repo': 'google-bert/bert-base-cased',
                    'raw_model_revision': REVISION, 'dataset': DATASET['repository'], 'dataset_config': DATASET['configuration'],
                    'dataset_revision': DATASET['revision'], 'dataset_split_sizes': SIZES,
                    'num_labels': 28, 'label_names': NAMES, 'rare_ids': RARE, 'fixed_threshold': .30,
                    'threshold_candidates': cfg['threshold_candidates'], 'dataset_signatures': signatures,
                    'checkpoint_reload_verified': True, 'checkpoint_reload_evidence': 'synthetic', 'best_epoch': 4}
                write(run / 'experiment_metadata.json', meta)
                write(run / 'config.json', cfg)
                write(run / 'checkpoint_info.json', {'best_epoch': 4, 'selection': cfg['checkpoint_selection']})
                (run / 'validation_metrics_by_epoch.csv').write_text('epoch,validation_macro_f1\n1,.1\n2,.2\n3,.3\n4,.4\n')
                write(run / 'SUCCESS.json', {'status': 'NORMALIZED_RUN_PASS', 'model': 'lgsa', 'seed': seed,
                    'native_success_sha256': 'synthetic', 'artifacts_sha256': {p.name: sha(p) for p in run.iterdir() if p.is_file()}})
                manifest = {'hierarchy_sha256': 'synthetic_tree', 'dataset_signatures': signatures}
                with patch.object(extension, 'BASELINES', baselines):
                    proof = paired_baseline(seed, run, root, manifest, expected)
                    self.assertEqual(proof['paired_seed'], seed)
                    with self.assertRaises(RuntimeError):
                        paired_baseline(42, run, root, manifest, expected)
                    (run / 'config.json').write_text('{}')
                    with self.assertRaises(RuntimeError):
                        paired_baseline(seed, run, root, manifest, expected)

    def test_zero_denominator_is_not_zero_or_infinity(self):
        self.assertIsNone(ratio(0, 0)); self.assertIsNone(ratio(5, 0))
        self.assertEqual(ratio(0, 3), 0.0); self.assertEqual(ratio(6, 3), 2.0)

    def test_sample_std_ddof1(self):
        mean, std = mean_std([.1, .2, .3])
        self.assertAlmostEqual(mean, .2); self.assertAlmostEqual(std, .1)

    def test_missing_seed_cannot_summarize(self):
        with self.assertRaises(RuntimeError):
            mean_std([.1, .2])

    def test_spearman_average_ties(self):
        np.testing.assert_array_equal(ranks([10, 20, 20, 40]), [1, 2.5, 2.5, 4])
        self.assertAlmostEqual(correlation([1, 2, 3], [9, 8, 7], True), -1)

    def test_constant_correlation_unavailable(self):
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3], True))

    def test_depth_edges_and_nodes_explicit(self):
        chain = paths(self.h)
        self.assertEqual(chain['neutral'], ['ROOT', 'NEUTRAL', 'neutral'])
        self.assertEqual(len(chain['neutral']) - 1, 2)
        self.assertEqual(max(len(p)-1 for p in chain.values()), 9)

    def fixture_labels(self):
        y = np.zeros((5, 28), dtype=np.int64); y[:, 0] = 1; y[0, 16] = 1
        z0 = np.full((5, 28), -10, dtype=np.float32)
        z1 = z0.copy(); z1[:, 0] = 10
        ids = np.array([str(i) for i in range(5)])
        support = np.arange(1, 29)
        return y, z0, ids, y.copy(), z1, ids.copy(), support, self.h

    def test_label_count_metrics_and_fixed_threshold(self):
        a, b, rows = paired_labels(123, *self.fixture_labels())
        self.assertEqual(len(rows), 28); self.assertEqual(b['threshold'], .30)
        self.assertIsNone(rows[0]['prediction_count_ratio'])
        self.assertEqual(rows[0]['exp4b_predicted_positive_count'], 5)
        self.assertEqual(rows[0]['exp4b_precision'], 1)
        self.assertEqual(rows[0]['exp4b_recall'], 1)
        self.assertEqual(rows[16]['group'], 'rare')

    def test_id_order_mismatch_rejected(self):
        args = list(self.fixture_labels()); args[5] = args[5][::-1]
        with self.assertRaises(RuntimeError):
            paired_labels(123, *args)

    def test_internal_output_columns_rejected(self):
        y = np.zeros((2, 56)); ids = np.array(['a', 'b'])
        with self.assertRaises(RuntimeError):
            paired_labels(123, y, y, ids, y, y, ids, np.ones(28), self.h)

    def test_precision_recall_attribution_sums_to_delta(self):
        for p0, r0, p1, r1 in [(0, 0, .2, .5), (.8, .5, .9, .2), (.5, .6, .3, .8)]:
            a, b = attribution(p0, r0, p1, r1)
            self.assertAlmostEqual(a+b, harmonic(p1, r1)-harmonic(p0, r0))

    def write_losses(self, path, mutate=None):
        records = []
        for step in range(1, 10857):
            alpha = min(1.0, (step-1)/max(1, int(10856*self.cfg['hga_ramp_ratio'])))
            records.append({'step': step, 'epoch': (step-1)//2714+1, 'classification': 1.0,
                'mlm': 2.0, 'hga': 3.0, 'alpha': alpha,
                'loss': 3.0+self.cfg['gof_loss_weight']*alpha*3.0})
        if mutate: mutate(records)
        path.write_text('\n'.join(json.dumps(r) for r in records))

    def test_loss_decomposition_without_model_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'steps.jsonl'; self.write_losses(path)
            before = sha(path)
            records, proof = loss_trace(path, self.cfg)
            self.assertEqual(proof['status'], 'SAVED_COMPONENTS_VERIFIED')
            self.assertEqual(len(records), 10856); self.assertEqual(before, sha(path))
            self.assertEqual(records[0]['hierarchy_hga_weighted'], 0)
            self.assertEqual(records[-1]['total_loss'], 6)

    def test_inconsistent_saved_total_loss_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'steps.jsonl'; self.write_losses(path, lambda r: r[-1].update(loss=999))
            with self.assertRaises(RuntimeError): loss_trace(path, self.cfg)

    def test_missing_loss_component_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'steps.jsonl'; self.write_losses(path, lambda r: r[0].pop('classification'))
            records, proof = loss_trace(path, self.cfg)
            self.assertEqual(proof['status'], 'TOTAL_ONLY_COMPONENTS_NOT_SAVED')
            self.assertIsNone(records[0]['classification_loss'])

    def test_absent_loss_file_not_recomputed(self):
        records, proof = loss_trace(Path('/private/tmp/exp4b_absent_loss_fixture_file.jsonl'), self.cfg)
        self.assertEqual(records, []); self.assertEqual(proof['status'], 'NOT_SAVED')

    def test_incomplete_loss_step_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'steps.jsonl'; self.write_losses(path, lambda r: r.pop())
            with self.assertRaises(RuntimeError): loss_trace(path, self.cfg)

    def test_end_to_end_diagnostics_synthetic_three_seeds(self):
        import diagnostics as diagnostic
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            work = repo / 'results/exp4b/multiseed_fixture'
            work.mkdir(parents=True)
            target = repo / '数据集/GoEmotions/processed/train.tsv'
            target.parent.mkdir(parents=True)
            shutil.copy2(REPOSITORY / '数据集/GoEmotions/processed/train.tsv', target)
            tree = repo / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json'
            tree.parent.mkdir(parents=True)
            shutil.copy2(TREE, tree)
            base = repo / 'baselines'
            candidates = {seed: work / f'seed{seed}/full' for seed in SEEDS}
            cfg = dict(self.cfg, total_max_length=68)
            audits = {}
            for seed in SEEDS:
                y = np.zeros((5427, 28), dtype=np.int64); y[:, 0] = 1
                z = np.full(y.shape, -10, dtype=np.float32); z[:, 0] = 10
                ids = np.array([str(i) for i in range(len(y))])
                baseline = base / f'seed{seed}'
                baseline.mkdir(parents=True)
                np.savez_compressed(baseline / 'test_predictions.npz', gold=y, logits=z, ids=ids)
                audit = (None, None, None, None, {'recomputed_fixed_0_30_metrics': metrics(y, probabilities(z), .30)[0]}, {'fixture': seed})
                audits[seed] = audit
                candidate = candidates[seed]
                candidate.mkdir(parents=True)
                np.savez_compressed(candidate / 'test_predictions.npz', gold=y, logits=z, ids=ids)
                write(candidate / 'config.json', dict(cfg, seed=seed))
                write(candidate / 'experiment_metadata.json', {'phase': 'full', 'seed': seed, 'config': dict(cfg, seed=seed),
                    'optimizer_steps': 10856, 'test_used': True, 'test_examples': 5427})
                self.write_losses(candidate / 'steps.jsonl')
                status = 'EXP4B_SEED42_PILOT_COMPLETE_VERIFIED' if seed == 42 else 'EXP4B_SEED_FULL_COMPLETE_VERIFIED'
                write(candidate / 'SUCCESS.json', {'status': status, 'seed': seed, 'phase': 'full',
                    'artifacts_sha256': {p.name: sha(p) for p in candidate.iterdir() if p.is_file()}})
            out = work / 'diagnostics'
            before = {str(p): sha(p) for p in repo.rglob('*') if p.is_file()}
            with patch.object(diagnostic, 'REPO', repo), patch.object(diagnostic, 'BASELINES', base), \
                 patch.object(diagnostic, 'PILOT_RUN', candidates[42]), \
                 patch.object(diagnostic, 'candidate_paths', return_value=candidates), \
                 patch.object(diagnostic, 'seed_audit', side_effect=lambda seed, *args: audits[seed]), \
                 contextlib.redirect_stdout(io.StringIO()):
                diagnostic.run(work, out)
            require(all(sha(Path(p)) == s for p, s in before.items()), 'Fixture inputs were modified')
            with (out / 'label_diagnostics_by_seed.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 84)
            with (out / 'label_diagnostics_mean_std.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 28)
            with (out / 'loss_trace_by_step.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3*10856)
            report = (out / 'EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY.md').read_text()
            for question in ('### 1.', '### 2.', '### 3.', '### 4.', '### 5.'):
                self.assertIn(question, report)
            self.assertIn('不是三 seed 一致下降', report)
            self.assertNotIn('EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE', report)
            verify_run(out, 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE')

def main():
    global REPOSITORY, TREE
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--hierarchy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    REPOSITORY, TREE = a.repo.resolve(), a.hierarchy.resolve()
    require(not a.out.exists(), 'Never overwrite unit evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionTests))
    write(a.out, {'status': 'EXP4B_EXTENSION_UNIT_PASS' if result.wasSuccessful() else 'EXP4B_EXTENSION_UNIT_FAIL',
        'system': platform.system(), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'extension_hashes': extension_hashes(), 'pilot_package_sha256': objsha(package_hashes()),
        'scope': 'CPU fixtures only; not real GPU training or three-seed results'})
    require(result.wasSuccessful(), 'Extension tests failed')
    print('EXP4B_EXTENSION_UNIT_PASS', result.testsRun)

if __name__ == '__main__':
    main()
