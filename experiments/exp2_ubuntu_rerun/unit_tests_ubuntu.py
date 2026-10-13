"""CPU tests of original-loop identity, gates and paired report adaptation."""
import argparse
import ast
import contextlib
import csv
import io
import platform
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from r2lib import *
import r2lib
import compare_ubuntu
import diagnostics

class ProtocolTests(unittest.TestCase):
    @contextlib.contextmanager
    def full_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); cfg = formal_config(123)
            identity = {'seed': 123, 'inputs': {'rerun_code_sha256': code_hashes()}}
            meta = {'identity': identity, 'seed': 123, 'phase': 'full', 'config': cfg,
                'optimizer_steps': 10856, 'test_used': True, 'test_examples': 5427,
                'train_examples': 43410, 'validation_examples': 5426,
                'checkpoint_reload_logits_match': True, 'finite_training_passed': True,
                'backbone_gradient_passed': True, 'gat_gradient_passed': True, 'best_epoch': 4}
            raw = {'tensors_compared': 197, 'pooler_present': False,
                'legacy_LayerNorm_aliases_verified': [str(i) for i in range(50)],
                'trained_checkpoint_loaded': False, 'smoke_weights_loaded': False}
            def seal():
                write(path / 'config.json', cfg); write(path / 'experiment_metadata.json', meta)
                write(path / 'raw_initialization.json', raw)
                for e in range(1, 5): write(path / f'validation_epoch_{e}.json', {'macro_f1': e/10})
                write(path / 'SUCCESS.json', {'status': FULL_STATUS, 'seed': 123,
                    'identity': identity, 'artifacts_sha256': artifacts(path)})
            seal()
            yield path, meta, raw, seal
    def test_full_completion_schema(self):
        with self.full_fixture() as (path, _, _, _): verify_full(path, 123)
    def test_full_wrong_seed_rejected(self):
        with self.full_fixture() as (path, _, _, _):
            with self.assertRaises(RuntimeError): verify_full(path, 2026)
    def test_full_wrong_checkpoint_rejected(self):
        with self.full_fixture() as (path, meta, _, seal):
            meta['best_epoch'] = 3; seal()
            with self.assertRaises(RuntimeError): verify_full(path, 123)
    def test_full_pooler_drift_rejected(self):
        with self.full_fixture() as (path, _, raw, seal):
            raw['pooler_present'] = True; seal()
            with self.assertRaises(RuntimeError): verify_full(path, 123)
    def test_frozen_package_pins(self):
        validate_sources()
    def test_only_seed_changes_exp2_config(self):
        cfg = read(FROZEN / 'lgsa.json')
        for seed in SEEDS:
            self.assertEqual(formal_config(seed), dict(cfg, seed=seed))
            self.assertEqual(formal_config(seed)['total_max_length'], 54)
    def test_unsupported_seed_rejected(self):
        with self.assertRaises(RuntimeError): formal_config(7)
    def test_original_training_ast_has_zero_changes(self):
        source = (ROOT / 'goemotions_adaptation/run.py').read_text()
        fn, proof = compile_original(source, {})
        self.assertEqual(proof['training_ast_changes'], 0)
        original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        self.assertEqual(proof['original_train_ast_sha256'], hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest())
        self.assertIn(32, fn.__code__.co_consts)
        self.assertIn(33, fn.__code__.co_consts)
        self.assertIn(-35, fn.__code__.co_consts)
        self.assertIn(-3, fn.__code__.co_consts)
        self.assertNotIn(56, fn.__code__.co_consts)
    def test_formal_configs_match_exp4b_except_derived_length(self):
        for seed in SEEDS:
            old = formal_config(seed); new = dict(old, total_max_length=68)
            self.assertEqual([k for k in old if old[k] != new[k]], ['total_max_length'])
    def test_raw_guard_no_pooler_schema(self):
        from preflight import expected_lgsa_backbone_keys
        keys = expected_lgsa_backbone_keys()
        self.assertEqual(len(keys), 197)
        self.assertFalse(any('pooler' in k for k in keys))
    def test_runtime_seed_guard_is_genuine(self):
        for seed in SEEDS:
            self.assertIn(str(seed), seed_environment(seed).__code__.co_consts)
    def test_report_math_unchanged_except_filename_status(self):
        transformed = compare_ubuntu.reporting_tree()
        for node in ast.walk(transformed):
            if isinstance(node, ast.Constant) and node.value == compare_ubuntu.REPORT_NAME:
                node.value = 'EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY.md'
            elif isinstance(node, ast.Constant) and node.value == COMPLETE:
                node.value = 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE'
        self.assertEqual(ast.dump(transformed), ast.dump(ast.parse(inspect_source(diagnostics.run))))
    def test_zero_denominator_remains_undefined(self):
        self.assertIsNone(diagnostics.ratio(3, 0))
        self.assertEqual(diagnostics.ratio(3, 6), .5)
    def test_std_sample_not_population(self):
        self.assertEqual(diagnostics.mean_std([1, 2, 3]), (2., 1.))
    def test_28_output_only_and_weighted_metric(self):
        y = np.zeros((5, 28), dtype=int); y[:, 0] = 1
        z = np.full((5, 28), -10., dtype=np.float32); z[:, 0] = 10
        score, rows = metrics(y, probabilities(z), .30)
        self.assertEqual(score['weighted_f1'], 1.)
        self.assertEqual(len(rows), 28)
        with self.assertRaises(RuntimeError): metrics(np.zeros((5, 32)), np.zeros((5, 32)), .30)
    def test_hash_gate_rejects_changed_completed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); (path / 'file.txt').write_text('original')
            write(path / 'SUCCESS.json', {'status': 'PASS', 'artifacts_sha256': artifacts(path)})
            verify_run(path, 'PASS')
            (path / 'file.txt').write_text('changed')
            with self.assertRaises(RuntimeError): verify_run(path, 'PASS')
    def test_end_to_end_same_host_report_from_saved_fixtures(self):
        from hierarchy_adapter import load_hierarchy
        from preflight import canonical_records
        global REAL_TREE
        h = load_hierarchy(REAL_TREE)
        raw = canonical_records(REPO / '数据集/GoEmotions/processed/test.tsv')
        y = np.zeros((5427, 28), dtype=int)
        for i, r in enumerate(raw): y[i, r['labels']] = 1
        ids = np.array([r['id'] for r in raw])
        env = {'hostname': 'D3020-A3', 'torch': '2.5.1+cu118', 'cuda': '11.8', 'transformers': '4.44.2',
            'numpy': '1.26.4', 'torch_geometric': '2.6.1', 'gpu': 'RTX4090D',
            'deterministic_algorithms': False, 'matmul_tf32': False, 'cudnn_tf32': True}
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / 'work'; work.mkdir()
            oldpaths, newpaths, hpcpaths = {}, {}, {}
            for seed in SEEDS:
                oldpaths[seed], newpaths[seed], hpcpaths[seed] = work / f'baselines/seed{seed}', Path(temp) / f'b4/seed{seed}', Path(temp) / f'hpc/seed{seed}'
                for name, path in [('old', oldpaths[seed]), ('new', newpaths[seed]), ('hpc', hpcpaths[seed])]:
                    path.mkdir(parents=True)
                    z = np.random.default_rng(seed).normal(size=(5427, 28)).astype(np.float32)
                    np.savez_compressed(path / 'test_predictions.npz', gold=y, logits=z, ids=ids)
                    cfg = formal_config(seed)
                    if name == 'new': cfg = dict(cfg, total_max_length=68)
                    write(path / 'config.json', cfg)
                    write(path / 'environment.json', env)
                    write(path / 'experiment_metadata.json', {'seed': seed, 'config': cfg})
                    write(path / 'test_fixed_0.30_complete_metrics.json', metrics(y, probabilities(z), .30)[0])
                    if name != 'hpc':
                        with (path / 'steps.jsonl').open('w') as f:
                            import json
                            for step in range(1, 10857):
                                alpha = min(1., (step-1)/int(10856*cfg['hga_ramp_ratio']))
                                f.write(json.dumps({'step': step, 'epoch': (step-1)//2714+1,
                                    'classification': 1., 'mlm': 2., 'hga': 3., 'alpha': alpha, 'loss': 3+alpha*3})+'\n')
                    write(path / 'SUCCESS.json', {'status': 'SYNTHETIC_ONLY', 'seed': seed,
                        'artifacts_sha256': artifacts(path)})
            fake_multi_baselines = Path(temp) / 'hpc'
            def fake_old_audit(seed, repo, root, tree, baseline):
                return h, {}, dict(formal_config(seed), total_max_length=68), {}, {}, {'fake_fixture_only': True}
            def fake_verified(path, seed, manifest=None):
                return {}, read(Path(path) / 'experiment_metadata.json')
            old_inputs = {p: sha(p) for path in [*oldpaths.values(), *newpaths.values(), *hpcpaths.values()]
                for p in path.rglob('*') if p.is_file()}
            with patch.object(compare_ubuntu, 'B4_RUNS', newpaths), patch.object(compare_ubuntu.multi, 'BASELINES', fake_multi_baselines), \
                 patch.object(compare_ubuntu, 'audit', return_value=({}, {}, {})), patch.object(compare_ubuntu, 'seed_audit', side_effect=fake_old_audit), \
                 patch.object(compare_ubuntu, 'verify_full', side_effect=fake_verified), patch.object(compare_ubuntu.multi, 'verify_candidate', side_effect=fake_verified), \
                 patch.dict(diagnostics.run.__globals__, {'REPO': REPO, 'PILOT_RUN': newpaths[42], 'verify_candidate': fake_verified}), \
                 patch.object(compare_ubuntu, 'validate_sources', return_value={}):
                # Copy the real tree into a read-only fixture repo namespace, not the real repo.
                import shutil
                fixture_repo = Path(temp) / 'repo'
                canonical = fixture_repo / '数据集/GoEmotions/processed'
                canonical.mkdir(parents=True)
                shutil.copy2(REPO / '数据集/GoEmotions/processed/train.tsv', canonical / 'train.tsv')
                tree_path = fixture_repo / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json'
                tree_path.parent.mkdir(parents=True); shutil.copy2(REAL_TREE, tree_path)
                with patch.object(compare_ubuntu, 'REPO', fixture_repo), contextlib.redirect_stdout(io.StringIO()):
                    compare_ubuntu.run(work, work / 'comparison')
            out = work / 'comparison'
            verify_run(out, COMPLETE)
            with (out / 'label_diagnostics_by_seed.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 84)
            with (out / 'loss_exp2_ubuntu_by_step.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 32568)
            text = (out / compare_ubuntu.REPORT_NAME).read_text()
            self.assertIn('均来自同一台', text)
            self.assertNotIn('Exp2来自历史HPC', text)
            provenance = read(out / 'input_provenance.json')
            self.assertEqual(provenance['exp2_seeds_retrained'], list(SEEDS))
            self.assertEqual(provenance['exp4b_seeds_retrained'], [])
            self.assertTrue(all(sha(p) == wanted for p, wanted in old_inputs.items()))

def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hierarchy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); REAL_TREE = a.hierarchy.resolve()
    require(not a.out.exists(), 'Never overwrite unit evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    write(a.out, {'status': 'EXP2_UBUNTU_UNIT_PASS' if result.wasSuccessful() else 'EXP2_UBUNTU_UNIT_FAIL',
        'system': platform.system(), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'code_sha256': code_hashes(), 'scope': 'CPU synthetic fixtures, original function AST and frozen pins; no GPU training'})
    require(result.wasSuccessful(), 'Unit tests failed')
    print('EXP2_UBUNTU_UNIT_PASS', result.testsRun)
