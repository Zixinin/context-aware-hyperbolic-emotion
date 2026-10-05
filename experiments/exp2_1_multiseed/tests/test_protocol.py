"""CPU-only synthetic unit checks; never research results or pretrained training."""
import ast
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mslib import config, NAMES, SEEDS, RARE, CANDIDATES, METRICS, read, write, csvwrite, ROOT
from metrics_ms import score, threshold_search, probabilities, load_predictions
from report_ms import stats, generate


class ProtocolTests(unittest.TestCase):
    def test_only_seed_changes(self):
        for model in ['bert', 'lgsa']:
            base = config(model, 42)
            for seed in SEEDS:
                c = config(model, seed)
                self.assertEqual(c.pop('seed'), seed)
                self.assertEqual(c, {k: v for k, v in base.items() if k != 'seed'})

    def test_threshold_ties_choose_lowest(self):
        y = np.ones((3, 28), dtype=int)
        z = np.full((3, 28), 10, dtype=np.float32)
        for model in ['bert', 'lgsa']:
            chosen, rows = threshold_search(y, z, model)
            self.assertEqual(chosen, .05)
            self.assertEqual([r['threshold'] for r in rows], CANDIDATES)

    def test_counts_zero_division_and_rare_mean(self):
        y = np.zeros((3, 28), dtype=int)
        y[0, 16] = y[1, 16] = 1
        p = np.zeros_like(y, dtype=float)
        p[0, 16] = p[2, 16] = .3
        m, rows = score(y, p, .3)
        self.assertEqual([rows[16][k] for k in ['tp', 'fp', 'fn', 'support', 'number_predicted_positive']], [1, 1, 1, 2, 2])
        self.assertEqual(rows[16]['f1'], .5)
        self.assertAlmostEqual(m['rare_f1'], .5/7)
        self.assertEqual(rows[0]['f1'], 0)
        with self.assertRaises(RuntimeError):
            score(y, p[:, :27], .3)

    def test_sample_sd_and_pairing(self):
        self.assertEqual(stats([1, 2, 3])['std'], 1)
        with self.assertRaises(RuntimeError):
            stats([1, 2])

    def test_preserve_sigmoid_precision(self):
        z = np.array([[.1, -2]], dtype=np.float32)
        self.assertEqual(probabilities(z, 'bert').dtype, np.float64)
        self.assertEqual(probabilities(z, 'lgsa').dtype, np.float32)

    def test_reject_duplicate_prediction_ids(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'bad.npz'
            np.savez(p, ids=['same', 'same'], gold=np.zeros((2,28)), logits=np.zeros((2,28)))
            with self.assertRaises(RuntimeError):
                load_predictions(p)

    def test_no_test_input_in_threshold_api(self):
        import inspect
        self.assertEqual(list(inspect.signature(threshold_search).parameters), ['gold', 'logits', 'model'])
        code = (ROOT/'normalize_results.py').read_text()
        self.assertIn('threshold_search(val_y, val_z, model)', code)

    def test_original_lgsa_loader_receives_each_seed(self):
        import torch
        path = ROOT.parent/'exp2_lgsa/code/goemotions_adaptation/run.py'
        if not path.exists():
            from mslib import PROJECTS
            path = PROJECTS['lgsa']/'goemotions_adaptation/run.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loader')
        current = {'seed': 42}
        ns = {'random': random, 'torch': torch, 'config': lambda: current, 'TensorRows': lambda data, order: order}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), ns)
        sequences = []
        for seed in SEEDS:
            current['seed'] = seed
            batches = ns['loader']({'input_ids': list(range(31))}, 7, epoch=2, shuffle=True)
            actual = [x for batch in batches for x in batch.tolist()]
            expected = list(range(31)); random.Random(seed+2).shuffle(expected)
            self.assertEqual(actual, expected)
            sequences.append(tuple(actual))
        self.assertEqual(len(set(sequences)), 3)

    def test_raw_fingerprint_rejects_checkpoint_and_changed_tensor(self):
        import torch
        from safetensors.torch import save_file
        import run_model
        from mslib import REVISION, sha
        with tempfile.TemporaryDirectory() as t:
            base = Path(t); raw = base/REVISION; raw.mkdir()
            state = {'embeddings.word_embeddings.weight': torch.arange(12, dtype=torch.float32).reshape(3,4)}
            save_file({'bert.'+k: v for k,v in state.items()}, str(raw/'model.safetensors'))
            write(base/'assets/manifest.json', {'model_snapshot': str(raw), 'model_sha256': {'model.safetensors': sha(raw/'model.safetensors')}})
            with mock.patch.dict(run_model.PROJECTS, {'bert': base}):
                self.assertEqual(run_model.verify_raw(state, raw)['status'], 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED')
                with self.assertRaises(RuntimeError):
                    run_model.verify_raw({k:v+1 for k,v in state.items()}, raw)
                with self.assertRaises(RuntimeError):
                    run_model.verify_raw(state, base/'seed42_trained_checkpoint')

    def test_raw_fingerprint_accepts_only_exact_legacy_layernorm_tensors(self):
        import torch
        from safetensors.torch import save_file
        import run_model
        from mslib import REVISION, sha
        with tempfile.TemporaryDirectory() as t:
            base = Path(t); raw = base/REVISION; raw.mkdir()
            gamma = torch.arange(4, dtype=torch.float32)
            beta = -gamma
            save_file({'bert.embeddings.LayerNorm.gamma': gamma,
                       'bert.embeddings.LayerNorm.beta': beta}, str(raw/'model.safetensors'))
            write(base/'assets/manifest.json', {'model_snapshot': str(raw),
                  'model_sha256': {'model.safetensors': sha(raw/'model.safetensors')}})
            state = {'embeddings.LayerNorm.weight': gamma.clone(), 'embeddings.LayerNorm.bias': beta.clone()}
            with mock.patch.dict(run_model.PROJECTS, {'bert': base}):
                proof = run_model.verify_raw(state, raw)
                self.assertEqual(proof['status'], 'FRESH_RAW_CASED_INITIALIZATION_VERIFIED')
                self.assertEqual(len(proof['legacy_layernorm_keys_verified']), 2)
                with self.assertRaisesRegex(RuntimeError, 'Backbone not raw'):
                    run_model.verify_raw({**state, 'embeddings.LayerNorm.weight': gamma + 1}, raw)

    def test_accounting_gate_rejects_failed_or_missing_batch(self):
        import mslib
        with mock.patch.object(mslib, 'capture', return_value='123|COMPLETED|0:0|00:01:00||cpu=4'):
            with self.assertRaises(RuntimeError):
                mslib.sacct('123')
        with mock.patch.object(mslib, 'capture', return_value='123|COMPLETED|0:0|00:01:00||cpu=4\n123.batch|FAILED|1:0|00:01:00||cpu=4'):
            with self.assertRaises(RuntimeError):
                mslib.sacct('123')

    def test_error_log_gate_stops_on_oom(self):
        import mslib
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'error.log';p.write_text('RuntimeError: CUDA out of memory\n')
            with self.assertRaises(RuntimeError):
                mslib.check_logs([p])

    def test_integrity_checker_refuses_missing_seed(self):
        from scripts.verify_multiseed_results import verify
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'bert_baseline/seed42').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'Missing/unexpected'):
                verify(root)

    def test_gpu_only_submission_and_report_gate(self):
        submit = (ROOT/'submit.py').read_text()
        shell = (ROOT/'slurm/train_chain.sbatch').read_text()
        self.assertIn("'slurm/train_chain.sbatch'", submit)
        self.assertNotIn('finalize.sbatch', submit)
        self.assertNotIn('afterok:', submit)
        self.assertIn('#SBATCH --partition=gpu4090', shell)
        self.assertIn('#SBATCH --gres=gpu:1', shell)
        self.assertLess(shell.index('python -u chain.py'), shell.index('python -u finalize.py'))
        self.assertFalse((ROOT/'slurm/finalize.sbatch').exists())

    def test_preliminary_report_promotion_is_deterministic(self):
        import hashlib
        from status import BANNER, PRELIMINARY_END, FINAL_END, HEADING, final_summary_text, verify_preliminary_report
        source = HEADING + BANNER + 'Body\n\n' + PRELIMINARY_END + '\n'
        with tempfile.TemporaryDirectory() as t:
            report = Path(t)
            summary = report/'MULTISEED_SUMMARY.md'
            summary.write_text(source)
            manifest = {'artifacts_sha256': {'MULTISEED_SUMMARY.md': hashlib.sha256(source.encode()).hexdigest()}}
            verify_preliminary_report(report, manifest)
            summary.write_text(final_summary_text(source))
            self.assertNotIn(BANNER, summary.read_text())
            self.assertIn(FINAL_END, summary.read_text())
            verify_preliminary_report(report, manifest)
            summary.write_text(summary.read_text().replace(FINAL_END, 'tampered'))
            with self.assertRaises(RuntimeError):
                verify_preliminary_report(report, manifest)

    def test_synthetic_reporting_completeness(self):
        # This fixture never enters outputs/ and has no SUCCESS markers.
        with tempfile.TemporaryDirectory(prefix='fyp21-synthetic-unit-') as temp:
            d = Path(temp)
            records = []
            rng = np.random.default_rng(773)
            y = (rng.random((32,28)) < .12).astype(int)
            for model in ['bert', 'lgsa']:
                for seed in SEEDS:
                    path = d/f'{model}{seed}'; path.mkdir()
                    p = rng.random((32,28))
                    m, labels = score(y, p, .3)
                    for setting in ['fixed_0.30', 'validation_selected']:
                        write(path/f'test_{setting}_metrics.json', m)
                        csvwrite(path/f'per_label_{setting}_metrics.csv', labels)
                    csvwrite(path/'validation_metrics_by_epoch.csv', [{'epoch': e, 'validation_micro_f1': .1*e,
                             'validation_macro_f1': .1*e} for e in range(1,5)])
                    records.append({'model': model, 'seed': seed, 'path': str(path),
                        'metadata': {'seed42_decision': 'SYNTHETIC_UNIT_TEST', 'slurm_job_id': 'NONE',
                                     'best_epoch': 4, 'selected_threshold': .3}})
            out = d/'synthetic_report'
            result = generate(records, out)
            self.assertEqual(result['overall_rows'], 12)
            self.assertEqual(result['validation_curve_rows'], 24)
            self.assertEqual(result['plot_count'], 11)
            for name in ['overall_summary.csv', 'multiseed_mean_std.csv', 'paired_seed_deltas.csv',
                         'rare_label_seed_stability.csv', 'grief_relief_seed_analysis.csv',
                         'per_label_multiseed_comparison.csv', 'MULTISEED_SUMMARY.md']:
                self.assertTrue((out/name).is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
