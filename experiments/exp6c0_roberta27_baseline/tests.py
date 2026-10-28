"""CPU synthetic tests only; no real backbone load or Mac deep-learning run."""
from __future__ import annotations

import copy
import contextlib
import csv
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from common import CONFIG, HERE, RARE, artifact_hashes, check_package, check_reference, read, seal, check_seal, write
from data27 import filter_records, load_dataset, targets
from metrics27 import probabilities, save_predictions, score, write_csv
from model27 import Roberta27, gradients, optimizer_for
from run import compare_reload, smoke, train, verify


NAMES = RARE + ["emotion_" + str(i) for i in range(20)]


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(10, 8)

    def forward(self, input_ids, attention_mask, return_dict):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DatasetTests(unittest.TestCase):
    def test_neutral_index_not_assumed_and_mixed_labels_survive(self):
        canonical = NAMES.copy()
        canonical.insert(4, "neutral")
        rows = [dict(id="single", text=" x ", labels=[0]),
                dict(id="multi", text="y", labels=[2, 5]),
                dict(id="only", text="z", labels=[4]),
                dict(id="mixed", text="u", labels=[4, 6])]
        before = copy.deepcopy(rows)
        kept, excluded, mapping = filter_records(rows, canonical)
        self.assertEqual([r["id"] for r in kept], ["single", "multi", "mixed"])
        self.assertEqual(kept[1]["labels"], [2, 4])
        self.assertEqual(kept[2]["labels"], [5])
        self.assertEqual(excluded[0]["id"], "only")
        self.assertEqual(rows, before)
        self.assertEqual(kept[0]["text"], " x ")
        np.testing.assert_array_equal(targets(kept).sum(1), [1, 2, 1])

    def test_invalid_labels_refused(self):
        for labels in [[], [0, 0], [28], [-1]]:
            with self.subTest(labels=labels), self.assertRaises(RuntimeError):
                filter_records([dict(id="x", text="x", labels=labels)], NAMES + ["neutral"])

    def test_official_counts_and_mapping(self):
        rows, manifest = load_dataset()
        self.assertEqual([len(rows[s]) for s in rows], [30587, 3834, 3821])
        self.assertEqual([manifest['splits'][s]['retained_neutral_plus_emotion'] for s in rows], [1396, 174, 181])
        self.assertEqual([manifest['splits'][s]['multi_emotion'] for s in rows], [5767, 712, 667])
        self.assertEqual(math.ceil(len(rows['train']) / 16) * 4, 7648)
        self.assertEqual(math.ceil(7648 * 0.1), 765)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gold = np.zeros((4, 27), dtype=np.float32)
        self.gold[0, 0] = 1
        self.gold[1:, 1] = 1
        self.probs = np.zeros((4, 27), dtype=np.float64)
        self.probs[:3, 0] = .30
        self.probs[0, 1] = .9
        self.probs[1, 1] = .9

    def test_inclusive_threshold_correct_precision_recall_direction(self):
        metrics, rows, rare, pred = score(self.gold, self.probs, NAMES)
        self.assertEqual(pred[:, 0].sum(), 3)
        self.assertEqual(rows[0]['tp'], 1)
        self.assertEqual(rows[0]['fp'], 2)
        self.assertEqual(rows[0]['fn'], 0)
        self.assertAlmostEqual(rows[0]['precision'], 1 / 3)
        self.assertEqual(rows[0]['recall'], 1)
        self.assertAlmostEqual(metrics['micro_precision'], 2 / 5)
        self.assertAlmostEqual(metrics['micro_recall'], 2 / 4)
        self.assertAlmostEqual(rare['rare_f1'], sum(r['f1'] for r in rows[:7]) / 7)
        self.assertEqual(rows[2]['precision'], 0)
        self.assertEqual(metrics['zero_prediction_examples'], 1)

    def test_stable_sigmoid_and_invalid_inputs(self):
        np.testing.assert_array_equal(probabilities(np.array([-1000., 0., 1000.])), [0, .5, 1])
        with self.assertRaises(RuntimeError):
            probabilities(np.array([np.nan]))
        with self.assertRaises(RuntimeError):
            score(self.gold[:, :26], self.probs[:, :26], NAMES)
        with self.assertRaises(RuntimeError):
            score(self.gold + .1, self.probs, NAMES)

    def test_reload_checks_all_examples_ids_targets_and_predictions(self):
        logits = np.zeros((4, 27), dtype=np.float32)
        ids = np.array(['a', 'b', 'c', 'd'])
        before = (logits, self.gold, ids)
        self.assertEqual(compare_reload(before, before, NAMES)['num_examples'], 4)
        altered = logits.copy()
        altered[-1, 0] = 1
        with self.assertRaises(AssertionError):
            compare_reload(before, (altered, self.gold, ids), NAMES)
        with self.assertRaises(RuntimeError):
            compare_reload(before, (logits, self.gold, ids[::-1]), NAMES)


class ModelTests(unittest.TestCase):
    def test_eval_is_linear_first_token_and_bce_mean(self):
        model = Roberta27(FakeEncoder()).eval()
        ids, mask = torch.tensor([[1, 2, 3], [3, 4, 5]]), torch.ones(2, 3, dtype=torch.long)
        labels = torch.zeros(2, 27)
        labels[0, [1, 2]] = 1
        labels[1, 3] = 1
        out = model(ids, mask, labels)
        h = model.encoder.embedding(ids)[:, 0]
        expected = nn.functional.linear(h, model.classifier_weight, model.classifier_bias)
        torch.testing.assert_close(out['logits'], expected)
        torch.testing.assert_close(out['loss'], nn.functional.binary_cross_entropy_with_logits(expected, labels))
        out['loss'].backward()
        self.assertGreater(gradients(model)['encoder']['norm'], 0)
        self.assertGreater(gradients(model)['classifier']['norm'], 0)
        self.assertEqual(set(dict(model.named_parameters())), {'encoder.embedding.weight','classifier_weight','classifier_bias'})
        altered = ids.clone()
        altered[:, 1:] = 8
        torch.testing.assert_close(out['logits'], model(altered, mask)['logits'])

    def test_dropout_matches_existing_fyp_cls_and_optimizer_groups(self):
        model = Roberta27(FakeEncoder()).train()
        ids, mask = torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long)
        torch.manual_seed(42)
        out = model(ids, mask)['logits']
        torch.manual_seed(42)
        h = model.encoder.embedding(ids)[:, 0]
        expanded = h[:, None].expand(-1, 27, -1)
        expected = (nn.functional.dropout(expanded, .1, training=True) * model.classifier_weight[None]).sum(-1) + model.classifier_bias
        torch.testing.assert_close(out, expected)
        optimizer, groups = optimizer_for(model)
        self.assertIn('classifier_bias', groups[1]['parameter_names'])
        self.assertIn('classifier_weight', groups[0]['parameter_names'])
        self.assertEqual(optimizer.defaults['betas'], (.9, .999))
        self.assertEqual(optimizer.defaults['eps'], 1e-8)


class IntegrityTests(unittest.TestCase):
    def test_inspection_imports_without_site_packages_or_numpy(self):
        result = subprocess.run([sys.executable, '-S', str(HERE/'inspect_ubuntu.py'), '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Standard-library environment inspection', result.stdout)
        # Execute the actual inspection main under -S, with only OS commands mocked.
        # Both environment candidates are deliberately absent; it must reach the
        # controlled stop after printing snapshot/environment evidence, without NumPy.
        code = '''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import inspect_ubuntu as inspect
with patch.object(inspect.platform, 'node', return_value='D3020-A3'), patch.object(inspect.platform, 'system', return_value='Linux'), patch.object(inspect, 'REPO', Path('/home/zixinfeng/Documents/context-aware-hyperbolic-emotion')), patch.object(inspect, 'check_package', return_value={}), patch.object(inspect, 'check_reference', return_value={}), patch.object(inspect, 'find_snapshots', return_value=[]), patch.object(inspect.subprocess, 'run'), patch.object(Path, 'is_file', return_value=False):
    try:
        inspect.main()
    except RuntimeError as e:
        assert 'No existing candidate environment with NumPy' in str(e), str(e)
        print('STDLIB_INSPECTION_CONTROLLED_STOP_PASS')
    else:
        raise AssertionError('Expected environment gate')
assert 'numpy' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-S', '-c', code, str(HERE)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STDLIB_INSPECTION_CONTROLLED_STOP_PASS', result.stdout)

    def test_manifests_and_references(self):
        self.assertTrue(check_package())
        self.assertTrue(check_reference())

    def test_seal_tamper_and_log_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'metrics.json', {'x': 1})
            seal(root, 'SUCCESS.json', {'status': 'PASS'})
            (root / 'train.log').write_text('new log')
            (root / 'train_exit_code.txt').write_text('0')
            check_seal(root, 'SUCCESS.json')
            with self.assertRaises(RuntimeError):
                seal(root, 'SUCCESS.json', {})
            write(root / 'metrics.json', {'x': 2})
            with self.assertRaises(RuntimeError):
                check_seal(root, 'SUCCESS.json')


class FullVerifierTests(unittest.TestCase):
    """Synthetic complete artifact fixture exercises no-training verification and refusal gates."""
    def make_fixture(self, root):
        data = {s: [dict(id=s+'a', text='a', labels=[0]), dict(id=s+'b', text='b', labels=[1, 2])]
                for s in ['train','validation','test']}
        manifest = dict(names27=NAMES)
        package = {'x': 'sha'}
        write(root/'plan.json', dict(config=CONFIG, package_sha256=package, snapshot='/fake', snapshot_files_sha256={}))
        write(root/'config.json', CONFIG)
        write(root/'dataset_manifest.json', manifest)
        write(root/'DATASET_MANIFEST.json', manifest)
        write(root/'protected_before.json', {})
        (root/'best_checkpoint').mkdir()
        (root/'best_checkpoint/model.pt').write_bytes(b'fake-checkpoint')
        from common import sha
        digest = sha(root/'best_checkpoint/model.pt')
        rare = {}
        logits = np.full((2,27), -5, dtype=np.float32)
        logits[0,0] = 5
        logits[1,[1,2]] = 5
        for split in ['validation','test']:
            metrics, rare[split] = save_predictions(root, split, logits, targets(data[split]),
                                                   [r['id'] for r in data[split]], NAMES)
        write(root/'rare_label_metrics.json', rare)
        write(root/'checkpoint_selection.json', dict(epoch=2, global_step=2, validation_macro_f1=metrics['macro_f1'],
            threshold=.30, selection_split='validation', checkpoint_sha256=digest))
        history = [dict(epoch=i, global_step=i, training_examples=2, training_loss=.1, validation_loss=.1,
                        validation_macro_f1=0 if i == 1 else metrics['macro_f1']) for i in range(1,5)]
        write_csv(root/'training_log.csv', history)
        write_csv(root/'batch_log.csv', [dict(step=i, loss=.1, unclipped_grad_norm=.1, lr_used=0, lr_after=0) for i in range(1,5)])
        write(root/'optimizer_protocol.json', dict(total_steps=4, warmup_steps=1))
        for epoch in range(1,5):
            write(root/f'epoch{epoch}_first_batch.json', dict(gradients={g:dict(norm=1) for g in ['encoder','classifier']}))
        write(root/'checkpoint_reload_verification.json', dict(num_examples=2, threshold_predictions_equal=True,
            metrics_equal=True, ids_equal=True, targets_equal=True))
        write(root/'FORMAL_TEST_CLAIM.json', dict(call=1, threshold=.30, checkpoint_sha256=digest))
        write(root/'environment.json', dict(completed_epochs=4, completed_optimizer_steps=4,
            formal_test_inference_calls=1, existing_files_unchanged=True))
        return data, manifest, package

    def test_complete_fixture_and_corruption_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, manifest, package = self.make_fixture(root)
            with patch('run.run_path', return_value=root), patch('run.check_package', return_value=package), \
                    patch('run.check_reference'), patch('run.load_dataset', return_value=(data,manifest)), \
                    patch('run.protected_inventory', return_value={}), patch('run.snapshot_inventory', return_value={}):
                verify(root, sealed=False)
                selection = read(root/'checkpoint_selection.json')
                write(root/'checkpoint_selection.json', dict(selection, epoch=3, global_step=3))
                with self.assertRaisesRegex(RuntimeError, 'Selection mismatch'):
                    verify(root, sealed=False)
                write(root/'checkpoint_selection.json', selection)
                env = read(root/'environment.json')
                write(root/'environment.json', dict(env, formal_test_inference_calls=2))
                with self.assertRaisesRegex(RuntimeError, 'runtime proof'):
                    verify(root, sealed=False)
                write(root/'environment.json', env)
                claim = read(root/'FORMAL_TEST_CLAIM.json')
                write(root/'FORMAL_TEST_CLAIM.json', dict(claim, threshold=.5))
                with self.assertRaisesRegex(RuntimeError, 'test claim'):
                    verify(root, sealed=False)


class SyntheticFlowTests(unittest.TestCase):
    def test_complete_smoke_train_reload_single_test_seal_flow_with_toy_stubs(self):
        """Run the orchestration with a tiny CPU toy, never a real backbone or GPU."""
        class ToyConfig:
            hidden_size = 8
            def to_dict(self):
                return {'hidden_size': 8, 'id2label': {0: 'label0'}}
            def save_pretrained(self, destination):
                write(Path(destination)/'config.json', self.to_dict())
        class ToyEncoder(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.config = ToyConfig()
        class ToyModel(Roberta27):
            def to(self, *args, **kwargs):
                return super().to('cpu')
        class ToyTokenizer:
            vocab_size = 10
            bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
            init_kwargs = {'toy': True}
            def __call__(self, texts, **kwargs):
                return dict(input_ids=torch.tensor([[0,3,2] for _ in texts]),
                            attention_mask=torch.ones(len(texts),3,dtype=torch.long))
            def save_pretrained(self, destination):
                write(Path(destination)/'tokenizer_config.json', self.init_kwargs)
        tokenizer = ToyTokenizer()
        def toy_load(snapshot):
            return ToyModel(ToyEncoder()), tokenizer, {'missing_keys': [], 'unexpected_keys': []}
        data = {s: [dict(id=s+str(i), text=str(i), labels=[0] if i < 8 else [1,2]) for i in range(16)]
                for s in ['train','validation','test']}
        manifest = dict(protocol='synthetic test only', names27=NAMES,
            mapping=[dict(new_index=i, original_index=i, emotion=n) for i,n in enumerate(NAMES)],
            text_overlap_note='synthetic', text_overlap_audit={}, splits={s: dict(
                original_count=16, retained_count=16, removed_neutral_only=0, single_emotion=8,
                multi_emotion=8, multi_fraction=.5, mean_positive_labels=1.5, retained_neutral_plus_emotion=0,
                positive_support=targets(rows).sum(0).astype(int).tolist()) for s,rows in data.items()})
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu', *args[1:])
            return original_to(tensor, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run_synthetic'
            root.mkdir()
            snapshot = Path(tmp)/('a'*40)
            snapshot.mkdir()
            write(snapshot/'config.json', {'synthetic': True})
            (snapshot/'model.safetensors').write_bytes(b'not-real-weights')
            with patch('run.run_path', side_effect=lambda x:Path(x)), \
                    patch('run.runtime', return_value={'gpu':'synthetic CPU toy, not actual GPU'}), \
                    patch('run.protected_inventory', return_value={}), \
                    patch('run.find_snapshots', return_value=[str(snapshot)]), \
                    patch('run.load_dataset', return_value=(data,manifest)), \
                    patch('model27.load_model', side_effect=toy_load), \
                    patch('torch.Tensor.to', new=cpu_to), \
                    patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                    contextlib.redirect_stdout(io.StringIO()) as flow_output:
                # A nonempty protected inventory is required, so use a stable synthetic entry.
                with patch('run.protected_inventory', return_value={'synthetic_protected':'sha'}):
                    smoke(root)
                    self.assertFalse((root/'FORMAL_TEST_CLAIM.json').exists())
                    train(root)
                    verify(root)
                    self.assertEqual(read(root/'environment.json')['formal_test_inference_calls'],1)
                    self.assertEqual(read(root/'checkpoint_selection.json')['epoch'],1)
                    with self.assertRaises(FileExistsError):
                        train(root)
                    self.assertEqual(flow_output.getvalue().count('EXP6C0_ROBERTA27_MULTILABEL_BASELINE_VERIFIED'), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
