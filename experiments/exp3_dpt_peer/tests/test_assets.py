"""CPU-only provenance and data preparation tests using synthetic fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import baseline
import prepare_ubuntu_assets as preparation
from protocol import read, sha, write


class Split(list):
    features = {'labels': types.SimpleNamespace(feature=types.SimpleNamespace(names=['a', 'b']))}


class AssetTests(unittest.TestCase):
    def test_signature_matches_historical_canonical_format(self):
        rows = [{'id': 'id1', 'text': '中文 test', 'labels': [2, 0]},
                {'id': 'id2', 'text': 'other', 'labels': [27]}]
        signature = baseline.dataset_signature(rows)
        y = np.zeros((2, 28), dtype=np.uint8)
        y[0, [2, 0]], y[1, 27] = 1, 1
        encoded = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')) + '\n' for r in rows).encode()
        self.assertEqual(signature['rows_sha256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(signature['gold_sha256'], hashlib.sha256(y.tobytes()).hexdigest())
        self.assertNotEqual(signature, baseline.dataset_signature(rows[::-1]))
        for invalid in ([], [{'id': 'x', 'text': '', 'labels': []}],
                        [{'id': 'x', 'text': '', 'labels': [0, 0]}], rows + rows):
            with self.assertRaises(RuntimeError):
                baseline.dataset_signature(invalid)

    def fixture_data(self):
        return {'train': Split([{'id': 't1', 'text': 'train', 'labels': [0]},
                                {'id': 't2', 'text': 'train2', 'labels': [1]}]),
                'validation': Split([{'id': 'v', 'text': 'validation', 'labels': [0]}]),
                'test': Split([{'id': 'x', 'text': 'test', 'labels': [1]}])}

    def prepare(self, data, target, signatures):
        with patch.object(preparation, 'SIZES', {'train': 2, 'validation': 1, 'test': 1}), \
             patch.object(preparation, 'RARE', list(range(2, 9))):
            return preparation.prepare_records(data, target, signatures, ['a', 'b'])

    def test_prep_writes_original_format_only_to_new_stage(self):
        data = self.fixture_data()
        signatures = {s: baseline.dataset_signature(rows) for s, rows in data.items()}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'assets'
            stage, observed, hashes, frequencies = self.prepare(data, target, signatures)
            self.assertFalse(target.exists())
            self.assertEqual(observed, signatures)
            self.assertEqual(frequencies[:2], [1, 1])
            self.assertEqual(hashes['train.jsonl'], sha(stage / 'train.jsonl'))
            self.assertEqual((stage / 'train.jsonl').read_text(),
                             ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data['train']))

    def test_prep_rejects_changed_text_and_retains_partial(self):
        data = self.fixture_data()
        signatures = {s: baseline.dataset_signature(rows) for s, rows in data.items()}
        data['train'][0]['text'] = 'changed'
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'assets'
            with self.assertRaisesRegex(RuntimeError, 'text/IDs/labels/order'):
                self.prepare(data, target, signatures)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_name('assets.exp3-preparing').is_dir())

    def test_prep_never_overwrites_and_rejects_split_leakage(self):
        data = self.fixture_data()
        signatures = {s: baseline.dataset_signature(rows) for s, rows in data.items()}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'assets'
            target.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                self.prepare(data, target, signatures)
        data['validation'][0]['id'] = 't1'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'leakage'):
                self.prepare(data, Path(tmp) / 'assets', signatures)


class NormalizedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.root, self.run = self.repo / 'lgsa', self.repo / 'run'
        self.run.mkdir()
        adapter = self.root / 'goemotions_adaptation'
        self.cfg = read(HERE.parents[1] / 'experiments/exp2_1_multiseed/frozen/lgsa.json')
        self.hierarchy = read(HERE.parents[1] / 'experiments/exp2_1_multiseed/frozen/hierarchy.json')
        write(adapter / 'lgsa_goemotions_seed42.json', self.cfg)
        write(adapter / 'hierarchy.json', self.hierarchy)
        self.expected = {p.name: sha(p) for p in adapter.iterdir()}
        write(self.repo / 'experiments/exp2_1_multiseed/frozen/expected_sources.json', {'lgsa': self.expected})
        write(self.run / 'config.json', self.cfg)
        write(self.run / 'checkpoint_info.json', {'best_epoch': 4, 'selection': self.cfg['checkpoint_selection']})
        (self.run / 'validation_metrics_by_epoch.csv').write_text(
            'epoch,validation_macro_f1\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n')
        self.signatures = {}
        for split in baseline.SIZES:
            rows = [{'id': 'test-' + split, 'text': split, 'labels': [0]}]
            file = self.root / 'assets/goemotions-v0' / (split + '.jsonl')
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.signatures[split] = {**baseline.dataset_signature(rows), 'size': baseline.SIZES[split]}
        for split in ('validation', 'test'):
            np.savez(self.run / (split + '_predictions.npz'), gold=np.ones((1, 28)),
                     logits=np.zeros((1, 28)), ids=np.array([split]))
        self.metadata = {'status': 'COMPLETED', 'model': 'lgsa', 'seed': 42, 'phase': 'full',
            'slurm_state': 'COMPLETED', 'slurm_exit_code': '0:0', 'native_success_sha256': 'historical-digest',
            'config': self.cfg, 'legacy_source_sha256': self.expected,
            'hierarchy_sha256': sha(adapter / 'hierarchy.json'), 'lgsa_patch_commit': baseline.PATCH,
            'raw_model_repo': 'google-bert/bert-base-cased', 'raw_model_revision': baseline.REVISION,
            'dataset': baseline.DATASET['repository'], 'dataset_config': 'simplified',
            'dataset_revision': baseline.DATASET['revision'], 'dataset_split_sizes': baseline.SIZES,
            'label_names': [r['fine_label'] for r in self.hierarchy['labels']], 'num_labels': 28,
            'rare_ids': baseline.RARE, 'fixed_threshold': .3, 'threshold_candidates': self.cfg['threshold_candidates'],
            'checkpoint_reload_verified': True, 'checkpoint_reload_evidence': 'historical reload verified',
            'best_epoch': 4, 'dataset_signatures': self.signatures}
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        (self.run / 'experiment_metadata.json').write_text(json.dumps(self.metadata))
        success = {'status': 'NORMALIZED_RUN_PASS', 'model': 'lgsa', 'seed': 42,
            'native_success_sha256': 'historical-digest',
            'artifacts_sha256': {p.name: sha(p) for p in self.run.iterdir() if p.name != 'SUCCESS.json'}}
        (self.run / 'SUCCESS.json').write_text(json.dumps(success))

    def check(self):
        with patch.object(baseline, 'REPO', self.repo):
            return baseline.normalized_metadata(self.run, self.root)

    def test_normalized_gate_accepts_matching_local_provenance(self):
        self.assertEqual(self.check(), self.metadata)

    def test_changed_artifact_rejected(self):
        (self.run / 'config.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
            self.check()

    def test_bad_accounting_protocol_source_and_reload_rejected(self):
        original = copy.deepcopy(self.metadata)
        for key, value in [('slurm_exit_code', '1:0'), ('raw_model_revision', 'main'),
                           ('dataset_revision', 'main'), ('checkpoint_reload_verified', False),
                           ('legacy_source_sha256', {}), ('best_epoch', 3)]:
            self.metadata = {**original, key: value}
            self.save()
            with self.assertRaises(RuntimeError, msg=key):
                self.check()

    def test_prepared_data_mismatch_rejected_before_prediction_use(self):
        with patch.object(baseline, 'REPO', self.repo):
            with self.assertRaisesRegex(RuntimeError, 'Prepared split differs'):
                baseline.verify_baseline(self.run, self.root, {'dataset': baseline.DATASET})

    def setup_matched_predictions(self):
        sizes = {'train': 1, 'validation': 1, 'test': 1}
        self.metadata['dataset_split_sizes'] = sizes
        for split in sizes:
            rows = [{'id': 'test-' + split, 'text': split, 'labels': [0]}]
            self.metadata['dataset_signatures'][split] = baseline.dataset_signature(rows)
            if split != 'train':
                y = np.zeros((1, 28), dtype=np.uint8)
                y[0, 0] = 1
                np.savez(self.run / (split + '_predictions.npz'), gold=y, logits=np.zeros((1, 28)),
                         ids=np.array(['test-' + split]))
        self.save()
        return sizes

    def test_full_normalized_gate_matches_data_and_predictions(self):
        sizes = self.setup_matched_predictions()
        with patch.object(baseline, 'REPO', self.repo), patch.object(baseline, 'SIZES', sizes):
            proof = baseline.verify_baseline(self.run, self.root, {'dataset': baseline.DATASET})
        self.assertEqual(proof['format'], 'normalized')
        self.assertEqual(proof['success_sha256'], sha(self.run / 'SUCCESS.json'))

    def test_rehashed_wrong_prediction_ids_rejected(self):
        sizes = self.setup_matched_predictions()
        y = np.zeros((1, 28), dtype=np.uint8)
        y[0, 0] = 1
        np.savez(self.run / 'test_predictions.npz', gold=y, logits=np.zeros((1, 28)), ids=np.array(['wrong-id']))
        self.save()
        with patch.object(baseline, 'REPO', self.repo), patch.object(baseline, 'SIZES', sizes):
            with self.assertRaisesRegex(RuntimeError, 'official IDs/GT/order'):
                baseline.verify_baseline(self.run, self.root, {'dataset': baseline.DATASET})


if __name__ == '__main__':
    unittest.main()
