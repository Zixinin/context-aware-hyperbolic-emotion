"""CPU synthetic protocol tests. This is NOT a formal model/GPU smoke test."""
import argparse
import ast
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from b4lib import NAMES, HERE, read, require, package_hashes, sha, write
from hierarchy_adapter import adapt, load_hierarchy, targets
from data_adapter import encode, tensors
from loop_adapter import derive
from evaluate import EmotionClassifier, metrics, probabilities
from geometry_adapter import describe

REPOSITORY = None
TREE = None

class FakeTokenizer:
    vocab_size, cls_token_id, sep_token_id, pad_token_id, mask_token_id = 200, 101, 102, 0, 103
    def __call__(self, text, **kwargs):
        self.last_kwargs = kwargs
        words = [150 if w[0].isupper() else 151 for w in text.split()]
        return {'input_ids': [101] + words[:kwargs['max_length'] - 2] + [102]}

def upstream_symbols(path, names, scope):
    tree = ast.parse(path.read_text())
    if path.name == 'prompt.py':
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Prompt')
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    require(len(nodes) == len(names), 'Upstream synthetic-test symbol lookup failed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
    return scope

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = load_hierarchy(TREE)
        cls.root = REPOSITORY / 'experiments/exp2_lgsa/LGSA-GoEmotions'
        cls.source = cls.root / 'working_lgsa/models'
        cls.old = read(cls.root / 'goemotions_adaptation/hierarchy.json')
        cls.cfg = read(REPOSITORY / 'experiments/exp2_1_multiseed/frozen/lgsa.json')

    def rejected(self, mutate):
        obj = copy.deepcopy(self.h['source'])
        mutate(obj)
        with self.assertRaises(RuntimeError):
            adapt(obj)

    def test_complete_tree_shape_and_no_pruning(self):
        self.assertEqual((self.h['num_nodes'], self.h['num_depths']), (56, 9))
        self.assertEqual(self.h['summary']['retained_unstable_clusters'], 8)
        self.assertEqual(self.h['summary']['leaves'], 28)
        self.assertEqual(self.h['summary']['neutral_path'], ['ROOT', 'NEUTRAL', 'neutral'])
        self.assertEqual({i: self.h['labels'][i] for i in range(28)}, dict(enumerate(NAMES)))
        self.assertEqual(len(self.h['summary']['sanity_check_pairs']), 6)

    def test_duplicate_node_rejected(self):
        self.rejected(lambda o: o['root']['children'].append(copy.deepcopy(o['root']['children'][1])))

    def test_missing_leaf_rejected(self):
        self.rejected(lambda o: o['root']['children'][1]['children'].clear())

    def test_extra_leaf_rejected(self):
        self.rejected(lambda o: o['root']['children'].append({'node_id': 'fake', 'label_id': 28, 'children': []}))

    def test_fine_ids_changed_rejected(self):
        self.rejected(lambda o: o['root']['children'][1]['children'][0].update(label_id=0))

    def test_internal_prediction_id_rejected(self):
        self.rejected(lambda o: o['root']['children'][0].update(label_id=28))

    def test_cycle_rejected(self):
        def mutate(o):
            o['root']['children'][0]['children'].append(o['root'])
        self.rejected(mutate)

    def test_root_branches_changed_rejected(self):
        self.rejected(lambda o: o['root']['children'].reverse())

    def test_pruning_rejected(self):
        self.rejected(lambda o: o['root']['children'][0]['children'][0]['children'].pop())

    def test_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tree.json'
            path.write_text(json.dumps(self.h['source']))
            with self.assertRaises(RuntimeError):
                load_hierarchy(path)

    def test_ancestor_targets_all_depths(self):
        encoded, fine = targets([0, 27], self.h)
        y = np.array(encoded).reshape(9, 56)
        self.assertEqual((y != -100).sum(), 56)
        active = set()
        for leaf in [0, 27]:
            cur = leaf
            while cur != -1:
                active.add(cur)
                cur = self.h['parents'][cur]
        self.assertEqual(set(np.where(y == 1)[1]), active)
        self.assertEqual(sum(fine), 2)
        for d, ids in self.h['depths'].items():
            self.assertEqual(set(np.where(y[d] != -100)[0]), set(ids))

    def test_neutral_is_single_leaf_with_one_auxiliary_ancestor(self):
        y, fine = targets([27], self.h)
        self.assertEqual(sum(v == 1 for v in y), 2)
        self.assertEqual(sum(fine), 1)
        for bad in ([], [0, 0], [28], [-1], [True]):
            with self.assertRaises(RuntimeError):
                targets(bad, self.h)

    def test_total68_text50_and_nine_masks(self):
        tokenizer = FakeTokenizer()
        for text in ['Hello hello', ' '.join(['Word'] * 100)]:
            r = encode({'text': text, 'labels': [0]}, tokenizer, self.h)
            self.assertEqual(len(r['input_ids']), 68)
            self.assertEqual(r['input_ids'].count(265), 9)
            actual = [i for i, a in zip(r['input_ids'], r['attention_mask']) if a]
            original = tokenizer(text, max_length=50)['input_ids'][:-1]
            self.assertEqual(actual[:-19], original)
            self.assertEqual(actual[-19:], sum(([256 + d, 265] for d in range(9)), []) + [102])
        with self.assertRaises(RuntimeError):
            encode({'text': 'Text', 'labels': [0]}, tokenizer, self.h, 36)

    def test_generic_data_adapter_exactly_matches_old_two_layer_encoding(self):
        scope = upstream_symbols(self.root / 'goemotions_adaptation/hierarchy_goemotions.py',
            {'load_hierarchy', 'targets'}, {'read': lambda p: self.old, 'NAMES': NAMES, 'require': require})
        old_h = scope['load_hierarchy'](None)
        old_scope = upstream_symbols(self.root / 'goemotions_adaptation/data_goemotions.py',
            {'encode'}, {'require': require, 'targets': scope['targets']})
        generic_h = dict(old_h, num_nodes=32, num_depths=2)
        for text in ['Hello hello', ' '.join(['Word'] * 100)]:
            for labels in [[0], [27], [14, 19], [0, 27]]:
                r = {'text': text, 'labels': labels}
                self.assertEqual(encode(r, FakeTokenizer(), generic_h), old_scope['encode'](r, FakeTokenizer(), old_h))

    def test_public_classifier_gathers_actual_leaf_depths(self):
        z = torch.arange(2 * 9 * 56, dtype=torch.float32).reshape(2, 9, 56)
        fine = EmotionClassifier.select(z, self.h)
        self.assertEqual(fine.shape, (2, 28))
        for i in range(28):
            torch.testing.assert_close(fine[:, i], z[:, self.h['fine_depths'][i], i])
        self.assertFalse(torch.equal(fine, z[:, 1, :28]))
        self.assertEqual(EmotionClassifier.output_dimension, 28)

    def test_old_32_two_depth_loop_is_exact_ast_identity(self):
        source = (self.root / 'goemotions_adaptation/run.py').read_text()
        tree, report, diff = derive(source, 32, 2)
        old = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        self.assertEqual(ast.dump(tree.body[0]), ast.dump(old))
        self.assertEqual(diff, '\n')

    def test_new_loop_only_five_dimension_literals_change(self):
        source = (self.root / 'goemotions_adaptation/run.py').read_text()
        tree, report, diff = derive(source, 56, 9)
        self.assertEqual(sum(report['literal_edits'].values()), 5)
        text = ast.unparse(tree)
        self.assertIn('num_labels=56', text)
        self.assertIn('[-66:-10]', text)
        self.assertIn('vocab_size + 66', text)
        # Reverse the exact approved transform and recover original AST.
        back = text.replace('num_labels=56', 'num_labels=32').replace('item() == 64', 'item() == 33')
        back = back.replace('vocab_size + 66', 'vocab_size + 35').replace('[-66:-10]', '[-35:-3]')
        old, _, _ = derive(source, 32, 2)
        self.assertEqual(ast.dump(ast.parse(back)), ast.dump(old))

    def test_drifted_loop_fails_loudly(self):
        source = (self.root / 'goemotions_adaptation/run.py').read_text().replace('num_labels=32', 'num_labels=33')
        with self.assertRaises(RuntimeError):
            derive(source, 56, 9)

    def test_metrics_match_sklearn_including_weighted(self):
        from sklearn.metrics import precision_score, recall_score, f1_score
        rng = np.random.default_rng(42)
        y, p = rng.integers(0, 2, (35, 28)), rng.random((35, 28))
        summary, labels = metrics(y, p, .30)
        for average in ['micro', 'macro']:
            for key, fn in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
                self.assertAlmostEqual(summary[average + '_' + key], fn(y, p >= .30, average=average, zero_division=0))
        self.assertAlmostEqual(summary['weighted_f1'], f1_score(y, p >= .30, average='weighted', zero_division=0))
        self.assertEqual(len(labels), 28)

    def test_metrics_reject_internal_columns_and_nonfinite(self):
        with self.assertRaises(RuntimeError):
            metrics(np.zeros((3, 56)), np.zeros((3, 56)), .30)
        p = np.zeros((3, 28)); p[0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            metrics(np.zeros((3, 28)), p, .30)

    def test_sigmoid_preserves_exp2_float32(self):
        z = np.array([[-80, -1, 0, 1, 80]], dtype=np.float32)
        old = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        np.testing.assert_array_equal(probabilities(z), old)
        self.assertEqual(probabilities(z).dtype, np.float32)

    def test_unchanged_hgs_supports_full_depth(self):
        scope = upstream_symbols(self.source / 'prompt.py',
            {'_gram_schmidt', 'semantically_anchored_gof_initialization'}, {'torch': torch})
        class Toy: pass
        for name in ['_gram_schmidt', 'semantically_anchored_gof_initialization']:
            setattr(Toy, name, scope[name])
        x = torch.randn(56, 64, generator=torch.Generator().manual_seed(42))
        result = Toy().semantically_anchored_gof_initialization(x, self.h['depths'], self.h['parents'])
        report = describe(result, self.h)
        self.assertLess(report['top_max_abs_cosine'], 1e-4)
        self.assertTrue(all(f['max_abs_cosine'] < 1e-4 for f in report['families'].values()))
        torch.testing.assert_close(x.norm(dim=1), result.norm(dim=1))

    def test_original_classification_and_hga_backward_full_tree(self):
        scope = upstream_symbols(self.source / 'loss.py',
            {'multilabel_categorical_crossentropy', 'HierarchicalGofLoss'},
            {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'math': __import__('math')})
        y = torch.tensor([targets([0, 27], self.h)[0], targets([14, 19], self.h)[0]]).reshape(-1, 56)
        z = torch.randn(18, 56, generator=torch.Generator().manual_seed(2), requires_grad=True)
        cls = scope['multilabel_categorical_crossentropy'](y, z)
        manual = []
        for b in range(2):
            selected = y[b*9:(b+1)*9] != -100
            labels, logits = y[b*9:(b+1)*9][selected], z[b*9:(b+1)*9][selected]
            manual.append(torch.logsumexp(torch.cat([logits[labels == 0], torch.zeros(1)]), 0) +
                torch.logsumexp(torch.cat([-logits[labels == 1], torch.zeros(1)]), 0))
        torch.testing.assert_close(cls, torch.stack(manual).mean())
        x = torch.randn(56, 64, generator=torch.Generator().manual_seed(3), requires_grad=True)
        gof = scope['HierarchicalGofLoss'](self.h['depths'], self.h['parents'], parent_child_angle_deg=10,
            length_decay_factor=.9, return_components=True)
        geo, parts = gof(x)
        (cls + geo).backward()
        self.assertTrue(torch.isfinite(z.grad).all() and torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.norm()), 0)

    def test_original_forward_backward_generate_contract_on_toy_encoder(self):
        from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput
        loss_scope = upstream_symbols(self.source / 'loss.py', {'multilabel_categorical_crossentropy'}, {'torch': torch})
        scope = upstream_symbols(self.source / 'prompt.py', {'forward', 'generate'},
            {'torch': torch, 'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
             'MaskedLMOutput': MaskedLMOutput, **loss_scope})
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.table = torch.nn.Embedding(266, 32)
            def forward(self, ids, **kwargs):
                return BaseModelOutput(last_hidden_state=self.table(ids))
        class Toy(torch.nn.Module):
            forward = scope['forward']
            generate = scope['generate']
            def __init__(self):
                super().__init__()
                from types import SimpleNamespace
                self.bert, self.cls = Encoder(), torch.nn.Linear(32, 266)
                self.tokenizer = FakeTokenizer()
                self.config = SimpleNamespace(use_return_dict=True, pad_token_id=0)
                self.vocab_size, self.num_labels = 200, 56
                self.multiclass_bias = torch.nn.Parameter(torch.zeros(56))
            def get_input_embeddings(self):
                from types import SimpleNamespace
                return SimpleNamespace(size=266)
        data = tensors([{'text': ' '.join(['Word']*100), 'labels': [0, 27]}] * 4, FakeTokenizer(), self.h, 50)
        model = Toy().train()
        output, mlm, classification = model(**{k: v for k, v in data.items() if k != 'fine'})
        self.assertTrue(torch.isfinite(torch.stack([output.loss, mlm, classification])).all())
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.bert.table.weight.grad).all())
        model.eval()
        logits = EmotionClassifier(model, self.h).forward(data['input_ids'])
        self.assertEqual(logits.shape, (4, 28))

    def test_normalized_baseline_metrics_and_hash_gate(self):
        from preflight import verify_baseline, DATASET
        from b4lib import RARE, SIZES, REVISION, signature
        with tempfile.TemporaryDirectory() as temp:
            root, run = Path(temp) / 'runtime', Path(temp) / 'baseline'
            (root / 'goemotions_adaptation').mkdir(parents=True)
            run.mkdir()
            write(root / 'goemotions_adaptation/lgsa_goemotions_seed42.json', self.cfg)
            expected = read(REPOSITORY / 'experiments/exp2_1_multiseed/frozen/expected_sources.json')['lgsa']
            signatures = {}
            for split in ['validation', 'test']:
                size = SIZES[split]
                rs = [{'id': split + str(i), 'text': 'Synthetic', 'labels': [0]} for i in range(size)]
                signatures[split] = signature(rs)
                gold = np.zeros((size, 28), dtype=np.int64); gold[:, 0] = 1
                logits = np.zeros((size, 28), dtype=np.float32)
                np.savez_compressed(run / (split + '_predictions.npz'), gold=gold,
                    logits=logits, ids=np.array([r['id'] for r in rs]))
            write(run / 'test_fixed_0.30_metrics.json', metrics(gold, probabilities(logits), .30)[0])
            meta = {'status': 'COMPLETED', 'model': 'lgsa', 'seed': 42, 'phase': 'full',
                'slurm_state': 'COMPLETED', 'slurm_exit_code': '0:0', 'native_success_sha256': 'synthetic',
                'config': self.cfg, 'legacy_source_sha256': expected,
                'lgsa_patch_commit': '209f83762dae1b354ba40e68b6952482ecf96208',
                'hierarchy_sha256': 'synthetic_tree', 'raw_model_repo': 'google-bert/bert-base-cased',
                'raw_model_revision': REVISION, 'dataset': DATASET['repository'],
                'dataset_config': DATASET['configuration'], 'dataset_revision': DATASET['revision'],
                'dataset_split_sizes': SIZES, 'num_labels': 28, 'label_names': NAMES, 'rare_ids': RARE,
                'fixed_threshold': .30, 'threshold_candidates': self.cfg['threshold_candidates'],
                'dataset_signatures': signatures, 'checkpoint_reload_verified': True,
                'checkpoint_reload_evidence': 'synthetic', 'best_epoch': 4}
            write(run / 'experiment_metadata.json', meta)
            write(run / 'config.json', self.cfg)
            write(run / 'checkpoint_info.json', {'best_epoch': 4, 'selection': self.cfg['checkpoint_selection']})
            (run / 'validation_metrics_by_epoch.csv').write_text('epoch,validation_macro_f1\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n')
            write(run / 'SUCCESS.json', {'status': 'NORMALIZED_RUN_PASS', 'model': 'lgsa', 'seed': 42,
                'native_success_sha256': 'synthetic', 'artifacts_sha256': {
                    p.name: sha(p) for p in run.iterdir() if p.is_file()}})
            manifest = {'hierarchy_sha256': 'synthetic_tree', 'dataset_signatures': signatures}
            result = verify_baseline(run, root, manifest, expected)
            self.assertEqual(result['status'], 'EXP2_SEED42_BASELINE_VERIFIED')
            self.assertIn('weighted_f1', result['recomputed_fixed_0_30_metrics'])
            (run / 'config.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                verify_baseline(run, root, manifest, expected)

    def test_canonical_quoted_tsv_decoding_and_byte_preservation(self):
        import csv
        from preflight import canonical_records
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'quoted.tsv'
            content = '"""Hello"" world"\t0,27\tquoted-id\nPlain text\t14,19\tplain-id\n'
            path.write_text(content)
            before = sha(path)
            rows = canonical_records(path)
            self.assertEqual(rows, [
                {'text': '"Hello" world', 'labels': [0, 27], 'id': 'quoted-id'},
                {'text': 'Plain text', 'labels': [14, 19], 'id': 'plain-id'}])
            with path.open(newline='') as stream:
                wrong = list(csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE))
            self.assertNotEqual(wrong[0][0], rows[0]['text'])
            self.assertEqual(sha(path), before)

    @contextmanager
    def raw_backbone_fixture(self):
        from safetensors.torch import save_file
        from preflight import expected_lgsa_backbone_keys
        from b4lib import REVISION
        state = {name: torch.tensor([i / 100, i / 100 + 1], dtype=torch.float32)
            for i, name in enumerate(sorted(expected_lgsa_backbone_keys()))}
        raw = {}
        for name, value in state.items():
            legacy = name
            if name.endswith('.LayerNorm.weight'):
                legacy = name[:-len('weight')] + 'gamma'
            elif name.endswith('.LayerNorm.bias'):
                legacy = name[:-len('bias')] + 'beta'
            raw['bert.' + legacy] = value.clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / REVISION
            path.mkdir()
            save_file(raw, str(path / 'model.safetensors'))
            model = SimpleNamespace(bert=SimpleNamespace(pooler=None, state_dict=lambda: state))
            manifest = {'model': {'path': str(path),
                'sha256': {'model.safetensors': sha(path / 'model.safetensors')}}}
            yield state, raw, model, manifest, path

    def test_real_bert_no_pooler_schema_is_197_not_199(self):
        from transformers import BertConfig, BertModel
        from preflight import expected_lgsa_backbone_keys
        cfg = BertConfig(vocab_size=32, hidden_size=12, intermediate_size=24,
            num_hidden_layers=12, num_attention_heads=3)
        no_pooler = BertModel(cfg, add_pooling_layer=False)
        with_pooler = BertModel(cfg, add_pooling_layer=True)
        self.assertEqual(set(no_pooler.state_dict()), expected_lgsa_backbone_keys())
        self.assertEqual(len(no_pooler.state_dict()), 197)
        self.assertEqual(len(with_pooler.state_dict()), 199)
        self.assertEqual(set(with_pooler.state_dict()) - set(no_pooler.state_dict()),
            {'pooler.dense.weight', 'pooler.dense.bias'})

    def test_raw_lgsa_197_and_50_legacy_aliases_pass(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (_, _, model, manifest, _):
            proof = verify_raw(model, manifest)
            self.assertEqual(proof['tensors_compared'], 197)
            self.assertEqual(len(proof['legacy_LayerNorm_aliases_verified']), 50)
            self.assertFalse(proof['pooler_present'])

    def test_raw_changed_value_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (state, _, model, manifest, _):
            state['encoder.layer.11.output.dense.weight'][0] += 1
            with self.assertRaisesRegex(RuntimeError, 'not freshly RAW'):
                verify_raw(model, manifest)

    def test_raw_changed_dtype_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (state, _, model, manifest, _):
            state['embeddings.word_embeddings.weight'] = state['embeddings.word_embeddings.weight'].double()
            with self.assertRaisesRegex(RuntimeError, 'not freshly RAW'):
                verify_raw(model, manifest)

    def test_raw_changed_shape_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (state, _, model, manifest, _):
            state['embeddings.word_embeddings.weight'] = state['embeddings.word_embeddings.weight'].reshape(1, 2)
            with self.assertRaisesRegex(RuntimeError, 'not freshly RAW'):
                verify_raw(model, manifest)

    def test_raw_missing_backbone_tensor_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (state, _, model, manifest, _):
            del state['encoder.layer.0.output.dense.bias']
            with self.assertRaisesRegex(RuntimeError, 'backbone schema'):
                verify_raw(model, manifest)

    def test_raw_same_count_wrong_backbone_key_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (state, _, model, manifest, _):
            state['unexpected.bias'] = state.pop('encoder.layer.0.output.dense.bias')
            with self.assertRaisesRegex(RuntimeError, 'backbone schema'):
                verify_raw(model, manifest)

    def test_raw_pooler_in_lgsa_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (_, _, model, manifest, _):
            model.bert.pooler = object()
            with self.assertRaisesRegex(RuntimeError, 'must not contain a pooler'):
                verify_raw(model, manifest)

    def test_raw_legacy_alias_contract_not_relaxed(self):
        from preflight import verify_raw
        from safetensors.torch import save_file
        with self.raw_backbone_fixture() as (_, raw, model, manifest, path):
            raw['bert.embeddings.LayerNorm.weight'] = raw.pop('bert.embeddings.LayerNorm.gamma')
            save_file(raw, str(path / 'model.safetensors'))
            manifest['model']['sha256']['model.safetensors'] = sha(path / 'model.safetensors')
            with self.assertRaisesRegex(RuntimeError, 'legacy_aliases=49'):
                verify_raw(model, manifest)

    def test_raw_changed_file_hash_rejected(self):
        from preflight import verify_raw
        with self.raw_backbone_fixture() as (_, _, model, manifest, path):
            with (path / 'model.safetensors').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'file hash changed'):
                verify_raw(model, manifest)

    def test_all_canonical_splits_match_recorded_exp2_signatures(self):
        from preflight import canonical_records, CANONICAL_TSV_SHA
        from b4lib import signature, SIZES
        expected = {
            'train.tsv': ('82b870f286f41c007975ada70d028be8e0aa16718d8a2f90656d1cb3dfc927a7',
                '05e2837aff2b8bf740a9ed2abb5732abfa6b24685533dbcac5bd9263683ccc28',
                'aad37f43989f8ebb85ee52318f635159d79a73835019db190a79c11d9ebcf70d'),
            'dev.tsv': ('e6d7ce756f282c9b6ec8519279e8cc76fc7a023f28ee4733f2c0b398b48c8afc',
                'e28524ada92e6e6b08ee88b52c513a3f4bd52e9b43d047ceb5103349fa3f56b0',
                '93ca4531e1681c9f4296d8d19315b838936398fd7d51133c5ac52629c7b40e4d'),
            'test.tsv': ('89137413db73846cb973dabfc8926164f741e35c4c4d3a0184c737c99d604fa8',
                '4293c84d72c5c392e0238583dc5139ef666652ed4d55114d615a58a849ac0daa',
                'b44eb5c2d10adfc715fa26288d97ba2a78568a6ba7cf7d1550a39888b803e400')}
        for filename, hashes in expected.items():
            path = REPOSITORY / '数据集/GoEmotions/processed' / filename
            self.assertEqual(sha(path), CANONICAL_TSV_SHA[filename])
            got = signature(canonical_records(path))
            self.assertEqual(tuple(got[k] for k in ('rows_sha256', 'ids_sha256', 'gold_sha256')), hashes)
            self.assertEqual(got['size'], SIZES['validation' if filename == 'dev.tsv' else filename[:-4]])

def main():
    global REPOSITORY, TREE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--hierarchy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    REPOSITORY, TREE = args.repo.resolve(), args.hierarchy.resolve()
    require(not args.out.exists(), 'Never overwrite an existing unit proof')
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'EXP4B_UNIT_TESTS_PASS' if result.wasSuccessful() else 'EXP4B_UNIT_TESTS_FAIL',
        'system': platform.system(), 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'package_sha256': package_hashes(), 'hierarchy_sha256': sha(TREE),
        'scope': 'CPU synthetic fixtures, frozen function excerpts and read-only canonical split signatures; NOT real BERT/GAT smoke or formal training'}
    write(args.out, report)
    require(result.wasSuccessful(), 'Unit tests failed; do not launch formal stages')
    print('EXP4B_UNIT_TESTS_PASS', result.testsRun)

if __name__ == '__main__':
    main()
