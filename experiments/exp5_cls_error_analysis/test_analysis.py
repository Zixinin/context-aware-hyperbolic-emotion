"""Synthetic CPU regression tests. Fixtures are NOT empirical experiment results."""
from __future__ import annotations

import ast
import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

import analyze as a


def fixture_data():
    result = {}
    for index, split in enumerate(('train','validation','test')):
        rng = np.random.RandomState(100+index)
        n = 32
        gold = (rng.rand(n,28) < .15).astype(np.uint8)
        gold[:,0] = np.arange(n)%2
        result[split] = dict(gold=gold,ids=np.array([f'{split}_{i}' for i in range(n)]),
                             texts=[f'Synthetic fixture {split} {i}' for i in range(n)])
    return result


def fixture_predictions(data):
    return {seed:{split:np.random.RandomState(seed+len(split)).uniform(.01,.8,size=data[split]['gold'].shape)
                  for split in ('validation','test')} for seed in a.SEEDS}


def seal(root):
    source = {name:{} for name in a.SOURCE_DIRS}
    expected = dict(controls={},exp5a={})
    seed = int(root.name.split('seed')[-1])
    if seed != 42:
        expected = dict(multiseed={},frozen=expected)
    a.write_json(root/'SUCCESS.json',dict(seed=seed,status='EXP5A_CONTROL_MODEL_VERIFIED' if seed==42 else 'EXP5A_MULTISEED_MODEL_VERIFIED',
                 source_sha256=expected,artifacts_sha256={str(p.relative_to(root)):a.sha(p) for p in root.rglob('*') if p.is_file()}))
    return source


def fixture_run(base, seed, data, predictions):
    base.mkdir(parents=True,exist_ok=True)
    root = base/f'cls_seed{seed}'
    root.mkdir()
    expected = dict(controls={},exp5a={})
    if seed != 42:
        expected = dict(multiseed={},frozen=expected)
    cfg = dict(seed=seed,mode='full',representation='cls',epochs=4,fixed_threshold=.3,
               max_length=50,effective_batch_size=16,learning_rate=5e-5)
    a.write_json(root/'config.json',cfg)
    preflight = base/f'preflight_{seed}.json'
    parameters=dict(trainable_parameters=107741212,query_parameters=0)
    a.write_json(preflight,dict(source_sha256=expected,pip_freeze='numpy==fixture\n',packages={'numpy':'fixture'},
                               parameters={'cls':parameters},seed42_reference={'fixture':True}))
    a.write_json(root/'environment.json',dict(status='completed',seed=seed,representation='cls',source_sha256=expected,
                 first_batch_backward_verified=True,checkpoint_reload_verified=True,completed_optimizer_steps=10856,
                 total_optimizer_steps=10856,hostname='D3020-A3',gpu='RTX 4090 D',packages={'numpy':'fixture'},
                 parameters_runtime=parameters,seed42_reference={'fixture':True},
                 preflight_path=str(preflight),preflight_sha256=a.sha(preflight)))
    (root/'pip-freeze.txt').write_text('numpy==fixture\n')
    (root/'checkpoint_best.pt').write_bytes(b'synthetic inert checkpoint bytes, never deserialized')
    a.write_json(root/'sample_ids.json',{k:data[k]['ids'].tolist() for k in ('train','validation')})
    a.write_json(root/'dataset_statistics.json',dict(train_label_counts=data['train']['gold'].sum(0).tolist(),rare_ids=list(a.RARE),
                 label_order={str(i):x for i,x in enumerate(a.LABELS)},splits={k:len(v['ids']) for k,v in data.items()}))
    for split in ('validation','test'):
        probs = predictions[seed][split]
        logits = np.log(probs/(1-probs))
        np.savez_compressed(root/f'{split}_predictions.npz',logits=logits,targets=data[split]['gold'],ids=data[split]['ids'])
        rows = a.metric_rows(data[split]['gold'],a.sigmoid(logits),split,seed)
        metrics = dict(a.global_metrics(rows),threshold=.3,num_examples=len(probs))
        a.write_json(root/f'{split}_fixed_0.30_metrics.json',metrics)
        # Original artifact schema has a literal decimal point in this key.
        a.write_json(root/f'{split}_metrics.json',{'fixed_0.30':metrics})
        pd.DataFrame(rows)[['label_id','label','support','precision','recall','f1']].to_csv(root/f'{split}_fixed_0.30_per_label.csv',index=False)
    val = a.read(root/'validation_fixed_0.30_metrics.json')['macro_f1']
    pd.DataFrame(dict(epoch=[1,2,3,4],global_step=[2714,5428,8142,10856],training_loss=[.2,.1,.09,.08],
                     validation_macro_f1=[val-.02,val-.01,val,val],validation_micro_f1=[.5]*4)).to_csv(root/'training_log.csv',index=False)
    a.write_json(root/'checkpoint_selection.json',dict(epoch=3,global_step=8142,threshold=.3,validation_macro_f1=val))
    return root,seal(root)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data=fixture_data()
        self.probs=fixture_predictions(self.data)

    def test_exact_tp_fp_fn(self):
        gold=np.zeros((4,28),dtype=np.uint8)
        gold[:2,0]=1
        probs=np.zeros_like(gold,dtype=float)
        probs[[0,2],0]=.3
        row=a.metric_rows(gold,probs,'test',42)[0]
        self.assertEqual([row[k] for k in ('tp','fp','fn','tn','predicted_positive')],[1,1,1,1,2])
        self.assertEqual([row[k] for k in ('precision','recall','f1')],[.5,.5,.5])

    def test_threshold_is_inclusive(self):
        gold=np.ones((3,28),dtype=np.uint8)
        probs=np.full((3,28),.3)
        probs[0]=np.nextafter(.3,0)
        probs[2]=np.nextafter(.3,1)
        row=a.metric_rows(gold,probs,'validation',123)[0]
        self.assertEqual((row['tp'],row['fn']),(2,1))

    def test_zero_predictions_not_false_precision_cause(self):
        gold=np.ones((2,28),dtype=np.uint8)
        row=a.metric_rows(gold,np.zeros((2,28)),'test',42)[0]
        self.assertFalse(row['precision_defined'])
        self.assertEqual((row['precision'],row['fp']),(0,0))

    def test_gold_zero_flag(self):
        row=a.metric_rows(np.zeros((2,28)),np.ones((2,28)),'test',42)[0]
        self.assertFalse(row['recall_defined'])
        self.assertIsNone(row['predicted_to_gold_ratio'])

    def test_sigmoid_stable(self):
        values=a.sigmoid(np.array([-10000,0,10000]))
        np.testing.assert_array_equal(values,[0,.5,1])
        with self.assertRaises(ValueError): a.sigmoid(np.array([np.nan]))

    def test_id_order_rejected(self):
        d=self.data['validation']
        logits=np.zeros_like(d['gold'],dtype=float)
        with self.assertRaisesRegex(ValueError,'IDs/order'):
            a.validate_predictions(logits,d['gold'],d['ids'][::-1],d)

    def test_gold_rejected(self):
        d=self.data['validation']
        with self.assertRaisesRegex(ValueError,'targets'):
            a.validate_predictions(np.zeros_like(d['gold']),1-d['gold'],d['ids'],d)

    def test_shape_rejected(self):
        d=self.data['validation']
        with self.assertRaisesRegex(ValueError,'shape'):
            a.validate_predictions(np.zeros((len(d['ids']),27)),d['gold'],d['ids'],d)

    def test_diagnosis_rules(self):
        self.assertEqual(a.diagnosis(.7,.3),'recall_dominant')
        self.assertEqual(a.diagnosis(.3,.7),'precision_dominant')
        self.assertEqual(a.diagnosis(.3,.3),'both_low')
        self.assertEqual(a.diagnosis(.7,.7),'no_dominant_flag')

    def test_rare5_definition(self):
        self.assertEqual(set(a.LABELS[i] for i in a.RARE5),{'pride','nervousness','embarrassment','remorse','fear'})

    def test_pair_legitimate_cooccurrence(self):
        gold=np.zeros((4,28),dtype=np.uint8)
        gold[:,0]=[0,1,0,1]
        gold[:,1]=[0,0,1,1]
        probs=gold.astype(float)
        r=a.pair_counts(gold,probs,0,1)
        self.assertEqual(r['both_gold_both_recalled'],1)
        self.assertEqual(r['a_only_a_FN_b_FP']+r['b_only_b_FN_a_FP'],0)

    def test_pair_cross_error(self):
        gold=np.zeros((3,28),dtype=np.uint8)
        gold[0,0]=1
        gold[1,1]=1
        gold[2,:2]=1
        probs=np.zeros_like(gold,dtype=float)
        probs[0,1]=.9
        probs[1,0]=.9
        r=a.pair_counts(gold,probs,0,1)
        self.assertEqual((r['a_only_a_FN_b_FP'],r['b_only_b_FN_a_FP'],r['both_gold_both_missed']),(1,1,1))
        self.assertEqual(r['a_only_a_FN_b_FP_rate'],1)

    def test_pair_empty_denominator(self):
        r=a.pair_counts(np.zeros((2,28)),np.zeros((2,28)),0,1)
        self.assertIsNone(r['a_only_a_FN_b_FP_rate'])
        self.assertIsNone(r['predicted_jaccard'])

    def test_pair_pattern_totals(self):
        pairs,patterns,examples=a.pair_tables(self.probs,self.data)
        self.assertEqual(len(pairs),378*2*3)
        self.assertEqual(len(patterns),3*2*3*16)
        totals=patterns.groupby(['split','seed','label_a','label_b'])['count'].sum()
        self.assertTrue((totals==32).all())
        self.assertTrue((examples.split=='validation').all())

    def test_boundary_exact_bins_and_conservation(self):
        p=copy.deepcopy(self.probs)
        for seed in a.SEEDS:
            p[seed]['validation'][:4,0]=[.249999,.25,.3,.35]
        _,hist,boundary,_=a.probability_tables(p,self.data)
        self.assertTrue((hist.groupby(['seed','label'])['count'].sum()==32).all())
        self.assertTrue((boundary.fn==boundary.fn_p_lt_025+boundary.fn_p_025_to_030).all())
        self.assertTrue((boundary.fp==boundary.fp_p_030_to_035+boundary.fp_p_ge_035).all())

    def test_neighborhood_boundary_assignment(self):
        data=fixture_data()
        for split in data:
            data[split]['gold'][:,0]=0
            data[split]['gold'][:2,0]=1
        probs=fixture_predictions(data)
        for seed in a.SEEDS:
            probs[seed]['validation'][:,0]=0
            probs[seed]['validation'][:4,0]=[.249999,.25,.3,.35]
        boundary=a.probability_tables(probs,data)[2]
        r=boundary[(boundary.seed==42)&(boundary.label==a.LABELS[0])].iloc[0]
        self.assertEqual([r[k] for k in ('fn_p_lt_025','fn_p_025_to_030','fp_p_030_to_035','fp_p_ge_035')],[1,1,1,1])

    def test_bce_empty_populations_are_not_nan(self):
        for split in self.data:
            self.data[split]['gold'][:,0]=0
            self.data[split]['gold'][:,1]=1
        self.probs=fixture_predictions(self.data)
        *_,loss=a.probability_tables(self.probs,self.data)
        self.assertTrue(np.isfinite(loss.negative_BCE_sum).all())
        self.assertTrue(np.isfinite(loss.positive_BCE_sum).all())

    def test_validation_probability_does_not_read_test(self):
        validation_only={s:{'validation':self.probs[s]['validation']} for s in a.SEEDS}
        tables=a.probability_tables(validation_only,self.data)
        self.assertTrue(all((table.split=='validation').all() for table in tables))

    def test_group_conservation(self):
        rows=pd.DataFrame([r for s in a.SEEDS for split in ('validation','test') for r in a.metric_rows(self.data[split]['gold'],self.probs[s][split],split,s)])
        groups=a.group_metrics(rows)
        for split in ('validation','test'):
            for seed in a.SEEDS:
                g=groups[(groups.seed==seed)&(groups.split==split)].set_index('group')
                for k in ('tp','fp','fn','support'):
                    self.assertEqual(g.loc['rare7',k],g.loc['rare5_excluding_grief_relief',k]+g.loc['grief_relief_only',k])
                self.assertAlmostEqual(g.loc['rare7','macro_f1'],(5*g.loc['rare5_excluding_grief_relief','macro_f1']+2*g.loc['grief_relief_only','macro_f1'])/7)

    def test_sample_sd_and_no_ensemble(self):
        rows=pd.DataFrame([r for s in a.SEEDS for split in ('validation','test') for r in a.metric_rows(self.data[split]['gold'],self.probs[s][split],split,s)])
        result=a.aggregate_labels(rows,self.probs,self.data)
        local=rows[(rows.split=='test')&(rows.label_id==0)].f1.to_numpy()
        r=result[(result.split=='test')&(result.label_id==0)].iloc[0]
        self.assertAlmostEqual(r.f1_mean,local.mean())
        self.assertAlmostEqual(r.f1_sd,local.std(ddof=1))

    def test_frozen_inventory_and_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root,source=fixture_run(Path(name),42,self.data,self.probs)
            result,proof=a.verify_run(root,42,self.data,source)
            self.assertEqual(proof['best_epoch'],3)
            self.assertEqual(set(result),{'validation','test'})
            (root/'checkpoint_best.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):
                a.verify_run(root,42,self.data,source)

    def test_new_seed_provenance(self):
        with tempfile.TemporaryDirectory() as name:
            for seed in (123,2026):
                root,source=fixture_run(Path(name),seed,self.data,self.probs)
                _,proof=a.verify_run(root,seed,self.data,source)
                self.assertEqual(proof['seed'],seed)

    def test_inventory_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root,source=fixture_run(Path(name),42,self.data,self.probs)
            (root/'unexpected.txt').write_text('x')
            with self.assertRaisesRegex(ValueError,'inventory'):
                a.verify_run(root,42,self.data,source)

    def test_manifest_and_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'example.py').write_text('x=1\n')
            (root/'PACKAGE_MANIFEST.sha256').write_text(a.sha(root/'example.py')+'  example.py\n')
            a.manifest_check(root)
            (root/'example.py').write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'Source package'):
                a.manifest_check(root)

    def test_snapshot_detects_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'a').write_text('x')
            before=a.snapshot([root],[])
            (root/'b').write_text('y')
            self.assertNotEqual(before,a.snapshot([root],[]))

    def test_no_model_no_training_no_tuning_imports(self):
        tree=ast.parse(Path(a.__file__).read_text())
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imported.extend(x.name.split('.')[0] for x in node.names)
            if isinstance(node,ast.ImportFrom): imported.append((node.module or '').split('.')[0])
        self.assertTrue({'torch','transformers','train','model','control_train','multi_train'}.isdisjoint(imported))
        calls={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertTrue({'backward','step','load_state_dict','from_pretrained'}.isdisjoint(calls))

    def test_full_synthetic_report_and_csv(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            evidence=[dict(seed=s,best_epoch=3,steps=10856,validation_macro_f1=.5,checkpoint_sha256='synthetic-not-a-model') for s in a.SEEDS]
            tables=a.analyze_predictions(self.data,self.probs,evidence,out)
            self.assertEqual(len(tables['label_metrics_by_seed.csv']),168)
            self.assertEqual(len(tables['label_summary.csv']),56)
            self.assertEqual(len(tables['label_counts.csv']),28)
            self.assertEqual(len(tables['rare_group_metrics.csv']),30)
            text=(out/'EXP5_CLS_ERROR_ANALYSIS.md').read_text()
            self.assertIn('Rare-5',text)
            self.assertIn('n.a.',text)
            for name,table in tables.items():
                loaded=pd.read_csv(out/name)
                self.assertEqual(list(loaded.columns),list(table.columns))
                self.assertEqual(len(loaded),len(table))
            self.assertFalse(any(p.suffix=='.pt' for p in out.iterdir()))

    def test_full_cli_fixture_and_output_overwrite_refusal(self):
        # Inert synthetic checkpoint + synthetic predictions. No empirical findings.
        with tempfile.TemporaryDirectory() as name:
            repo=Path(name)
            source={k:{} for k in a.SOURCE_DIRS}
            for directory in a.SOURCE_DIRS:
                (repo/'experiments'/directory).mkdir(parents=True)
            base_cfg=dict(epochs=4,fixed_threshold=.3,max_length=50,effective_batch_size=16,learning_rate=5e-5,seed=42)
            a.write_json(repo/'experiments/exp5_label_aware_euclidean/config.json',base_cfg)
            asset=repo/'experiments/exp2_lgsa/LGSA-GoEmotions/assets/goemotions-v0/manifest.json'
            asset.parent.mkdir(parents=True)
            a.write_json(asset,dict(status='GOEMOTIONS_ASSETS_READY',rare_ids=list(a.RARE),split_sizes={k:32 for k in self.data}))
            data_dir=repo/'数据集/GoEmotions/processed'
            data_dir.mkdir(parents=True)
            for filename,*_ in a.SPLITS.values():
                (data_dir/filename).write_text('synthetic data file, canonical loading mocked in this test\n')
            first_hash=None
            for seed in a.SEEDS:
                parent=repo/('results/exp5a_controls/run_20261013T103802Z' if seed==42 else 'results/exp5a_controls_multiseed/run_20261013T113904Z')
                root,_=fixture_run(parent,seed,self.data,self.probs)
                cfg=dict(base_cfg,experiment='Exp5A-Control',representation='cls',mode='full',
                         dropout_layout='independent_per_label_matching_frozen_exp5a',
                         head_initialization='skip_query_draw_then_normal_0.02_matching_exp5a')
                env=a.read(root/'environment.json')
                env['dataset_manifest_sha256']=a.sha(asset)
                if seed!=42:
                    cfg.update(experiment='Exp5A-Control-MultiSeed',seed=seed,authorized_seeds=[123,2026],aggregation_seeds=list(a.SEEDS))
                    env['seed42_reference']={'cls_success_sha256':first_hash}
                    preflight=Path(env['preflight_path'])
                    audit=a.read(preflight)
                    audit['seed42_reference']=env['seed42_reference']
                    a.write_json(preflight,audit)
                    env['preflight_sha256']=a.sha(preflight)
                a.write_json(root/'config.json',cfg)
                a.write_json(root/'environment.json',env)
                (root/'SUCCESS.json').unlink()  # disposable synthetic proof only
                seal(root)
                if seed==42:
                    first_hash=a.sha(root/'SUCCESS.json')
            out=repo/'results/exp5_cls_error_analysis/run_fixture'
            argv=['analyze.py','--repo',str(repo),'--out',str(out)]
            with patch.object(sys,'argv',argv),patch.object(a,'check_sources',return_value=source),patch.object(a,'load_data',return_value=self.data),contextlib.redirect_stdout(io.StringIO()):
                a.main()
                self.assertEqual(a.read(out/'ANALYSIS_SUCCESS.json')['status'],'EXP5_CLS_ERROR_ANALYSIS_COMPLETE_VERIFIED')
                self.assertTrue(a.read(out/'PROTECTED_INPUTS.json')['unchanged'])
                self.assertEqual(a.read(out/'INPUT_AUDIT.json')['forward_calls'],0)
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    a.main()


if __name__=='__main__':
    unittest.main(verbosity=2)
