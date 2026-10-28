"""CPU contracts and archived seed42 order replay, never a GPU result."""
import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import pcommon as pc
import preport as pr
from data27 import loader as frozen_loader
from model27 import Roberta27


class ToyTokenizer:
    def __call__(self,texts,**kwargs):
        assert kwargs==dict(truncation=True,max_length=50,padding=True,return_tensors='pt',return_token_type_ids=False)
        ids=torch.tensor([[int(t),1] for t in texts])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids))


def records(n=43):
    return [dict(id=str(i),text=str(i),labels=[i%27] if i%2 else [i%27,(i+1)%27]) for i in range(n)]


def four_orders(dl):
    return [[rid for b in dl for rid in b['ids']] for _ in range(4)]


def nine_records():
    return [dict(seed=s,variant=k,**{m:.3+i*.1+{'A0':0.,'A1S':.01,'N1':.03}[k] for m in pr.METRICS})
            for i,s in enumerate(pc.ALL_SEEDS) for k in pc.KINDS]


class PairedContracts(unittest.TestCase):
    def test_seed42_loader_matches_frozen_four_epochs(self):
        data=records();tok=ToyTokenizer()
        old=four_orders(frozen_loader(data,tok,True))
        new=four_orders(pc.make_loader(data,tok,42,True))
        self.assertEqual(old,new)
        self.assertEqual(new,pc.planned_orders([r['id'] for r in data],42))

    def test_train_orders_are_dedicated_reproducible_and_seed_sensitive(self):
        ids=[r['id'] for r in records()]
        first=pc.planned_orders(ids,123)
        torch.manual_seed(9876);torch.randn(1000)
        self.assertEqual(first,pc.planned_orders(ids,123))
        self.assertNotEqual(first,pc.planned_orders(ids,2026))
        self.assertTrue(all(set(a)==set(ids) for a in first))
        self.assertNotEqual(first[0],first[1])

    def test_real_archived_seed42_all_30587_ids_four_epochs(self):
        evidence=pc.REPO/'results/exp6d_hypemo_code_audit/run_20261022_prepare_v1/evidence/A1S'
        if not evidence.exists():self.skipTest('Mac evidence unavailable; Ubuntu checks saved A0 first batches and full A1S/N1 histories')
        manifest=pc.read(evidence/'dataset_manifest.json')
        ids=manifest['splits']['train']['retained_ids']
        self.assertEqual(len(ids),30587)
        for epoch,order in enumerate(pc.planned_orders(ids,42),1):
            self.assertEqual(order,pc.read(evidence/f'train_order_epoch{epoch}.json'))
        reconstructed=pc.frozen_seed42_order_replay([dict(id=rid,text='',labels=[0]) for rid in ids])
        self.assertEqual(reconstructed,pc.planned_orders(ids,42))

    def test_collation_keeps_all_positive_labels_and_short_batch(self):
        data=records();batches=list(pc.make_loader(data,ToyTokenizer(),123,True))
        self.assertEqual([len(b['ids']) for b in batches],[16,16,11])
        by={r['id']:r for r in data}
        for batch in batches:
            self.assertEqual(batch['labels'].shape,(len(batch['ids']),27))
            self.assertEqual(batch['labels'].dtype,torch.float32)
            self.assertTrue(np.array_equal(batch['labels'],pc.targets([by[rid] for rid in batch['ids']])))
        self.assertTrue(any(bool((b['labels'].sum(1)>1).any()) for b in batches))

    def test_new_configs_change_only_authorized_metadata_and_seed(self):
        metadata={'experiment','seed','variant','new_test_inference','formal_test_inference_calls','training_data_generator_seed','selection','paired_seeds','matched_control','formula_source_control','comparison'}
        bases={'A0':pc.a0.CONFIG,'A1S':pc.nc.sc.CONFIG,'N1':pc.nc.CONFIG}
        for kind in pc.KINDS:
            for seed in pc.NEW_SEEDS:
                c=pc.config(kind,seed)
                for k,v in bases[kind].items():
                    if k not in metadata:self.assertEqual(c[k],v)
                self.assertEqual(c['seed'],seed);self.assertEqual(c['training_data_generator_seed'],seed)
                self.assertFalse(c['new_test_inference'])
                if kind=='N1':self.assertEqual(c['matched_control'],'same-seed A1S cell in this Exp6D2 suite')
        for s in [42,7]:
            with self.assertRaises(RuntimeError):pc.config('N1',s)
        with self.assertRaises(RuntimeError):pc.config('RawF',123)

    def test_seed_all_reproduces_frozen_classifier_initialization(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.config=SimpleNamespace(hidden_size=8);self.emb=nn.Embedding(50,8)
            def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
        def make(seed):
            pc.seed_all(seed);return Roberta27(Encoder())
        a=make(123);b=make(123);c=make(2026)
        self.assertEqual(pc.state_hash(a.state_dict()),pc.state_hash(b.state_dict()))
        self.assertNotEqual(pc.state_hash(a.state_dict()),pc.state_hash(c.state_dict()))
        x=dict(input_ids=torch.tensor([[0,2],[0,3]]),attention_mask=torch.ones(2,2,dtype=torch.long),labels=torch.ones(2,27))
        a.eval();b.eval();self.assertTrue(torch.equal(a(**x)['logits'],b(**x)['logits']))
        self.assertTrue(torch.equal(a(**x)['loss'],nn.functional.binary_cross_entropy_with_logits(a(**x)['logits'],x['labels'])))

    def test_runtime_accepts_real_new_hashseed_and_preserves_frozen_environment_gates(self):
        from contextlib import ExitStack
        packages={p:'pinned' for p in ['torch','transformers','numpy','scikit-learn','tokenizers','safetensors']}
        expected=dict(host='D3020-A3',python=pc.sys.version,executable=pc.sys.executable,packages=packages,
            pip_freeze='evidence',gpu='RTX 4090',cuda_runtime=torch.version.cuda)
        with ExitStack() as stack:
            for obj,name,value in [(pc.a0.platform,'system','Linux'),(pc.a0.platform,'node','D3020-A3'),
                                   (torch.cuda,'is_available',True),(torch.cuda,'device_count',1),
                                   (torch.cuda,'get_device_name','RTX 4090'),(pc.a0,'version','pinned'),(pc.a0,'capture','evidence')]:
                stack.enter_context(patch.object(obj,name,return_value=value))
            stack.enter_context(patch.object(pc,'REPO',pc.Path('/home/zixinfeng/Documents/context-aware-hyperbolic-emotion')))
            stack.enter_context(patch.object(pc,'read',return_value=dict(runtime=expected)))
            stack.enter_context(patch.dict(pc.os.environ,dict(PYTHONHASHSEED='2026',CUBLAS_WORKSPACE_CONFIG=':4096:8')))
            self.assertEqual(pc.runtime()['env']['PYTHONHASHSEED'],'2026')
            self.assertEqual(pc.os.environ['PYTHONHASHSEED'],'2026')
            with patch.dict(pc.os.environ,dict(PYTHONHASHSEED='7')):
                with self.assertRaises(RuntimeError):pc.runtime()
            with patch.object(torch.cuda,'get_device_name',return_value='RTX 3090'):
                with self.assertRaises(RuntimeError):pc.runtime()
            with patch.object(pc.a0,'version',return_value='changed'):
                with self.assertRaises(RuntimeError):pc.runtime()

    def test_statistics_use_sample_sd_and_paired_seed_differences(self):
        summary,paired=pr.compute_statistics(nine_records())
        r=next(r for r in summary if r['variant']=='A0' and r['metric']=='macro_f1')
        self.assertAlmostEqual(r['mean'],.4);self.assertAlmostEqual(r['sample_sd_ddof1'],.1)
        self.assertNotAlmostEqual(r['sample_sd_ddof1'],np.std([.3,.4,.5],ddof=0))
        p=next(r for r in paired if r['comparison']=='N1-A1S' and r['metric']=='macro_f1')
        self.assertAlmostEqual(p['mean_delta_pp'],2.);self.assertAlmostEqual(p['sample_sd_delta_pp_ddof1'],0.)
        self.assertEqual(p['positive_seeds'],3)

    def test_statistics_keep_negative_and_tied_seeds(self):
        data=nine_records()
        for r in data:
            if r['variant']=='N1':r['macro_f1']+=dict(zip(pc.ALL_SEEDS,[-.05,-.02,0.]))[r['seed']]
        p=next(r for r in pr.compute_statistics(data)[1] if r['comparison']=='N1-A1S' and r['metric']=='macro_f1')
        self.assertEqual((p['positive_seeds'],p['tied_seeds'],p['negative_seeds']),(1,1,1))
        self.assertAlmostEqual(p['mean_delta_pp'],-1/3)

    def test_duplicate_missing_or_unregistered_cells_rejected(self):
        a=nine_records()
        for rows in [a[:-1],a+[a[0]],a[:-1]+[a[0]]]:
            with self.assertRaises(RuntimeError):pr.compute_statistics(rows)
        a[0]['seed']=9
        with self.assertRaises(RuntimeError):pr.compute_statistics(a)

    def test_nested_success_seals_are_bound_and_logs_do_not_change_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pc.Path(tmp);pc.write(root/'SUCCESS.json',dict(status='parent'))
            pc.write(root/'child/SUCCESS.json',dict(status='child'))
            before=pc.artifacts(root);self.assertNotIn('SUCCESS.json',before);self.assertIn('child/SUCCESS.json',before)
            (root/'suite.log').write_text('new log',encoding='utf-8');self.assertEqual(before,pc.artifacts(root))
            pc.write(root/'child/SUCCESS.json',dict(status='tampered'));self.assertNotEqual(before,pc.artifacts(root))

    def test_frozen_source_inventory_is_unchanged(self):
        for name,expected in pc.read(pc.HERE/'FROZEN_SOURCE_REFERENCE.json').items():
            current={p.name:pc.sha(p) for p in (pc.HERE.parent/name).iterdir() if p.is_file()}
            self.assertEqual(current,expected)
        ref=pc.read(pc.HERE/'RECOVERY_REFERENCE.json')
        self.assertEqual({p.name:pc.sha(p) for p in (pc.HERE.parent/ref['old_source']).iterdir() if p.is_file()},ref['old_source_sha256'])

    def fake_saved_history(self,tmp):
        roots={kind:pc.Path(tmp)/kind for kind in pc.KINDS}
        orders={e:order for e,order in enumerate(pc.planned_orders([r['id'] for r in records()],42),1)}
        for kind,root in roots.items():
            root.mkdir();files={}
            for epoch,order in orders.items():
                name=f'epoch{epoch}_first_batch.json' if kind=='A0' else f'train_order_epoch{epoch}.json'
                pc.write(root/name,dict(ids=order[:16],batch_size=16) if kind=='A0' else order)
                files[name]=pc.sha(root/name)
            pc.write(root/'SUCCESS.json',dict(artifacts_sha256=files))
        return roots,orders

    def test_A0_first_batch_only_history_is_accepted_and_not_filled_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots,orders=self.fake_saved_history(tmp)
            before={str(p):p.read_bytes() for p in pc.Path(tmp).rglob('*') if p.is_file()}
            report=pc.historical_order_audit(orders,roots)
            self.assertFalse(report['A0_full_order_saved']);self.assertTrue(report['A1S_N1_full_epoch_orders_checked'])
            self.assertEqual(len(report['checks']),12)
            self.assertEqual([r['observed_id_count'] for r in report['checks'] if r['variant']=='A0'],[16]*4)
            self.assertEqual(before,{str(p):p.read_bytes() for p in pc.Path(tmp).rglob('*') if p.is_file()})

    def test_required_A1S_or_N1_full_history_cannot_be_skipped(self):
        for kind in ['A1S','N1']:
            with tempfile.TemporaryDirectory() as tmp:
                roots,orders=self.fake_saved_history(tmp);(roots[kind]/'train_order_epoch2.json').unlink()
                with self.assertRaisesRegex(RuntimeError,'Missing required sealed history'):pc.historical_order_audit(orders,roots)

    def test_wrong_A0_first_batch_stops_even_when_its_seal_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots,orders=self.fake_saved_history(tmp);root=roots['A0'];p=root/'epoch3_first_batch.json'
            pc.write(p,dict(ids=list(reversed(orders[3][:16]))))
            proof=pc.read(root/'SUCCESS.json');proof['artifacts_sha256'][p.name]=pc.sha(p);pc.write(root/'SUCCESS.json',proof)
            with self.assertRaisesRegex(RuntimeError,'Historical order mismatch'):pc.historical_order_audit(orders,roots)

    def test_changed_historical_evidence_stops_without_resealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots,orders=self.fake_saved_history(tmp);pc.write(roots['A0']/'epoch1_first_batch.json',dict(ids=['tampered']))
            with self.assertRaisesRegex(RuntimeError,'Changed sealed order evidence'):pc.historical_order_audit(orders,roots)

    def test_failed_prepare_is_preserved_and_progressed_attempt_cannot_be_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=pc.read(pc.HERE/'RECOVERY_REFERENCE.json');repo=pc.Path(tmp);root=repo/ref['failed_suite'];source=root/'source';source.mkdir(parents=True)
            for p in (pc.HERE.parent/ref['old_source']).iterdir():
                if p.is_file():(source/p.name).write_bytes(p.read_bytes())
            (root/'exit_code.txt').write_text('1\n',encoding='utf-8')
            pc.write(root/'FAILURE_prepare.json',dict(phase='prepare',seed=None,variant=None,new_test=False,
                error="FileNotFoundError(2, 'No such file or directory')",traceback='exp6c0_roberta27_baseline/train_order_epoch1.json'))
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(pc,'REPO',repo):
                inventory=pc.recovery_inputs();self.assertEqual(set(inventory),set(before))
                self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
                (root/'seed123').mkdir()
                with self.assertRaisesRegex(RuntimeError,'progressed beyond failed prepare'):pc.recovery_inputs()

    def test_summary_writer_includes_all_seeds_labels_and_grief_decomposition(self):
        # Synthetic metric inputs exercise the report, never saved as scientific results.
        with tempfile.TemporaryDirectory() as tmp:
            root=pc.Path(tmp);old=root/'seed42_reused';old.mkdir()
            names=[f'l{i}' for i in range(20)]+sorted(pc.a0.RARE)
            self.assertEqual(len(names),27);pc.write(root/'dataset_manifest.json',dict(names27=names))
            for r in nine_records():
                s,k=r['seed'],r['variant'];rows=[dict(label=n,support=10,precision=.5,recall=.5,f1=r['macro_f1'],average_precision=r['macro_f1'],tp=5,fp=5,fn=5) for n in names]
                if s==42:
                    pc.write(old/(k+'_metrics.json'),r);pc.table(old/(k+'_per_label.csv'),rows)
                    pc.write(old/(k+'_PROVENANCE.json'),dict(checkpoint_selection=dict(epoch=4),original_run='synthetic-only'))
                else:
                    cell=root/f'seed{s}'/k;cell.mkdir(parents=True);pc.write(cell/'validation_metrics.json',r);pc.table(cell/'validation_per_label.csv',rows)
                    pc.write(cell/'checkpoint_selection.json',dict(epoch=4))
            with patch.object(pr,'run_path',lambda suite,seed,kind:suite/f'seed{seed}'/kind):pr.write_reports(root)
            self.assertEqual(len(pc.csvrows(root/'EXP6D2_NINE_RUN_VALIDATION.csv')),9)
            self.assertEqual(len(pc.csvrows(root/'EXP6D2_PER_LABEL_ALL_SEEDS.csv')),243)
            self.assertEqual(len(pc.csvrows(root/'EXP6D2_RARE_ALL_SEEDS.csv')),63)
            self.assertEqual(len(pc.csvrows(root/'EXP6D2_PAIRED_DELTAS.csv')),18)
            for r in pc.csvrows(root/'EXP6D2_GRIEF_CONTRIBUTION.csv'):
                self.assertAlmostEqual(float(r['macro_delta_pp']),float(r['grief_macro_contribution_pp'])+float(r['other26_macro_contribution_pp']))
            self.assertIn('本轮无新Test',(root/'EXP6D2_SUMMARY_CN.md').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
