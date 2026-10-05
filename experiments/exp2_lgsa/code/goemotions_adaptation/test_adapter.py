"""CPU synthetic tests; no real pretrained BERT training on the Mac/login node."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from common import ROOT, PACKAGE, config, NAMES, write, sha
from hierarchy_goemotions import load_hierarchy, targets
from data_goemotions import encode, sample, TensorRows
from geometry import accept
from evaluation import metrics, select_threshold


class FakeTokenizer:
    vocab_size = 200
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    def __call__(self, text, **kwargs):
        # Simulate case-sensitive tokenization and text truncation with preserved CLS/SEP.
        ids = [150 if word[0].isupper() else 151 for word in text.split()]
        return {'input_ids': [101] + ids[:kwargs['max_length']-2] + [102]}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.h = load_hierarchy(PACKAGE / 'hierarchy.json')

    def test_cross_parent_multilabel_and_mask(self):
        target, fine = targets([17, 19, 18], self.h)
        matrix = torch.tensor(target).reshape(2,32)
        self.assertEqual(matrix[0,28:].tolist(), [1,1,0,0])
        self.assertTrue((matrix[0,:28] == -100).all())
        self.assertTrue((matrix[1,28:] == -100).all())
        self.assertEqual(torch.where(matrix[1] == 1)[0].tolist(), [17,18,19])
        self.assertEqual(sum(fine),3)
        self.assertEqual(len(self.h['edges']),60)

    def test_neutral_and_all_labels(self):
        target, _ = targets([27], self.h)
        self.assertEqual(target[28:32], [0,0,0,1])
        target, fine = targets(list(range(28)), self.h)
        self.assertEqual(sum(v == 1 for v in target),32)
        self.assertEqual(sum(fine),28)
        for bad in ([], [1,1], [28], [-1]):
            with self.assertRaises(RuntimeError): targets(bad,self.h)

    def test_bad_hierarchy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = copy.deepcopy(self.h['source']); obj['labels'][0]['coarse_id'] = 0
            p = Path(directory)/'hierarchy.json'; write(p,obj)
            with self.assertRaises(RuntimeError): load_hierarchy(p)

    def test_prompt_mask_case_and_text_budget(self):
        tokenizer = FakeTokenizer()
        for text in ['Hello hello', ' '.join(['Word']*100)]:
            result = encode({'text':text,'labels':[17,19]},tokenizer,self.h)
            self.assertEqual(len(result['input_ids']),54)
            active = [i for i,a in zip(result['input_ids'],result['attention_mask']) if a]
            self.assertEqual(active[-5:],[232,234,233,234,102])
            self.assertEqual(active[:-5],tokenizer(text,max_length=50)['input_ids'][:-1])
        self.assertEqual(encode({'text':'Hello hello','labels':[0]},tokenizer,self.h)['input_ids'][:3],[101,150,151])

    def test_seeded_samples_preserve_input(self):
        original = [{'id':str(i)} for i in range(2000)]
        subset = sample(original,1000,42)
        self.assertEqual(subset,sample(original,1000,42)); self.assertEqual(len(original),2000)
        self.assertEqual(len({r['id'] for r in subset}),1000)

    def test_metrics_against_sklearn_and_threshold_tie(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        rng=np.random.default_rng(1); y=rng.integers(0,2,size=(31,28)); scores=rng.random((31,28))
        result,_=metrics(y,scores,.3,list(range(7)))
        for average in ['micro','macro']:
            for key,fn in [('f1',f1_score),('precision',precision_score),('recall',recall_score)]:
                self.assertAlmostEqual(result[average+'_'+key],fn(y,scores>=.3,average=average,zero_division=0))
        selected=select_threshold(np.zeros((2,28)),np.zeros((2,28)),[.3,.1,.2],list(range(7)))
        self.assertEqual(selected['selected_threshold'],.1)
        with self.assertRaises(RuntimeError): metrics(np.zeros((2,32)),np.zeros((2,32)),.3,[1])

    def test_actual_hgs_and_hard_geometry_failure(self):
        path=ROOT/'working_lgsa/models/prompt.py'
        tree=ast.parse(path.read_text())
        methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in
                 ['_gram_schmidt','semantically_anchored_gof_initialization']]
        scope={'torch':torch}; exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
        class Toy: pass
        for method in methods: setattr(Toy,method.name,scope[method.name])
        g=torch.Generator().manual_seed(42); initial=torch.randn(32,64,generator=g)
        result=Toy().semantically_anchored_gof_initialization(initial,self.h['depths'],self.h['parents'])
        accepted=accept(initial,result,self.h,self.h['parents'],config())
        self.assertEqual(accepted['status'],'GOEMOTIONS_HGS_PASS')
        with self.assertRaises(RuntimeError): accept(initial,initial,self.h,self.h['parents'],config())
        with self.assertRaises(RuntimeError): accept(initial,result,self.h,{0:1},config())
        broken=result.clone(); broken[29]=broken[28]
        with self.assertRaises(RuntimeError): accept(initial,broken,self.h,self.h['parents'],config())

    def test_original_multilabel_loss_retained(self):
        tree=ast.parse((ROOT/'working_lgsa/models/loss.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='multilabel_categorical_crossentropy')
        scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream-loss','exec'),scope)
        y=torch.tensor([targets([17,19],self.h)[0],targets([0,27],self.h)[0]]).view(-1,32)
        z=torch.arange(128,dtype=torch.float32).view(4,32)/100
        actual=scope[fn.name](y,z)
        manual=[]
        for row in range(2):
            active_y=y[row*2:row*2+2][y[row*2:row*2+2]!=-100]
            active_z=z[row*2:row*2+2][y[row*2:row*2+2]!=-100]
            manual.append(torch.logsumexp(torch.cat([active_z[active_y==0],torch.zeros(1)]),0)+
                          torch.logsumexp(torch.cat([-active_z[active_y==1],torch.zeros(1)]),0))
        torch.testing.assert_close(actual,torch.stack(manual).mean())

    def test_epoch_resume_order_and_rng(self):
        from run import loader, rng_state, restore_rng
        data={'input_ids':torch.arange(100).reshape(50,2)}
        torch.manual_seed(7)
        state=torch.get_rng_state().clone()
        all_batches=list(loader(data,16,2,True))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        continued=list(loader(data,16,2,True))[2:]
        self.assertTrue(torch.equal(all_batches[2]['input_ids'],continued[0]['input_ids']))


class GateTests(unittest.TestCase):
    def test_success_evidence_and_artifact_tamper(self):
        import gates
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'smoke';out.mkdir()
            write(root/'assets/goemotions-v0/manifest.json',{'fixture':True})
            proof={'status':'GOEMOTIONS_SMOKE_PASS','phase':'smoke',
                'config_sha256':sha(PACKAGE/'lgsa_goemotions_seed42.json'),
                'dataset_manifest_sha256':sha(root/'assets/goemotions-v0/manifest.json'),
                'adapter_source':{'fixture':True},'model':{'fixture':True},'train_examples':1000,
                'validation_examples':500,'test_used':False,'optimizer_steps':63,
                'checkpoint_reload_logits_match':True,'backbone_gradient_passed':True,
                'gat_gradient_passed':True,'finite_training_passed':True,
                'backbone_parameter_max_change':.001,'initialization':'raw_pretrained','initial_checkpoint':None}
            write(out/'geometry_acceptance.json',{'status':'GOEMOTIONS_HGS_PASS','map_matches':True,
                'fine_vectors_changed':True,'top_max_abs_cosine':1e-7,'families':{'28':{'rank':12,'children':12,'max_abs_cosine':1e-7}}})
            for name in ['checkpoint_best.pt','checkpoint_reload.json','validation_fixed_0.30_metrics.json','sample_ids.json','steps.jsonl']:
                (out/name).write_text('synthetic unit-test artifact, not a real run')
            proof['artifacts_sha256']={p.name:sha(p) for p in out.iterdir()}
            write(out/'SUCCESS.json',proof);write(out/'experiment_metadata.json',proof)
            with patch.object(gates,'ROOT',root),patch.object(gates,'source_record',return_value={'fixture':True}):
                gates.smoke_gate(out,{'model':{'fixture':True}})
                geo=json.loads((out/'geometry_acceptance.json').read_text());geo['fine_vectors_changed']=False
                write(out/'geometry_acceptance.json',geo)
                with self.assertRaises(RuntimeError): gates.smoke_gate(out,{'model':{'fixture':True}})

    def test_missing_or_wrong_phase_never_passes(self):
        import gates
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with self.assertRaises(FileNotFoundError): gates.smoke_gate(out,{})
            write(out/'SUCCESS.json',{'status':'GOEMOTIONS_FULL_PASS','phase':'full'})
            write(out/'experiment_metadata.json',{'status':'GOEMOTIONS_FULL_PASS','phase':'full'})
            with self.assertRaises(RuntimeError): gates.smoke_gate(out,{})


if __name__ == '__main__':
    unittest.main()
