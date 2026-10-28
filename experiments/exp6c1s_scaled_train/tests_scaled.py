"""Synthetic CPU contract tests. No actual RoBERTa training/GPU claims."""
import ast,copy,hashlib,json,math,random,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scaled_model import install_scale
from scommon import extended_score,probabilities,read,write,table,a1lib,REPO
from a1model import A1Model,loss_components,train_step,parameter_groups
from model27 import Roberta27
from probes import isolated_eval,run_probe
from reporting import write_reports

NAMES=['grief','pride','relief','nervousness','embarrassment','remorse','fear']+[f'emotion{i}' for i in range(20)]

def extract(p,names,ns):
    tree=ast.parse(p.read_text(encoding='utf-8'));body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(body)==len(names);exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),ns)

def actual_HNN():
    # Digest-pinned source definitions, with explicitly synthetic CPU imported
    # tanh/artanh stand-ins only. Ubuntu uses original math_utils helpers.
    src=REPO/'results/exp6c1s_input_scaling_diagnostics/run_20261021T133415Z_3VyOQEzH/source_review'
    reference=read(REPO/'experiments/exp6c1_roberta27_hypaux/UPSTREAM_REFERENCE.json')
    ns=dict(torch=torch,nn=nn,F=F,math=math,init=nn.init,Module=nn.Module,Manifold=object,tanh=torch.tanh,artanh=torch.atanh)
    for path,names in [('manifolds/poincare.py',{'PoincareBall'}),('layers/hyp_layers.py',{'get_dim_act_curv','HNNLayer','HypLinear','HypAct'})]:
        p=src/path.replace('/','__');assert hashlib.sha256(p.read_bytes()).hexdigest()==reference[path];extract(p,names,ns)
    ns.update(hyp_layers=SimpleNamespace(**{n:ns[n] for n in ['get_dim_act_curv','HNNLayer']}),manifolds=SimpleNamespace(PoincareBall=ns['PoincareBall']))
    p=src/'models__encoders.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==reference['models/encoders.py'];extract(p,{'Encoder','HNN'},ns)
    args=SimpleNamespace(num_layers=2,feat_dim=8,dim=100,c=1.,device='cpu',act='relu',dropout=.1,bias=True,manifold='PoincareBall')
    return ns['HNN'](torch.tensor([1.]),args),ns['PoincareBall']()

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.embeddings=nn.Embedding(10,8);self.encoder=nn.Module()
        self.encoder.layer=nn.ModuleList([nn.Linear(8,8) for _ in range(12)]);self.config=SimpleNamespace(hidden_size=8)
    def forward(self,input_ids,attention_mask,return_dict=True):
        h=self.embeddings(input_ids)
        for layer in self.encoder.layer:h=h+.05*torch.tanh(layer(h))
        return SimpleNamespace(last_hidden_state=h)
class Decoder(nn.Module):
    def __init__(self,manifold):
        super().__init__();self.manifold=manifold;self.c=torch.tensor([1.]);self.cls=nn.Linear(100,27)
    def decode(self,z):return self.cls(self.manifold.logmap0(z,self.c))
class Hyper(nn.Module):
    def __init__(self):
        super().__init__();self.encoder,m=actual_HNN();self.decoder=Decoder(m);self.inputs=[]
    def encode(self,h):self.inputs.append(h.detach().clone());return self.encoder.encode(h)
class Tokenizer:
    def __call__(self,texts,**kwargs):
        return dict(input_ids=torch.tensor([[0,3,2] for _ in texts]),attention_mask=torch.ones(len(texts),3,dtype=torch.long))
def toy():
    torch.manual_seed(42);return A1Model(Roberta27(Encoder()),Hyper(),torch.randn(27,100)*.01)
def batch(n=16):
    y=torch.zeros(n,27)
    for i in range(n):y[i,[i%27,(i+1)%27]]=1
    return dict(input_ids=torch.tensor([[0,3,2] for _ in range(n)]),attention_mask=torch.ones(n,3,dtype=torch.long),labels=y)

class ScaledTests(unittest.TestCase):
    def test_only_aux_input_scaled_once_parameter_keys_and_rng_preserved(self):
        m=toy().eval();x=batch();original=m(**x);h=m.hyper.inputs[-1].clone()
        state={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone();ids=[id(p) for p in m.parameters()]
        install_scale(m);changed=m(**x)
        self.assertTrue(torch.equal(m.hyper.inputs[-1],.03*h));self.assertTrue(torch.equal(changed['logits'],original['logits']))
        self.assertTrue(torch.equal(changed['components']['classification'],original['components']['classification']))
        self.assertEqual(ids,[id(p) for p in m.parameters()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,t in m.state_dict().items():self.assertTrue(torch.equal(t,state[k]))
        with self.assertRaisesRegex(RuntimeError,'repeated'):install_scale(m)

    def test_training_logits_and_random_draw_consumption_match_unscaled(self):
        m=toy().train();x=batch();rng=torch.get_rng_state().clone();a=m(**x)['logits'];after=torch.get_rng_state().clone()
        torch.set_rng_state(rng);install_scale(m);b=m(**x)['logits']
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_formula_multi_positive_reduction_and_gradient_paths(self):
        m=install_scale(toy());out=m(**batch());parts=out['components'];y=batch()['labels'];q=y/y.sum(-1,keepdim=True)
        expected=(-(q*F.log_softmax(out['decoder_logits'],-1)).sum(-1)-(q*(out['z']@m.emotion_embeddings.T)).sum(-1)).mean()
        torch.testing.assert_close(expected,parts['auxiliary'],atol=0,rtol=0)
        named=list(m.named_parameters());params=[p for _,p in named]
        gb=torch.autograd.grad(parts['classification'],params,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(.1*parts['auxiliary'],params,allow_unused=True)
        for (n,p),b,a in zip(named,gb,ga):
            if n.startswith('hyper.'):self.assertIsNone(b);self.assertIsNotNone(a)
            elif n.startswith('control.classifier_'):self.assertIsNone(a);self.assertIsNotNone(b)
            else:self.assertIsNotNone(a);self.assertIsNotNone(b)
            if a is not None:self.assertTrue(torch.isfinite(a).all())
        self.assertIsNone(m.emotion_embeddings.grad)

    def test_probe_restores_all_RNG_and_mixed_module_modes_even_on_error(self):
        m=toy().train();m.control.encoder.encoder.layer[0].eval();flags=[p.training for p in m.modules()]
        random.seed(3);np.random.seed(4);cpu=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
        with self.assertRaisesRegex(ValueError,'intentional'):
            with isolated_eval(m):
                random.random();np.random.rand();torch.rand(7);raise ValueError('intentional')
        self.assertEqual(flags,[p.training for p in m.modules()]);self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))

    def test_actual_probe_keeps_parameters_gradients_modes_and_scale(self):
        m=install_scale(toy()).train();before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        records=[dict(id=f'id{i}',text='synthetic',labels=[i%27,(i+1)%27]) for i in range(16)]
        selection=dict(cohorts={'multi_positive':[dict(id=r['id']) for r in records]})
        with tempfile.TemporaryDirectory() as tmp:
            pooled,_=run_probe(Path(tmp),'fixture',m,Tokenizer(),dict(train=records),selection)
            proof=read(Path(tmp)/'geometry/fixture/NO_UPDATE_PROOF.json');self.assertEqual(proof['state_before'],proof['state_after'])
            self.assertEqual(pooled['unique_ids'],16);self.assertEqual(pooled['first_clip_fraction'],0)
        self.assertTrue(m.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        for k,t in m.state_dict().items():self.assertTrue(torch.equal(t,before[k]))
        # The trace restored the permanent wrapper, so later training is scaled.
        m(**batch());self.assertTrue(torch.isfinite(m.hyper.inputs[-1]).all())

    def test_parameter_groups_identical_and_original_train_step_works(self):
        from transformers import get_linear_schedule_with_warmup
        a=toy();opt0,groups0=parameter_groups(a);del opt0
        b=install_scale(toy());opt,groups1=parameter_groups(b);self.assertEqual(groups0,groups1)
        scheduler=get_linear_schedule_with_warmup(opt,765,7648);old_tensor_to=torch.Tensor.to
        def cpu_to(t,*args,**kwargs):
            if args and args[0]=='cuda':args=('cpu',)+args[1:]
            return old_tensor_to(t,*args,**kwargs)
        old_e=b.emotion_embeddings.clone()
        with patch.object(torch.Tensor,'to',cpu_to):
            first=batch();first['ids']=[f'id{i}' for i in range(16)]
            self.assertEqual(opt.param_groups[0]['lr'],0);row=train_step(b,first,opt,scheduler,detailed=True)
            self.assertGreater(opt.param_groups[0]['lr'],0);self.assertTrue(all(r['norm']>0 for r in row['gradients'].values()))
            second=batch();second['ids']=[f'id{i}' for i in range(16)];train_step(b,second,opt,scheduler,detailed=True)
        self.assertTrue(torch.equal(b.emotion_embeddings,old_e));self.assertTrue(all(p.grad is None for p in b.parameters()))

    def test_fixed_threshold_AP_ties_and_gold_positive_probability(self):
        y=np.zeros((4,27));p=np.full((4,27),.1);y[[0,1],0]=1;p[:,0]=[.3,.8,.3,.2]
        metrics,rows,_,pred=extended_score(y,p,NAMES)
        self.assertEqual(rows[0]['tp'],2);self.assertEqual(rows[0]['fp'],1);self.assertEqual(pred[0,0],1)
        self.assertAlmostEqual(rows[0]['mean_positive_probability'],.55)
        self.assertAlmostEqual(rows[0]['average_precision'],(1+2/3)/2)
        self.assertIsNone(rows[1]['average_precision']);self.assertEqual(metrics['average_precision_defined_labels'],1)

    def test_strict_macro_selection_preserves_earliest_tie(self):
        history=[dict(epoch=1,validation_macro_f1=.5),dict(epoch=2,validation_macro_f1=.6),dict(epoch=3,validation_macro_f1=.6),dict(epoch=4,validation_macro_f1=.55)]
        self.assertEqual(max(history,key=lambda r:r['validation_macro_f1'])['epoch'],2)

    def test_reloaded_scaled_state_preserves_exact_logits(self):
        a=install_scale(toy()).eval();x=batch();ref=a(**x)['logits'].detach()
        b=install_scale(toy()).eval();b.load_state_dict(a.state_dict(),strict=True)
        self.assertTrue(torch.equal(ref,b(**x)['logits']));self.assertEqual(set(a.state_dict()),set(toy().state_dict()))

    def test_no_test_training_path_and_frozen_dependencies(self):
        import srun
        source=Path(srun.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in ['test','tested','formal_test'] for n in calls))
        self.assertIn("del data['test']",source)
        a1lib.check_sources()

    def test_original_saved_validation_values_recompute_on_CPU(self):
        import scommon
        evidence=REPO/'results/exp6c1d_hyperbolic_diagnostics/run_20261021_cpu_recovery_v1_2/recovered_from_cLRnltwT_v2/evidence'
        if not evidence.exists():self.skipTest('Local archived Validation fixtures only')
        with np.load(evidence/'A0/validation_predictions.npz',allow_pickle=False) as z:names=z['label_names'].tolist()
        with tempfile.TemporaryDirectory() as tmp,patch.object(scommon,'A0',evidence/'A0'),patch.object(scommon,'A1',evidence/'A1'):
            result=scommon.historical_validation(Path(tmp),names)
            self.assertAlmostEqual(result['A0']['metrics']['macro_f1']*100,58.1356,places=4)
            self.assertAlmostEqual(result['A1']['metrics']['macro_f1']*100,56.0209,places=4)

    def test_complete_UTF8_report_writes_f1_AP_geometry_and_negative_result(self):
        y=np.eye(27);old0=np.full((27,27),.1);old1=old0.copy();new=old0.copy()
        np.fill_diagonal(old0,.4);np.fill_diagonal(old1,.4);np.fill_diagonal(new,.4)
        old0[0,0]=.2;old1[[0,1],[0,1]]=.2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'historical_validation').mkdir()
            for tag,p in [('A0',old0),('A1',old1)]:
                m,r,_,_=extended_score(y,p,NAMES);write(root/'historical_validation'/(tag+'_metrics.json'),m);table(root/'historical_validation'/(tag+'_per_label.csv'),r)
            write(root/'checkpoint_selection.json',dict(epoch=2))
            for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
                d=root/'geometry'/tag;d.mkdir(parents=True)
                write(d/'pooled.json',dict(tag=tag,first_clip_fraction=0.,later_clip_fraction=0.,final_fraction_above_099=0.,
                    final_mean_norm=.75,final_norm_IQR=.1,median_shared_aux_to_BCE_gradient_norm=.3,median_shared_gradient_cosine=.5))
                table(d/'cohorts.csv',[dict(tag=tag)])
            m,r,_,_=extended_score(y,new,NAMES);write(root/'validation_metrics.json',m);table(root/'validation_per_label.csv',r)
            write_reports(root);text=(root/'EXP6C1S_TRAIN_SUMMARY.md').read_text(encoding='utf-8')
            self.assertIn('初步收益',text);self.assertIn('新Test推理=0',text);self.assertIn('macro_average_precision',text)
            # Rewriting only this temporary fixture checks the negative branch.
            m,r,_,_=extended_score(y,np.full_like(y,.1),NAMES);write(root/'validation_metrics.json',m);table(root/'validation_per_label.csv',r)
            write_reports(root);self.assertIn('单独缩放没有改善',(root/'EXP6C1S_TRAIN_SUMMARY.md').read_text(encoding='utf-8'))

    def test_smoke_seal_rejects_artifact_tampering(self):
        import srun
        with tempfile.TemporaryDirectory() as tmp,patch.object(srun,'source_hashes',return_value={'fixture':'unchanged'}):
            root=Path(tmp);(root/'smoke').mkdir()
            write(root/'plan.json',dict(fixture=True));write(root/'INITIALIZATION_PARITY.json',dict(fixture=True))
            write(root/'SCALED_INITIALIZATION_PARITY.json',dict(fixture=True));write(root/'config.json',srun.CONFIG)
            write(root/'smoke/SMOKE_REPORT.json',dict(status='PASS'))
            write(root/'smoke/SMOKE_VERIFIED.json',dict(status='EXP6C1S_SMOKE_PASS',artifacts_sha256=srun.smoke_hashes(root/'smoke'),
                plan_sha256=srun.sha(root/'plan.json'),source_sha256={'fixture':'unchanged'},initialization_sha256=srun.sha(root/'INITIALIZATION_PARITY.json'),
                scaled_initialization_sha256=srun.sha(root/'SCALED_INITIALIZATION_PARITY.json')))
            srun.verify_smoke(root)
            write(root/'smoke/SMOKE_REPORT.json',dict(status='CHANGED'))
            with self.assertRaisesRegex(RuntimeError,'Smoke artifacts differ'):srun.verify_smoke(root)

    def test_review_export_preserves_sealed_files_and_declares_missing_checkpoint(self):
        import srun
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'SUCCESS.json',dict(fixture=True));write(root/'中文.json',dict(message='几何诊断'))
            (root/'best_checkpoint').mkdir();(root/'best_checkpoint/model.pt').write_bytes(b'synthetic checkpoint')
            before=(root/'SUCCESS.json').read_bytes()
            subprocess.run([sys.executable,'-X','utf8',str(Path(srun.__file__).with_name('export_review.py')),'--root',str(root)],check=True,capture_output=True)
            self.assertEqual(before,(root/'SUCCESS.json').read_bytes())
            with tarfile.open(root/'review/Exp6C1S_Train_review.tar.gz') as archive:
                self.assertNotIn('best_checkpoint/model.pt',archive.getnames())
                export=json.loads(archive.extractfile('REVIEW_EXPORT.json').read().decode('utf-8'))
                self.assertEqual(export['omitted_large_checkpoints'],['best_checkpoint/model.pt'])
                self.assertIn('中文.json',export['included_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
