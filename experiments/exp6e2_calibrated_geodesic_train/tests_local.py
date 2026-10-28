"""CPU contract tests with toy parameters; NOT a RoBERTa/GPU training result."""
import ast
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

import crun
import ccommon as cc
import creport
from cmodel import GAMMA,GAMMA_HEX,install_weight
from cgeometry import ProjectionMonitor
from gmath import components


class ToyBall:
    eps={torch.float32:.004,torch.float64:1e-5}
    min_norm=1e-15
    def proj(self,x,c):
        r=x.norm(dim=-1,keepdim=True).clamp_min(self.min_norm)
        cap=(1-self.eps[x.dtype])/(c**.5)
        return torch.where(r>cap,x/r*cap,x)


class ToyLayer(nn.Module):
    def __init__(self,manifold):
        super().__init__();self.manifold=manifold;self.affine=nn.Linear(8,100)
    def forward(self,x):return self.manifold.proj(self.affine(x),1.)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.manifold=ToyBall();self.layer=ToyLayer(self.manifold)
    def encode(self,h):
        r=h.norm(dim=-1,keepdim=True).clamp_min(1e-15)
        x=self.manifold.proj(torch.tanh(r)*h/r,1.)
        return self.layer(x)


class ToyHyper(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=ToyEncoder();self.decoder=nn.Linear(100,27)
    def encode(self,h):return self.encoder.encode(h)


class ToyControl(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(5,8)
        self.classifier_weight=nn.Parameter(torch.randn(27,8)*.02)
        self.classifier_bias=nn.Parameter(torch.zeros(27));self.dropout=nn.Dropout(.1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.control=ToyControl();self.hyper=ToyHyper()
        self.register_buffer('emotion_embeddings',torch.randn(27,100)*.015)
        self._geodesic_alignment_installed=True
    def forward(self,input_ids,attention_mask=None,labels=None):
        h=self.control.encoder(input_ids)
        logits=torch.nn.functional.linear(self.control.dropout(h),self.control.classifier_weight,self.control.classifier_bias)
        z=self.hyper.encode(h);u=self.hyper.decoder(z)
        c=None if labels is None else components(logits,labels,z,u,self.emotion_embeddings)
        return dict(logits=logits,z=z,decoder_logits=u,components=c,loss=None if c is None else c['total'])


def batch(n):
    y=torch.zeros(n,27);y[:,0]=1;y[::2,7]=1
    return dict(input_ids=torch.randn(n,5),labels=y,ids=['id'+str(i) for i in range(n)])


class PilotTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_coefficient_matches_actual_calibration_exact_double(self):
        self.assertEqual(GAMMA.hex(),GAMMA_HEX)
        self.assertEqual(GAMMA,0.02091538317504222)
        self.assertNotEqual(GAMMA,.02)

    def test_install_keeps_rng_parameters_buffers_and_forward(self):
        model=ToyModel();x=batch(11);x.pop('ids')
        state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        model.eval();old=model(**x);install_weight(model);new=model(**x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(state),list(model.state_dict()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        for k in ['logits','z','decoder_logits']:self.assertTrue(torch.equal(old[k],new[k]))
        for k in ['classification','positive_ce','positive_distance','auxiliary']:
            self.assertTrue(torch.equal(old['components'][k],new['components'][k]))
        self.assertTrue(torch.equal(new['loss'],new['components']['classification']+GAMMA*new['components']['auxiliary']))

    def test_repeated_install_or_scaled_branch_rejected(self):
        m=ToyModel();install_weight(m)
        with self.assertRaises(RuntimeError):install_weight(m)
        m=ToyModel();m.hyper._exp6c1s_scale_installed=True
        with self.assertRaises(RuntimeError):install_weight(m)

    def test_classifier_bce_only_and_fixed_labels(self):
        m=install_weight(ToyModel()).eval();x=batch(16);x.pop('ids');o=m(**x)
        p=m.control.classifier_weight
        gb=torch.autograd.grad(o['components']['classification'],p,retain_graph=True)[0]
        ga=torch.autograd.grad(o['components']['auxiliary'],p,allow_unused=True,retain_graph=True)[0]
        gt=torch.autograd.grad(o['loss'],p)[0]
        self.assertIsNone(ga);self.assertTrue(torch.equal(gb,gt));self.assertIsNone(m.emotion_embeddings.grad)

    def test_decoder_ce_derivative_uses_gamma_and_distance_has_no_path(self):
        m=install_weight(ToyModel()).eval();x=batch(11);x.pop('ids');o=m(**x);c=o['components'];p=m.hyper.decoder.weight
        gc=torch.autograd.grad(GAMMA*c['positive_ce'].mean(),p,retain_graph=True)[0]
        gd=torch.autograd.grad(c['positive_distance'].mean(),p,retain_graph=True,allow_unused=True)[0]
        ga=torch.autograd.grad(GAMMA*c['auxiliary'],p)[0]
        self.assertIsNone(gd);self.assertTrue(torch.allclose(gc,ga,atol=1e-8,rtol=1e-6))

    def test_ce_distance_gradients_sum_to_actual_shared_aux(self):
        m=install_weight(ToyModel()).eval();x=batch(16);x.pop('ids');c=m(**x)['components'];p=m.control.encoder.weight
        ce=torch.autograd.grad(GAMMA*c['positive_ce'].mean(),p,retain_graph=True)[0]
        dist=torch.autograd.grad(GAMMA*c['positive_distance'].mean(),p,retain_graph=True)[0]
        combined=torch.autograd.grad(GAMMA*c['auxiliary'],p)[0]
        self.assertTrue(torch.isfinite(combined).all());self.assertGreater(float(combined.norm()),0)
        self.assertTrue(torch.allclose(ce+dist,combined,atol=2e-7,rtol=1e-5))

    def test_monitor_same_forward_rng_and_gradients_with_dropout(self):
        m=install_weight(ToyModel()).train();x=batch(16);x.pop('ids');rng=torch.get_rng_state().clone()
        old=m(**x);params=list(m.parameters());grad=torch.autograd.grad(old['loss'],params)
        after=torch.get_rng_state().clone();torch.set_rng_state(rng)
        with ProjectionMonitor(m,16) as monitor:new=m(**x);geo=monitor.values(new['z'])
        grad2=torch.autograd.grad(new['loss'],params)
        for key in ['logits','z','decoder_logits','loss']:self.assertTrue(torch.equal(old[key],new[key]))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(grad,grad2)))
        self.assertEqual(geo['text_projection_calls'],2)
        self.assertNotIn('proj',vars(m.hyper.encoder.manifold))

    def test_monitor_actual_first_later_counts_and_exception_restore(self):
        m=ToyModel();encoder=m.hyper.encoder
        with torch.no_grad():encoder.layer.affine.weight.zero_();encoder.layer.affine.bias.fill_(20.)
        h=torch.ones(11,8)*20.
        with ProjectionMonitor(m,11) as monitor:
            z=m.hyper.encode(h);values=monitor.values(z)
        self.assertEqual(values['first_text_clip_count'],11);self.assertEqual(values['later_text_clip_count'],11)
        self.assertLess(values['final_radius_max'],1)
        try:
            with ProjectionMonitor(m,11):raise ValueError('synthetic fault')
        except ValueError:pass
        self.assertNotIn('proj',vars(encoder.manifold))
        with self.assertRaises(RuntimeError):ProjectionMonitor(m,1)

    def test_actual_step_matches_manual_original_scaling_clip_optimizer(self):
        torch.manual_seed(7);a=install_weight(ToyModel()).eval()
        torch.manual_seed(7);b=install_weight(ToyModel()).eval()
        oa=torch.optim.AdamW(a.parameters(),lr=5e-5);ob=torch.optim.AdamW(b.parameters(),lr=5e-5)
        sa=get_linear_schedule_with_warmup(oa,765,7648);sb=get_linear_schedule_with_warmup(ob,765,7648)
        before={k:v.clone() for k,v in a.state_dict().items()}
        for i,n in enumerate([16,11]):
            x=batch(n);manual={k:v for k,v in x.items() if k!='ids'}
            expected=b(**manual)['loss'];(expected*n).backward()
            for p in b.parameters():
                if p.grad is not None:p.grad.div_(n)
            norm=torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
            ob.step();sb.step();ob.zero_grad(set_to_none=True)
            actual=crun.step(a,dict(x),oa,sa,detailed=True)
            self.assertEqual(actual['batch_size'],n);self.assertAlmostEqual(actual['loss'],float(expected.detach()),places=7)
            self.assertAlmostEqual(actual['unclipped_grad_norm'],float(norm),places=7)
            self.assertEqual(set(actual['gradients']),{'shared_encoder','classifier','hyperbolic_projection','auxiliary_decoder'})
            self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in a.parameters()))
        self.assertTrue(any(not torch.equal(v,a.state_dict()[k]) for k,v in before.items()))

    def test_loss_diagnostics_coefficient_and_weighted_components(self):
        m=install_weight(ToyModel()).eval();x=batch(11);x.pop('ids');out=m(**x);v=cc.diagnostic_values(out)
        self.assertAlmostEqual(v['weighted_auxiliary'],GAMMA*v['auxiliary'],places=14)
        self.assertAlmostEqual(v['weighted_ce']+v['weighted_distance'],v['weighted_auxiliary'],places=6)
        self.assertAlmostEqual(v['bce']+v['weighted_auxiliary'],v['loss'],places=6)

    def test_all_multi_positive_targets_contribute_no_negative_alignment(self):
        m=install_weight(ToyModel()).eval();x=batch(11);x.pop('ids');out=m(**x);c=out['components'];y=x['labels']
        d=c['distance_matrix'].float()
        expected=torch.stack([d[i,y[i].bool()].mean() for i in range(11)])
        self.assertTrue(torch.equal(c['positive_distance'],expected))

    def test_readonly_artifact_verifier_detects_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.pt').write_bytes(b'frozen');(root/'run.log').write_text('status')
            one=cc.artifact_hashes(root);(root/'run.log').write_text('more status');self.assertEqual(one,cc.artifact_hashes(root))
            (root/'model.pt').write_bytes(b'changed');self.assertNotEqual(one,cc.artifact_hashes(root))

    def test_all_seven_frozen_source_directories_unchanged(self):
        reference=cc.read(Path(cc.HERE)/'INPUT_REFERENCE.json')
        self.assertEqual(len(reference['frozen_source_directories']),7)
        for name,expected in reference['frozen_source_directories'].items():
            path=Path(cc.REPO)/'experiments'/name
            self.assertEqual({p.name:cc.sha(p) for p in path.iterdir() if p.is_file()},expected)

    def test_runner_fixed_budget_and_no_test_predict_contract(self):
        with self.assertRaises(RuntimeError):crun.predict(None,None,[None]*3821,'Test rejected')
        self.assertEqual(cc.CONFIG['seed'],42);self.assertEqual(cc.CONFIG['epochs'],4)
        self.assertEqual(cc.CONFIG['auxiliary_weight'],GAMMA);self.assertEqual(cc.CONFIG['hyperbolic_input_scale'],1.)
        self.assertFalse(cc.CONFIG['new_test_inference'])

    def test_negative_validation_report_keeps_all_labels_rare7_and_coefficients(self):
        # Temporary SYNTHETIC arrays exercise reporting only, no reported experiment.
        names=(cc.REPO/'数据集/GoEmotions/metadata/emotions.txt').read_text().splitlines()
        names=[n for n in names if n!='neutral']
        gold=cc.np.eye(27,dtype=cc.np.float32).repeat(2,axis=0)
        ids=cc.np.array(['synthetic_'+str(i) for i in range(len(gold))])
        manifest=dict(names27=names,splits={'validation':dict(retained_ids=ids.tolist())})
        with tempfile.TemporaryDirectory() as tmp:
            a0,a1,root=[Path(tmp)/n for n in ['A0','A1','candidate']]
            for directory,logits in [(a0,gold*6-3),(a1,gold*6-3),(root,cc.np.zeros_like(gold))]:
                directory.mkdir()
                cc.np.savez(directory/'validation_predictions.npz',logits=logits,targets=gold,ids=ids,label_names=cc.np.array(names))
                metrics=cc.extended_score(gold,cc.probabilities(logits),names)[0]
                cc.write(directory/'validation_metrics.json',metrics)
            cc.table(root/'training_log.csv',[dict(epoch=e,training_first_text_clip_fraction=1.,training_later_text_clip_fraction=.5,
                training_final_fraction_above099=.9,training_final_radius_mean=.993,global_gradient_clip_fraction=.7,gradient_norm_max=4.) for e in range(1,5)])
            for tag in ['fresh_initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
                directory=root/'probes'/tag;directory.mkdir(parents=True)
                cc.table(directory/'MODULE_GRADIENTS.csv',[dict(module='shared_encoder',geodesic_aux_to_bce=.4,bce_geodesic_aux_cosine=-.05,
                    weighted_ce_norm=.01,weighted_distance_norm=.1,ce_distance_cosine=.2)])
            with patch.multiple(creport,A0=a0,A1=a1):creport.report(root,manifest)
            report=(root/'GEODESIC_ALIGNMENT_SUMMARY_CN.md').read_text()
            self.assertIn('Macro-F1相对原A1降低',report)
            self.assertIn('不是仅替换内积的单变量消融',report)
            self.assertEqual(len(cc.csvrows(root/'PER_LABEL_COMPARISON.csv')),81)
            self.assertEqual(len(cc.csvrows(root/'PER_LABEL_GEODESIC_MINUS_COORDINATE.csv')),27)
            for name in cc.CONFIG['rare_labels']:self.assertIn('|'+name+'|',report)
            candidate=cc.csvrows(root/'VALIDATION_COMPARISON.csv')[-1]
            self.assertEqual(candidate['model'],'A1-G-C');self.assertEqual(float(candidate['auxiliary_coefficient']),GAMMA)


if __name__=='__main__':unittest.main(verbosity=2)
