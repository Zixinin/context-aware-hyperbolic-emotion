"""Meaningful CPU contracts; these are not Ubuntu/model training results."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from nmodel import loss_components,install_n1,describe
from a1model import loss_components as frozen
from nprobe import cpu_grad,pair

class N1Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(614)
        self.g=torch.randn(16,27,requires_grad=True);self.z=(torch.randn(16,100)*.015).requires_grad_()
        self.u=torch.randn(16,27,requires_grad=True);self.e=torch.randn(27,100);self.e=self.e/self.e.norm(dim=-1,keepdim=True)*.99
        self.y=torch.zeros(16,27)
        for i in range(16):self.y[i,[i%27,(i+5)%27] if i%2 else [i%27]]=1
    def call(self,b=16):return loss_components(self.g[:b],self.y[:b],self.z[:b],self.u[:b],self.e)
    def test_detached_weights_and_multilabel_BCE_derivative(self):
        c=self.call();self.assertFalse(c['normalized_weight'].requires_grad)
        grads=torch.autograd.grad(c['weighted_bce'],[self.g,self.z,self.u],allow_unused=True)
        self.assertIsNone(grads[1]);self.assertIsNone(grads[2])
        expected=c['normalized_weight'][:,None]*(self.g.sigmoid()-self.y)/(16*27)
        self.assertTrue(torch.allclose(grads[0],expected,atol=1e-8,rtol=1e-6))
        self.assertTrue(bool((grads[0][self.y==1]<0).all()))
    def test_all_positive_distances_and_B1_B11(self):
        for b in [1,11,16]:
            c=self.call(b);self.assertAlmostEqual(float(c['normalized_weight'].mean()),1.,places=6)
            for i in range(b):self.assertTrue(torch.allclose(c['positive_distance'][i],c['distance_matrix'][i,self.y[i]==1].mean(),atol=1e-12))
            if b==1:self.assertTrue(torch.equal(c['ordinary_bce'],c['weighted_bce']))
        self.assertFalse(torch.equal(self.call(11)['normalized_weight'],self.call(16)['normalized_weight'][:11]))
    def test_auxiliary_tensor_and_gradients_unchanged(self):
        base=frozen(self.g,self.y,self.z,self.u,self.e)
        c=loss_components(self.g,self.y,self.z,self.u,self.e,base=base)
        self.assertIs(c['auxiliary'],base['auxiliary'])
        a=torch.autograd.grad(.1*c['auxiliary'],[self.z,self.u,self.g],retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(.1*base['auxiliary'],[self.z,self.u,self.g],allow_unused=True)
        self.assertIsNone(a[2]);self.assertIsNone(b[2])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[:2],b[:2])))
    def test_label_permutation_equivariance(self):
        p=torch.randperm(27);a=self.call()
        b=loss_components(self.g[:,p],self.y[:,p],self.z,self.u[:,p],self.e[p])
        self.assertTrue(torch.allclose(a['total'],b['total'],atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(a['positive_distance'],b['positive_distance'],atol=1e-12,rtol=1e-12))
    def test_invalid_inputs_stop_without_repair(self):
        for kind in ['empty','nonbinary','outside','nonfinite']:
            y=self.y.clone();e=self.e.clone();g=self.g.clone()
            if kind=='empty':y[0]=0
            if kind=='nonbinary':y[0,0]=.5
            if kind=='outside':e[0]=torch.ones(100)
            if kind=='nonfinite':g[0,0]=float('nan')
            with self.assertRaises(RuntimeError):loss_components(g,y,self.z,self.u,e)
    def test_combined_module_norm_and_none_grad_paths(self):
        h=nn.Linear(5,100);head=nn.Linear(100,27);proj=nn.Linear(100,100);dec=nn.Linear(100,27)
        v=h(torch.randn(16,5));z=.01*proj(v).tanh();g=head(v);u=dec(z)
        c=loss_components(g,self.y,z,u,self.e);params=list(h.parameters())+list(head.parameters())+list(proj.parameters())+list(dec.parameters())
        n=cpu_grad(c['weighted_bce'],params);a=cpu_grad(.1*c['auxiliary'],params);total=cpu_grad(c['total'],params,retain=False)
        self.assertTrue(all(x is None for x in n[4:]))
        self.assertTrue(all(x is None for x in a[2:4]))
        combined=pair(n,a,range(len(params)))['sum_norm'];actual=pair(total,total,range(len(params)))['a_norm']
        self.assertAlmostEqual(combined,actual,places=6);self.assertTrue(all(p.grad is None for p in params))
    def test_epoch_statistics_are_from_samples_not_batch_quantile_average(self):
        v={'distance':np.array([1.,2.,40.]),'weight':np.array([.5,1.,1.5]),'bce':np.array([.1,.2,.9]),'positive_count':np.array([1,2,1])}
        s=describe(v);self.assertAlmostEqual(s['distance_p95'],np.percentile(v['distance'],95))
        self.assertAlmostEqual(s['weighted_bce_to_bce'],np.mean(v['weight']*v['bce'])/np.mean(v['bce']))
    def test_install_preserves_parameters_forward_outputs_and_rng(self):
        class Mini(nn.Module):
            def __init__(inner,e):
                super().__init__();inner.h=nn.Linear(5,100);inner.head=nn.Linear(100,27);inner.dec=nn.Linear(100,27);inner.drop=nn.Dropout(.1);inner.register_buffer('emotion_embeddings',e)
            def forward(inner,input_ids,attention_mask,labels=None):
                h=inner.drop(inner.h(input_ids));z=.01*h.tanh();g=inner.head(h);u=inner.dec(z)
                c=None if labels is None else frozen(g,labels,z,u,inner.emotion_embeddings)
                return dict(logits=g,z=z,decoder_logits=u,components=c,loss=None if c is None else c['total'])
        model=Mini(self.e);x=torch.randn(16,5);mask=torch.ones(16,5);state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state()
        before=model(input_ids=x,attention_mask=mask,labels=self.y);after_rng=torch.get_rng_state()
        torch.set_rng_state(rng);install_n1(model);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        after=model(input_ids=x,attention_mask=mask,labels=self.y)
        self.assertTrue(torch.equal(torch.get_rng_state(),after_rng))
        for k in ['logits','z','decoder_logits']:self.assertTrue(torch.equal(before[k],after[k]))
        self.assertTrue(torch.equal(before['components']['auxiliary'],after['components']['auxiliary']))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        self.assertEqual(set(state),set(model.state_dict()))
        self.assertIsNone(model(input_ids=x,attention_mask=mask)['components'])
        with self.assertRaises(RuntimeError):install_n1(model)

def cache_reference_checks(repo):
    """Actual saved frozen train outputs, independent of new model implementation."""
    D=repo/'results/exp6d_hypemo_code_audit/run_20261022_prepare_v1'
    spec=importlib.util.spec_from_file_location('local_completed_audit_formula',D/'tools/audit_math.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    source=D/'evidence/A1S';selection=__import__('json').loads((source/'FIXED_TRAIN_COHORTS.json').read_text())
    e=torch.from_numpy(np.load(source/'fixed_emotion_embeddings.npy',allow_pickle=False));checked=0
    for tag in ['initial','best_reload']:
        for cohort in selection['cohorts']:
            with np.load(source/'geometry'/tag/(cohort+'.npz'),allow_pickle=False) as z:
                args=[torch.from_numpy(z[k]).float() for k in ['logits','targets','z','decoder_logits']]+[e]
                current=loss_components(*args);old=ref.candidates(*args)
                for key,kref in [('positive_distance','dbar'),('normalized_weight','N_weight'),('weighted_bce','N_classification'),('total','N')]:
                    if not torch.allclose(current[key],old[kref],atol=1e-7,rtol=1e-6):raise AssertionError('Audit cache mismatch '+cohort+'/'+key)
                checked+=len(z['ids'])
    return dict(saved_train_cache_positions_checked=checked,cache_models='A1S fresh initialization and trained best',
        independent_Exp6D_candidate_formula_match=True,new_GPU_model_training=False)

if __name__=='__main__':unittest.main()
