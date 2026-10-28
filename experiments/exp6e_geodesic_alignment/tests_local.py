"""CPU math/gradient/data/schema contracts, never a GPU performance result."""
from pathlib import Path
import hashlib,math,unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gmath import ball_distance,components,neutral_mask_demo
from gmodel import install

REPO=Path(__file__).resolve().parents[2]

def reference(x,y,c):
    a,b=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    return np.arccosh(1+2*c*np.square(a-b).sum(-1)/((1-c*np.square(a).sum(-1))*(1-c*np.square(b).sum(-1))))/math.sqrt(c)

class Geometry(unittest.TestCase):
    def test_metric_against_independent_numpy_acosh_and_curvatures(self):
        rng=np.random.default_rng(17)
        for c in [.2,1.,2.]:
            x=torch.tensor(rng.normal(size=(8,7))*.04,dtype=torch.float64);y=torch.tensor(rng.normal(size=(8,7))*.06,dtype=torch.float64)
            d=ball_distance(x,y,c);self.assertTrue(np.allclose(d.numpy(),reference(x,y,c),atol=1e-13,rtol=1e-12))
            self.assertTrue(bool((d>=0).all()));self.assertTrue(torch.equal(d,ball_distance(y,x,c)))

    def test_identical_points_exact_zero_and_finite_zero_subgradient(self):
        for dtype in [torch.float32,torch.float64]:
            x=torch.tensor([[.1,.2],[.99,0.],[0.,0.]],dtype=dtype,requires_grad=True)
            d=ball_distance(x,x.detach());self.assertTrue(torch.equal(d,torch.zeros_like(d)))
            g=torch.autograd.grad(d.sum(),x)[0];self.assertTrue(torch.isfinite(g).all());self.assertTrue(torch.equal(g,torch.zeros_like(g)))

    def test_near_coincident_and_near_boundary_backward(self):
        for x,y in [([.2,.3],[.2+1e-12,.3]),([1-1e-10,0.],[0.,1-1e-10]),([.995999,0.],[.998914,0.])]:
            a=torch.tensor([x],dtype=torch.float64,requires_grad=True);b=torch.tensor([y],dtype=torch.float64,requires_grad=True)
            d=ball_distance(a,b);g=torch.autograd.grad(d.sum(),[a,b]);self.assertTrue(d.isfinite().all());self.assertTrue(all(v.isfinite().all() for v in g))

    def test_gradcheck_noncoincident_interior_and_near_boundary(self):
        for x,y in [([.2,.1],[-.3,.2]),([.995,.001],[.01,.998])]:
            a=torch.tensor([x],dtype=torch.float64,requires_grad=True);b=torch.tensor([y],dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(ball_distance,(a,b),eps=1e-7,atol=1e-5,rtol=1e-4))

    def test_invalid_points_curvature_and_nonfinite_rejected(self):
        for value in [1.,1.01,float('nan'),float('inf')]:
            with self.assertRaises(RuntimeError):ball_distance(torch.tensor([[value,0.]]),torch.zeros(1,2))
        for c in [0.,-1.,float('nan')]:
            with self.assertRaises(RuntimeError):ball_distance(torch.zeros(1,2),torch.zeros(1,2),c)

    def test_triangle_and_curvature_scaling(self):
        torch.manual_seed(6);x=torch.randn(20,4,dtype=torch.float64)*.1;y=torch.randn_like(x)*.2;z=torch.randn_like(x)*.1
        self.assertTrue(bool((ball_distance(x,z)<=ball_distance(x,y)+ball_distance(y,z)+1e-12).all()))
        c=2.;self.assertTrue(torch.allclose(ball_distance(x,y,c),ball_distance(x*math.sqrt(c),y*math.sqrt(c))/math.sqrt(c),atol=1e-12,rtol=1e-12))

class Auxiliary(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);self.g=torch.randn(16,27,requires_grad=True);self.u=torch.randn(16,27,requires_grad=True)
        self.z=(torch.randn(16,100)*.02).requires_grad_();e=torch.randn(27,100);self.e=e/e.norm(dim=-1,keepdim=True)*.995
        self.y=torch.zeros(16,27)
        for i in range(16):self.y[i,[i,(i+5)%26] if i%2 else [i]]=1
    def call(self,b=16,base=None):return components(self.g[:b],self.y[:b],self.z[:b],self.u[:b],self.e,base=base)

    def test_all_positive_means_then_full_batch_mean_B1_B11_B16(self):
        for b in [1,11,16]:
            c=self.call(b);manual=[]
            for i in range(b):manual.append(c['distance_matrix'][i,self.y[i]==1].to(torch.float32).mean())
            expected=torch.stack(manual)
            self.assertTrue(torch.allclose(c['positive_distance'],expected,atol=1e-6,rtol=1e-6))
            self.assertTrue(torch.equal(c['auxiliary'],(c['positive_ce']+c['positive_distance']).mean()))
            self.assertTrue(torch.equal(c['total'],c['classification']+.1*c['auxiliary']))

    def test_exact_same_CE_and_BCE_tensors_from_frozen_base(self):
        import sys
        sys.path.insert(0,str(REPO/'experiments/exp6c1_roberta27_hypaux'))
        from a1model import loss_components as frozen
        base=frozen(self.g,self.y,self.z,self.u,self.e);c=self.call(base=base)
        self.assertIs(c['positive_ce'],base['positive_ce']);self.assertIs(c['classification'],base['classification'])
        self.assertIs(c['coordinate_auxiliary'],base['auxiliary'])
        self.assertTrue(torch.equal(c['positive_count'],self.y.sum(-1)))

    def test_classifier_BCE_unchanged_decoder_CE_target_and_z_gradient_live(self):
        c=self.call();gg,gu,gz=torch.autograd.grad(c['total'],[self.g,self.u,self.z])
        self.assertTrue(torch.allclose(gg,(self.g.sigmoid()-self.y)/(16*27),atol=1e-8,rtol=1e-6))
        q=self.y/self.y.sum(-1,keepdim=True)
        self.assertTrue(torch.allclose(gu,.1*(self.u.softmax(-1)-q)/16,atol=1e-8,rtol=1e-6))
        self.assertTrue(gz.isfinite().all());self.assertGreater(float(gz.norm()),0.)

    def test_aux_has_no_direct_classifier_path_and_alignment_has_no_decoder_path(self):
        c=self.call();ga=torch.autograd.grad(c['auxiliary'],[self.g,self.z,self.u],allow_unused=True,retain_graph=True)
        self.assertIsNone(ga[0]);self.assertIsNotNone(ga[1]);self.assertIsNotNone(ga[2])
        gd=torch.autograd.grad(c['positive_distance'].sum(),[self.g,self.z,self.u],allow_unused=True)
        self.assertIsNone(gd[0]);self.assertIsNotNone(gd[1]);self.assertIsNone(gd[2])

    def test_negative_label_vector_not_used_and_positive_is_used(self):
        base=self.call();e=self.e.clone();e[26]=torch.zeros(100)
        no_negative=components(self.g,self.y,self.z,self.u,e)
        self.assertTrue(torch.equal(base['auxiliary'],no_negative['auxiliary']))
        e=self.e.clone();e[0]=torch.zeros(100);positive=components(self.g,self.y,self.z,self.u,e)
        self.assertNotEqual(float(base['positive_distance'][0]),float(positive['positive_distance'][0]))

    def test_label_permutation_equivariance(self):
        p=torch.randperm(27);a=self.call();b=components(self.g[:,p],self.y[:,p],self.z,self.u[:,p],self.e[p])
        self.assertTrue(torch.allclose(a['total'],b['total'],atol=1e-6,rtol=1e-6))

    def test_empty_nonbinary_and_trainable_labels_stop(self):
        for kind in ['empty','nonbinary','trainable']:
            y=self.y.clone();e=self.e.clone()
            if kind=='empty':y[0]=0
            if kind=='nonbinary':y[0,0]=.5
            if kind=='trainable':e.requires_grad_()
            with self.assertRaises(RuntimeError):components(self.g,y,self.z,self.u,e)

    def test_neutral_full_batch_dilution_and_mixed_neutral_emotion_demo(self):
        values=torch.tensor([[2.,4.],[8.,9.]])
        active=torch.tensor([[1.,1.],[0.,0.]])
        loss,mask=neutral_mask_demo(values,active);self.assertEqual(float(loss),1.5);self.assertEqual(mask.tolist(),[True,False])
        alone,_=neutral_mask_demo(values[:1],active[:1]);self.assertEqual(float(alone),3.)
        # This is an audit demonstration, not a28-label model or a neutral vector.

    def test_four_module_gradient_sum_and_probe_extractor(self):
        from gprobe import cpu_grad,pair
        h=nn.Linear(5,100);head=nn.Linear(100,27);projection=nn.Linear(100,100);decoder=nn.Linear(100,27)
        v=h(torch.randn(16,5));z=.01*projection(v).tanh();g=head(v);u=decoder(z)
        c=components(g,self.y,z,u,self.e);ps=list(h.parameters())+list(head.parameters())+list(projection.parameters())+list(decoder.parameters())
        gb=cpu_grad(c['classification'],ps);ga=cpu_grad(.1*c['auxiliary'],ps);gt=cpu_grad(c['total'],ps,retain=False)
        self.assertTrue(all(x is None for x in gb[4:]));self.assertTrue(all(x is None for x in ga[2:4]))
        self.assertTrue(all(x is not None for x in ga[:2]+ga[4:]))
        for indices in [range(0,2),range(2,4),range(4,6),range(6,8)]:
            self.assertAlmostEqual(pair(gb,ga,indices)['sum_norm'],pair(gt,gt,indices)['a_norm'],places=6)
        self.assertTrue(all(p.grad is None for p in ps))

    def test_install_changes_no_state_or_RNG_and_preserves_forward_tensors(self):
        class Mini(nn.Module):
            def __init__(m,e):
                super().__init__();m.h=nn.Linear(5,100);m.head=nn.Linear(100,27);m.dec=nn.Linear(100,27)
                m.drop=nn.Dropout(.1);m.hyper=nn.Module();m.hyper.c=torch.tensor([1.]);m.register_buffer('emotion_embeddings',e)
            def forward(m,input_ids,attention_mask,labels=None):
                h=m.drop(m.h(input_ids));z=.01*h.tanh();g=m.head(h);u=m.dec(z)
                q=None if labels is None else labels/labels.sum(-1,keepdim=True)
                base=None if q is None else dict(classification=F.binary_cross_entropy_with_logits(g,labels),positive_ce=-(q*F.log_softmax(u,dim=-1)).sum(-1),positive_dot=(q*(z@m.emotion_embeddings.T)).sum(-1))
                if base is not None:base['auxiliary']=(base['positive_ce']-base['positive_dot']).mean();base['total']=base['classification']+.1*base['auxiliary']
                return dict(logits=g,z=z,decoder_logits=u,components=base,loss=None if base is None else base['total'])
        m=Mini(self.e);x=torch.randn(16,5);before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state()
        a=m(input_ids=x,attention_mask=None,labels=self.y);after_rng=torch.get_rng_state()
        torch.set_rng_state(rng);install(m);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        b=m(input_ids=x,attention_mask=None,labels=self.y);self.assertTrue(torch.equal(after_rng,torch.get_rng_state()))
        for k in ['logits','z','decoder_logits']:self.assertTrue(torch.equal(a[k],b[k]))
        self.assertEqual(set(before),set(m.state_dict()));self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(torch.equal(a['components']['positive_ce'],b['components']['positive_ce']))
        self.assertIsNone(m(input_ids=x,attention_mask=None)['loss'])
        with self.assertRaises(RuntimeError):install(m)

class RealData(unittest.TestCase):
    def test_actual_canonical_neutral_and_multilabel_counts(self):
        import sys
        sys.path.insert(0,str(REPO/'experiments/exp6c0_roberta27_baseline'))
        from data27 import load_dataset
        data,m=load_dataset(REPO)
        self.assertEqual([len(data[k]) for k in ['train','validation','test']],[30587,3834,3821])
        self.assertEqual([m['splits'][k]['removed_neutral_only'] for k in data],[12823,1592,1606])
        self.assertEqual([m['splits'][k]['retained_neutral_plus_emotion'] for k in data],[1396,174,181])
        self.assertEqual([m['splits'][k]['multi_emotion'] for k in data],[5767,712,667])

    def test_frozen_source_hashes_unchanged(self):
        import json
        for name,files in json.loads((Path(__file__).parent/'FROZEN_SOURCES.json').read_text()).items():
            self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'experiments'/name).iterdir() if p.is_file()},files)

    def test_archived_A1_scale1_train_cache_geometry(self):
        # Actual old A1 best train-only outputs, no new text encoder inference.
        base=REPO/'results/exp6c1s_input_scaling_diagnostics/run_20261021T133415Z_3VyOQEzH'
        if not base.exists():self.skipTest('Historical cache unavailable locally')
        e=torch.from_numpy(np.load(base/'fixed_emotion_embeddings.npy',allow_pickle=False))
        for p in sorted((base/'cache').glob('*_s1p0.npz')):
            with np.load(p,allow_pickle=False) as z:
                x=torch.from_numpy(z['z']);d=ball_distance(x[:,None,:],e[None,:,:])
                self.assertTrue(np.allclose(d.numpy(),reference(x[:,None,:],e[None,:,:],1.),atol=1e-10,rtol=1e-10))
                zz=x.clone().requires_grad_();u=torch.from_numpy(z['decoder_logits']);g=torch.from_numpy(z['classification_logits']);y=torch.from_numpy(z['targets'])
                c=components(g,y,zz,u,e);grad=torch.autograd.grad(c['auxiliary'],zz)[0];self.assertTrue(grad.isfinite().all());self.assertGreater(float(grad.norm()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
