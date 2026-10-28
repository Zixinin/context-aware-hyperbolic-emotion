"""Only substitute classification-loss aggregation on a new frozen-A1S-shaped model."""
import numpy as np
import torch
from torch.nn import functional as F
from ncommon import need
from scaled_model import build as control_build
from a1model import loss_components as frozen_aux_components

def loss_components(logits,targets,z,decoder_logits,embeddings,base=None):
    b=len(logits)
    need(b>0 and logits.shape==targets.shape==decoder_logits.shape==(b,27),'Need multi-hot [B,27]')
    need(z.shape==(b,100) and embeddings.shape==(27,100),'Wrong projection/label dimensions')
    need(all(t.dtype==torch.float32 and t.device==logits.device and bool(torch.isfinite(t).all()) for t in [logits,targets,z,decoder_logits,embeddings]),'Nonfinite/dtype/device error')
    need(not embeddings.requires_grad,'Label embeddings must remain fixed')
    need(bool(((targets==0)|(targets==1)).all() and (targets.sum(-1)>0).all()),'Invalid positive labels')
    # Exact audited c=1 geodesic; FP64 distances. No alpha/epsilon/clip/temperature.
    with torch.no_grad():
        zz=z.detach().double();ee=embeddings.detach().double()
        rz=zz.square().sum(-1);re=ee.square().sum(-1)
        need(bool((rz<1).all() and (re<1).all()),'Coordinates outside ball; stop')
        squared=(zz[:,None,:]-ee[None,:,:]).square().sum(-1)
        d=torch.acosh(1+2*squared/((1-rz[:,None])*(1-re[None,:])))
        counts=targets.sum(-1)
        positive_distance=(d*targets.double()).sum(-1)/counts.double()
        mean=positive_distance.mean()
        need(bool(torch.isfinite(d).all() and torch.isfinite(mean)) and float(mean)>0,'Invalid distance normalization; stop')
        weight=(positive_distance/mean).to(logits.dtype)
        need(not weight.requires_grad and bool(torch.isfinite(weight).all()) and abs(float(weight.mean())-1.)<2e-6,'Invalid detached weight')
    ordinary_each=F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(-1)
    ordinary=ordinary_each.mean();weighted=(weight*ordinary_each).mean()
    old=base if base is not None else frozen_aux_components(logits,targets,z,decoder_logits,embeddings)
    # Retain the exact frozen auxiliary Tensor and its graph/reductions.
    need(torch.allclose(old['classification'],ordinary,atol=1e-7,rtol=1e-6),'Ordinary BCE differs from frozen A1-S beyond reduction rounding')
    auxiliary=old['auxiliary'];total=weighted+.1*auxiliary
    need(bool(torch.isfinite(total)),'Nonfinite N1 objective')
    return dict(old,total=total,classification=weighted,ordinary_bce=ordinary,weighted_bce=weighted,
        ordinary_per_sample=ordinary_each,distance_matrix=d,positive_distance=positive_distance,
        normalized_weight=weight,positive_count=counts)

def install_n1(model):
    need(not getattr(model,'_exp6d1_n1_installed',False),'N1 already installed')
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs)
        labels=kwargs.get('labels')
        if labels is not None:
            c=loss_components(out['logits'],labels,out['z'],out['decoder_logits'],model.emotion_embeddings,base=out['components'])
            out=dict(out,components=c,loss=c['total'])
        return out
    model.forward=forward;model._exp6d1_n1_installed=True
    return model

def build(snapshot,names):
    model,tokenizer,loading,identity=control_build(snapshot,names)
    return install_n1(model),tokenizer,loading,identity

def values(out):
    c=out['components'];scalar=lambda v:float(v.detach())
    return dict(loss=scalar(c['total']),bce=scalar(c['ordinary_bce']),weighted_bce=scalar(c['weighted_bce']),
        auxiliary=scalar(c['auxiliary']),weighted_auxiliary=.1*scalar(c['auxiliary']),
        positive_ce_mean=scalar(c['positive_ce'].mean()),positive_dot_mean=scalar(c['positive_dot'].mean()),
        weighted_bce_to_bce=scalar(c['weighted_bce']/c['ordinary_bce']),max_ball_norm=scalar(out['z'].norm(dim=-1).max()))

def vectors(out):
    c=out['components']
    return {name:c[key].detach().cpu().numpy() for name,key in [('distance','positive_distance'),('weight','normalized_weight'),('bce','ordinary_per_sample'),('positive_count','positive_count')]}

def describe(v):
    d,w,b,n=[np.asarray(v[k],dtype=np.float64) for k in ['distance','weight','bce','positive_count']]
    need(all(np.isfinite(a).all() for a in [d,w,b,n]),'Nonfinite saved N1 statistics')
    result=dict(distance_mean=float(d.mean()),distance_std=float(d.std()),distance_median=float(np.median(d)),
        distance_p95=float(np.percentile(d,95)),distance_max=float(d.max()),weight_mean=float(w.mean()),
        weight_std=float(w.std()),weight_min=float(w.min()),weight_p95=float(np.percentile(w,95)),weight_max=float(w.max()),
        weighted_bce_to_bce=float((w*b).mean()/b.mean()),weight_vs_bce_correlation=float(np.corrcoef(w,b)[0,1]) if w.std()>0 and b.std()>0 else None)
    return result

def train_step(model,batch,optimizer,scheduler):
    ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x)
    result=values(out);vec=vectors(out);n=len(ids)
    # Preserve A1-S backward arithmetic, clipping, optimizer and scheduler order.
    (out['loss']*n).backward()
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(n)
    need(model.emotion_embeddings.grad is None,'Fixed label vectors received gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return dict(result,unclipped_grad_norm=float(norm),batch_size=n,ids=ids),vec
