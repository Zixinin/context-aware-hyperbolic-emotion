"""Geodesic alignment only: exact c>0 metric, no epsilon/clamp/reprojection."""
import math
import torch
from torch.nn import functional as F

def need(ok,message):
    if not ok:raise RuntimeError(message)

def ball_distance(x,y,c=1.):
    """Broadcast pair distances. FP64 geometry, live autograd back to FP32 input.

    d_c=2/sqrt(c)*asinh(sqrt(c)*||x-y||/sqrt((1-c||x||²)(1-c||y||²))).
    Equivalent to acosh(1+2c||x-y||²/denom)/sqrt(c). vector_norm gives
    its zero subgradient at exact coincidence; we do not claim differentiability
    of the unsquared metric there. Coordinates outside the OPEN ball stop.
    """
    need(isinstance(c,(int,float)) and math.isfinite(c) and c>0,'Need fixed positive curvature magnitude')
    need(x.device==y.device and x.shape[-1]==y.shape[-1],'Geometry device/dimension mismatch')
    need(x.dtype in [torch.float32,torch.float64] and y.dtype==x.dtype,'Geometry requires matching FP32/FP64 inputs')
    a,b=x.double(),y.double()
    need(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'Nonfinite ball coordinates')
    da=1-c*a.square().sum(-1);db=1-c*b.square().sum(-1)
    need(bool((da>0).all() and (db>0).all()),'Coordinates outside open Poincare ball; no repair')
    v=math.sqrt(c)*torch.linalg.vector_norm(a-b,dim=-1)/torch.sqrt(da*db)
    d=2/math.sqrt(c)*torch.asinh(v)
    need(bool(torch.isfinite(d).all() and (d>=0).all()),'Nonfinite/negative distance')
    return d

def components(logits,targets,z,decoder_logits,labels,base=None,c=1.):
    # Use the actual frozen CE tensor and BCE tensor when called from the model.
    # Standalone CPU tests can supply their independently computed base.
    need(logits.ndim==2 and logits.shape==targets.shape==decoder_logits.shape and logits.shape[1]==27,'Need [B,27] emotion-only targets')
    need(len(logits)>0 and z.shape==(len(logits),100) and labels.shape==(27,100),'Wrong alignment dimensions')
    need(not labels.requires_grad and bool(((targets==0)|(targets==1)).all()) and bool((targets.sum(-1)>0).all()),'Empty/nonbinary positives or trainable label vectors')
    need(all(t.dtype==torch.float32 and t.device==logits.device and bool(torch.isfinite(t).all()) for t in [logits,targets,z,decoder_logits,labels]),'Loss dtype/device/nonfinite input')
    q=targets/targets.sum(-1,keepdim=True)
    if base is None:
        bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='mean')
        ce=-(q*F.log_softmax(decoder_logits,dim=-1)).sum(-1)
        dot=(q*(z@labels.T)).sum(-1)
        base=dict(classification=bce,positive_ce=ce,positive_dot=dot,auxiliary=(ce-dot).mean())
    distances=ball_distance(z[:,None,:],labels[None,:,:],c)
    # Explicit positive-label mean, then complete-batch mean. No sqrt(dim),
    # batch distance normalization, negative-label margin or distance weighting.
    positive_distance=(q*distances.to(torch.float32)).sum(-1)
    auxiliary_per_sample=base['positive_ce']+positive_distance
    auxiliary=auxiliary_per_sample.mean();total=base['classification']+.1*auxiliary
    need(bool(torch.isfinite(total)),'Nonfinite geodesic objective')
    return dict(base,total=total,classification=base['classification'],auxiliary=auxiliary,
        auxiliary_per_sample=auxiliary_per_sample,positive_distance=positive_distance,
        distance_matrix=distances,coordinate_auxiliary=base['auxiliary'],positive_count=targets.sum(-1))

def neutral_mask_demo(values,emotion_targets):
    """Audit-only 28-label masked positive reduction; never used in27 training."""
    count=emotion_targets.sum(-1);mask=count>0
    q=emotion_targets/count.clamp_min(1).unsqueeze(-1)
    return ((q*values).sum(-1)*mask.to(values.dtype)).mean(),mask
