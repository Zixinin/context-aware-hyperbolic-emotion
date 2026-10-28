"""Temporary instrumentation of a NEW model object; numerical operations unchanged."""
from contextlib import AbstractContextManager
import inspect
from pathlib import Path
import math
import numpy as np

SCALES=(1., .3, .1, .03, .01)

def need(ok,message):
    if not ok: raise RuntimeError(message)

def summary(a):
    a=np.asarray(a,dtype=np.float64)
    need(a.size>0 and np.isfinite(a).all(),'Empty/nonfinite measurement')
    return dict(n=int(a.size),minimum=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),maximum=float(a.max()),mean=float(a.mean()),
                population_std=float(a.std()),IQR=float(np.quantile(a,.75)-np.quantile(a,.25)))

class ScalingTrace(AbstractContextManager):
    """Scale ONLY raw h passed to hyper.encode; observe original manifold calls.

    HNN uses encode()/layers.forward(), bypassing nn.Module __call__ hooks.
    Wrapping manifold methods observes those exact calls without rewriting HNN.
    Every injected instance attribute is removed/restored even on exceptions.
    """
    OPS=('expmap0','proj','mobius_matvec','mobius_add','logmap0')

    def __init__(self,model,scale,ids,cohort):
        need(scale in SCALES,'Only preregistered scales allowed')
        self.model,self.scale,self.ids,self.cohort=model,float(scale),list(ids),cohort
        self.saved=[];self.events=[];self.rows=[];self.counters={}
        self.raw_h=None;self.initial_projected=None
        self.module_names={id(m):n or 'hyper' for n,m in model.hyper.named_modules()}

    def inject(self,obj,name,value):
        self.saved.append((obj,name,name in vars(obj),vars(obj).get(name)))
        setattr(obj,name,value)

    def restore(self):
        for obj,name,existed,previous in reversed(self.saved):
            if existed: setattr(obj,name,previous)
            else: delattr(obj,name)
        self.saved=[]

    def __enter__(self):
        original=self.model.hyper.encode
        def scaled(h,*args,**kwargs):
            self.raw_h=h
            # s=1 is the exact identity, preserving even the original strides.
            return original(h if self.scale==1. else h*self.scale,*args,**kwargs)
        try:
            self.inject(self.model.hyper,'encode',scaled)
            seen=set()
            for _,module in self.model.hyper.named_modules():
                manifold=getattr(module,'manifold',None)
                if manifold is None or id(manifold) in seen: continue
                seen.add(id(manifold))
                for op in self.OPS:
                    if hasattr(manifold,op): self.inject(manifold,op,self.wrapper(manifold,op))
        except BaseException:
            self.restore();raise
        return self

    def wrapper(self,manifold,op):
        original=getattr(manifold,op)
        def observed(*args,**kwargs):
            frame=inspect.currentframe().f_back
            owner=frame.f_locals.get('self')
            module=self.module_names.get(id(owner),type(owner).__name__)
            caller=frame.f_code.co_name
            location=Path(frame.f_code.co_filename).name+':'+str(frame.f_lineno)
            del frame
            key=module+'.'+caller+'.'+op
            self.counters[key]=self.counters.get(key,0)+1
            stage=key+'#'+str(self.counters[key])
            point=args[1] if op=='mobius_matvec' else args[0]
            c_index=2 if op in ['mobius_matvec','mobius_add'] else 1
            c=args[c_index] if len(args)>c_index else kwargs['c']
            # Original callable receives the original Tensor arguments. Recording
            # uses detached copies only and never replaces the output Tensor.
            result=original(*args,**kwargs)
            self.record(stage,module,op,location,point,result,c,manifold)
            if module=='encoder' and caller=='encode' and op=='proj':
                need(self.initial_projected is None,'Multiple unexplained initial mappings')
                self.initial_projected=result
            return result
        return observed

    def record(self,stage,module,op,location,x,y,c,manifold):
        import torch
        need(isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and x.ndim==y.ndim==2,'Unexpected manifold tensor shape')
        need(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'Nonfinite manifold stage')
        before=x.detach().norm(dim=-1).cpu().numpy()
        after=y.detach().norm(dim=-1).cpu().numpy()
        # Match the actual original scalar arithmetic/dtype, including the
        # FP32 rounding of its cap. Compare tensors as the original proj does.
        cap_tensor=(1-manifold.eps[x.dtype])/(c**.5)
        cap=float(cap_tensor.detach().cpu()) if isinstance(cap_tensor,torch.Tensor) else float(cap_tensor)
        clipped=(x.detach().norm(dim=-1).clamp_min(manifold.min_norm)>cap_tensor).cpu().numpy() if op=='proj' else None
        role='text' if len(x)==len(self.ids) else 'bias' if len(x)==1 else 'other'
        need(role!='other','Unrecognized manifold input batch')
        ids=self.ids if role=='text' else ['__fixed_bias__']
        self.events.append(dict(stage=stage,module=module,operation=op,source_location=location,role=role,
            input_dim=x.shape[1],output_dim=y.shape[1],output_space='tangent' if op=='logmap0' else 'ball',cap=cap))
        for i,record_id in enumerate(ids):
            self.rows.append(dict(cohort=self.cohort,scale=self.scale,id=record_id,stage=stage,module=module,
                operation=op,role=role,source_location=location,input_dim=x.shape[1],output_dim=y.shape[1],
                norm_before=float(before[i]),norm_after=float(after[i]),cap=cap,
                actual_proj_clipped=bool(clipped[i]) if op=='proj' else None,
                output_near_boundary=bool(after[i]>.99) if op!='logmap0' else None))

    def __exit__(self,exc_type,exc,traceback):
        self.restore()
        return False

def radial_decomposition(h,g):
    """Read gradient relative to raw h direction, with float64 reductions."""
    v=h.detach().double();grad=g.detach().double()
    norm=v.norm(dim=-1).clamp_min(1e-15);unit=v/norm[:,None]
    radial=(grad*unit).sum(-1)
    total=grad.norm(dim=-1)
    tangential=(total.square()-radial.square()).clamp_min(0).sqrt()
    return dict(h_norm=norm.cpu().numpy(),radial=radial.cpu().numpy(),gradient_norm=total.cpu().numpy(),
                tangential_norm=tangential.cpu().numpy())

def scale_summary(norms):
    """Unique IDs: first occurrence in declared cohort order, no biased duplicates."""
    result=[]
    for scale in SCALES:
        chosen={}
        for row in norms:
            if row['scale']==scale: chosen.setdefault(row['id'],row)
        rows=list(chosen.values());need(rows,'No scale rows')
        first=summary([r['initial_radius'] for r in rows]);final=summary([r['final_radius'] for r in rows])
        result.append(dict(scale=scale,unique_IDs=len(rows),initial_clip_fraction=float(np.mean([r['initial_clipped'] for r in rows])),
            downstream_any_clip_fraction=float(np.mean([r['downstream_any_clipped'] for r in rows])),
            final_near_boundary_fraction=float(np.mean([r['final_radius']>.99 for r in rows])),
            initial_radius_mean=first['mean'],initial_radius_std=first['population_std'],initial_radius_IQR=first['IQR'],
            final_radius_mean=final['mean'],final_radius_std=final['population_std'],final_radius_IQR=final['IQR'],
            initial_radius_min=first['minimum'],initial_radius_max=first['maximum'],
            final_radius_min=final['minimum'],final_radius_max=final['maximum'],
            initial_radial_derivative_abs_median=float(np.median([abs(r['initial_radial_derivative']) for r in rows]))))
    return result
