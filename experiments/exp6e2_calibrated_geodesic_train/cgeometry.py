"""Read-only counts of actual text projection clipping in EACH training batch."""
import inspect
import torch


class ProjectionMonitor:
    def __init__(self,model,n):
        if n<2:raise RuntimeError('Authorized actual batches have n>=2; refuse ambiguous bias batch')
        self.model,self.n=model,n
        self.saved=[];self.first=None;self.later=None;self.calls=0

    def __enter__(self):
        seen=set()
        try:
            for _,module in self.model.hyper.named_modules():
                manifold=getattr(module,'manifold',None)
                if manifold is None or id(manifold) in seen or not hasattr(manifold,'proj'):continue
                seen.add(id(manifold));original=manifold.proj
                self.saved.append((manifold,'proj' in vars(manifold),vars(manifold).get('proj')))
                manifold.proj=self.wrapper(manifold,original)
        except BaseException:
            self.restore();raise
        return self

    def wrapper(self,manifold,original):
        def observe(x,c,*args,**kwargs):
            frame=inspect.currentframe().f_back
            is_initial=(frame.f_locals.get('self') is self.model.hyper.encoder and frame.f_code.co_name=='encode')
            del frame
            y=original(x,c,*args,**kwargs)
            if x.ndim==2 and len(x)==self.n:
                with torch.no_grad():
                    cap=(1-manifold.eps[x.dtype])/(c**.5)
                    clipped=x.detach().norm(dim=-1).clamp_min(manifold.min_norm)>cap
                    if is_initial:
                        if self.first is not None:raise RuntimeError('Repeated first text projection')
                        self.first=clipped
                        self.later=torch.zeros_like(clipped)
                    else:
                        if self.first is None:raise RuntimeError('Later text projection before initial mapping')
                        self.later=torch.logical_or(self.later,clipped)
                    self.calls+=1
            return y
        return observe

    def values(self,z):
        if self.first is None or self.later is None or self.calls<2:raise RuntimeError('Missing original text projection observations')
        r=z.detach().norm(dim=-1)
        return dict(first_text_clip_count=int(self.first.sum()),later_text_clip_count=int(self.later.sum()),
            final_radius_above099_count=int((r>.99).sum()),final_radius_mean=float(r.double().mean()),
            final_radius_min=float(r.min()),final_radius_max=float(r.max()),text_projection_calls=self.calls)

    def restore(self):
        for manifold,existed,previous in reversed(self.saved):
            if existed:manifold.proj=previous
            else:delattr(manifold,'proj')
        self.saved=[]

    def __exit__(self,*args):
        self.restore();return False
