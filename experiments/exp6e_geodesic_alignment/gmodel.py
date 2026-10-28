"""One instance-level loss replacement on a fresh UNSCALED frozen A1 model."""
from gmath import components,need

def install(model):
    need(not getattr(model,'_geodesic_alignment_installed',False),'Refuse repeated installation')
    need(not getattr(model.hyper,'_exp6c1s_scale_installed',False),'This ablation is original A1, scale1, not A1-S')
    c=float(model.hyper.c.detach().cpu());need(c==1.,'Frozen A1 curvature differs')
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs);target=kwargs.get('labels')
        if target is not None:
            old=out['components']
            new=components(out['logits'],target,out['z'],out['decoder_logits'],model.emotion_embeddings,base=old,c=c)
            out=dict(out,components=new,loss=new['total'])
        return out
    model.forward=forward;model._geodesic_alignment_installed=True
    return model
