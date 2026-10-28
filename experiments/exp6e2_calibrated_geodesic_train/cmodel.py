"""New instance only: frozen geodesic objective with the calibrated coefficient."""
import math

GAMMA = 0.02091538317504222
GAMMA_HEX = '0x1.56ad79ae165d2p-6'
assert GAMMA.hex() == GAMMA_HEX


def install_weight(model):
    if not getattr(model,'_geodesic_alignment_installed',False):
        raise RuntimeError('Original geodesic loss must already be installed')
    if getattr(model,'_calibrated_geodesic_weight_installed',False):
        raise RuntimeError('Refuse repeated installation')
    if getattr(model.hyper,'_exp6c1s_scale_installed',False):
        raise RuntimeError('This experiment requires original s=1')
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs)
        if kwargs.get('labels') is not None:
            c=dict(out['components'])
            c['total']=c['classification']+GAMMA*c['auxiliary']
            out=dict(out,components=c,loss=c['total'])
        return out
    model.forward=forward
    model._calibrated_geodesic_weight_installed=True
    return model
