"""Original coordinate A1: change ONLY the whole auxiliary coefficient."""
GAMMA=0.02091538317504222
GAMMA_HEX='0x1.56ad79ae165d2p-6'
assert GAMMA.hex()==GAMMA_HEX


def install_weight(model):
    if getattr(model,'_geodesic_alignment_installed',False):raise RuntimeError('Coordinate control cannot wrap a geodesic model')
    if getattr(model,'_matched_coordinate_installed',False):raise RuntimeError('Repeated coordinate coefficient installation')
    if getattr(model.hyper,'_exp6c1s_scale_installed',False):raise RuntimeError('This control requires original s=1')
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs)
        if kwargs.get('labels') is not None:
            c=dict(out['components'])
            c['coordinate_auxiliary']=c['auxiliary']
            c['total']=c['classification']+GAMMA*c['auxiliary']
            out=dict(out,components=c,loss=c['total'])
        return out
    model.forward=forward;model._matched_coordinate_installed=True
    return model
