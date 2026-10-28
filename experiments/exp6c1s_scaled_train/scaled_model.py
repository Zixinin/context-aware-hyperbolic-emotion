"""The only model intervention: scale once at the frozen A1 auxiliary entry."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'exp6c1_roberta27_hypaux'))
from a1model import build as unscaled_build

SCALE=.03

def install_scale(model):
    if getattr(model.hyper,'_exp6c1s_scale_installed',False):
        raise RuntimeError('Refuse repeated auxiliary input scaling')
    original=model.hyper.encode
    def scaled(h,*args,**kwargs):
        return original(SCALE*h,*args,**kwargs)
    # No new module, tensor, parameter, buffer, random draw or state-dict key.
    model.hyper.encode=scaled
    model.hyper._exp6c1s_scale_installed=True
    return model

def build(snapshot,names):
    model,tokenizer,loading,identity=unscaled_build(snapshot,names)
    return install_scale(model),tokenizer,loading,identity
