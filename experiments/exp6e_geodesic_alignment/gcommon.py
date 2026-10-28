"""Frozen inputs, seed42 protocol and validation-only artifact utilities."""
from pathlib import Path
import os,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];BASE=REPO/'results/exp6e_geodesic_alignment'
sys.path.insert(0,str(HERE.parent/'exp6c1s_scaled_train'))
import scommon as sc
from scommon import read,write,sha,table,csvrows,need,md,now,state_hash,extended_score,probabilities,save_validation
from a1lib import runtime
from a1model import build as build_coordinate,parameter_groups
from model27 import seed_everything
from data27 import load_dataset,loader,Collator,targets
from probes import isolated_eval
from geometry_core import ScalingTrace
from gmodel import install
A0=sc.A0;A1=sc.A1;UPSTREAM=sc.a1lib.UPSTREAM
CONFIG=dict(sc.a1lib.CONFIG,experiment='Exp6E Geodesic Alignment Ablation A1-G',
    loss='mean_BCE_plus_0.1_soft_target_CE_and_positive_geodesic_alignment',
    auxiliary_formula='mean_B(soft_target_CE+mean_positive_unsquared_geodesic_c1)',
    alignment='positive geodesic distance instead of negative coordinate dot',hyperbolic_input_scale=1.,
    geodesic_arithmetic='FP64 exact asinh equivalent, FP32 distances in auxiliary reduction; FP32 model/optimizer',
    new_test_inference=False,formal_test_inference_calls=0,selection='Validation Macro-F1 at .30 earliest tie')
STATUS='EXP6E_A1G_SEED42_VALIDATION_PILOT_COMPLETE'
SEVERE_RATIO=10.

def build(snapshot,names,geodesic=True):
    model,tok,loading,identity=build_coordinate(snapshot,names)
    return (install(model) if geodesic else model),tok,loading,identity

def source_hashes():
    current={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    need(current==read(HERE/'PACKAGE_MANIFEST.json'),'New experiment source changed')
    for name,expected in read(HERE/'FROZEN_SOURCES.json').items():
        need({p.name:sha(p) for p in (HERE.parent/name).iterdir() if p.is_file()}==expected,'Frozen source changed: '+name)
    sc.source_hashes();sc.a1lib.verify_upstream()
    return current

def old_seals():
    for path in [A0,A1]:
        proof=read(path/'SUCCESS.json')
        for name,h in proof['artifacts_sha256'].items():need(sha(path/name)==h,'Old artifact changed: '+str(path/name))
    need(read(A0/'config.json')==sc.a0.CONFIG and read(A1/'config.json')==sc.a1lib.CONFIG,'Historical protocol changed')
    need(read(A1/'checkpoint_reload_verification.json')['metrics_equal'],'Original A1 lacks full reload proof')

def protected_inventory(snapshot):
    roots=[REPO/n for n in ['experiments','results','outputs','数据集','assets/pretrained','docs']]
    roots += [UPSTREAM,UPSTREAM.parent/'outputs',UPSTREAM.parent/'reports',Path('/home/zixinfeng/Documents/HypEmo-Exp6B'),Path(snapshot)]
    result={}
    for root in roots:
        if not root.exists():continue
        print('FREEZE_HASH_ROOT',root,flush=True)
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','runtime_cache','.pytest_cache'} and not d.startswith('.venv') and Path(directory)/d not in [HERE,BASE]]
            for name in files:
                p=Path(directory)/name
                if p.is_file() and not name.endswith(('.pyc','.DS_Store')):result[str(p)]=sha(p)
    return result

def freeze_check(root):
    plan=read(root/'PLAN.json');need(source_hashes()==plan['source_sha256'],'Plan/source mismatch')
    need(protected_inventory(plan['snapshot'])==read(root/'protected_before.json'),'Historical artifacts changed')
    old_seals()

def artifact_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p not in [root/'SUCCESS.json',root/'DIAGNOSTIC_SUCCESS.json']
        and not p.name.endswith(('.log','_exit_code.txt')) and p.name!='exit_code.txt'
        and not {'runtime_cache','__pycache__'} & set(p.relative_to(root).parts)}

def data27():
    data,manifest=load_dataset()
    need(manifest==read(A0/'dataset_manifest.json')==read(A1/'dataset_manifest.json'),'Canonical data differs')
    need([len(data[s]) for s in ['train','validation','test']]==[30587,3834,3821],'Wrong splits')
    del data['test'];return data,manifest

def diagnostic_values(out):
    c=out['components'];val=lambda x:float(x.detach())
    distance=val(c['positive_distance'].mean());ce=val(c['positive_ce'].mean());dot=val(c['positive_dot'].mean())
    bce=val(c['classification']);aux=val(c['auxiliary'])
    return dict(loss=val(out['loss']),bce=bce,auxiliary=aux,weighted_auxiliary=.1*aux,
        positive_ce_mean=ce,positive_dot_mean=dot,positive_distance_mean=distance,
        coordinate_auxiliary_same_forward=val(c['coordinate_auxiliary']),weighted_aux_to_BCE=.1*aux/bce,
        max_ball_norm=val(out['z'].norm(dim=-1).max()))

def run_path(path):
    root=Path(path).resolve();need(root.parent==BASE.resolve() and root.name.startswith('run_'),'Wrong isolated result root');return root

def release():
    import gc
    gc.collect();torch.cuda.empty_cache()
