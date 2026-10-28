"""Exp6D-1 provenance. Import frozen helpers read-only; never rewrite old seals."""
from pathlib import Path
import importlib.util
import os
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
S_SOURCE=HERE.parent/'exp6c1s_scaled_train'
sys.path.insert(0,str(S_SOURCE))
import scommon as sc
from scommon import read,write,sha,table,csvrows,need,md,now,state_hash,runtime,a0,extended_score,probabilities,save_validation
A0=sc.A0; A1=sc.A1
CONTROL=REPO/'results/exp6c1s_scaled_train/run_20261022T035554Z_cZrAxW97'
D=REPO/'results/exp6d_hypemo_code_audit/run_20261022T063911Z_Ktwq95al'
BASE=REPO/'results/exp6d1_normalized_distance_bce'
STATUS='EXP6D1_N1_SEED42_VALIDATION_PILOT_COMPLETE'
CONFIG=dict(sc.CONFIG,experiment='Exp6D-1 N1 seed42 Validation-only pilot',
    loss='mean_N1_weighted_BCE_plus_0.1_frozen_FormulaA_aux',classification_distance_weighting=True,
    distance_weight='stopgrad(mean_positive_geodesic/mean_current_minibatch_mean_positive_geodesic)',
    distance_precision='float64_then_normalized_weights_cast_float32_matching_Exp6D_audit',
    normalization='current_minibatch_including_short_final_batch',classification_coefficient=1.,
    matched_control=str(CONTROL.relative_to(REPO)),formal_test_inference_calls=0,
    comparison='Validation-only A1-S primary; A0/A1 historical context')

def run_path(path):
    p=Path(path).resolve();need(p.parent==BASE.resolve() and p.name.startswith('run_'),'Wrong isolated result root');return p

def check_sources():
    actual={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    need(actual==read(HERE/'PACKAGE_MANIFEST.json'),'Exp6D-1 source changed')
    expected=read(HERE/'INPUT_REFERENCE.json')
    need({p.name:sha(p) for p in S_SOURCE.iterdir() if p.is_file()}==expected['A1S_source_sha256'],'Frozen A1-S source changed')
    frozen=sc.source_hashes()
    for name,h in expected['Exp6D_source_sha256'].items():need(sha(D/'tools'/name)==h,'Frozen Exp6D source differs: '+name)
    return dict(n1=actual,frozen=frozen,Exp6D=expected['Exp6D_source_sha256'])

def check_old_seals():
    check_sources();sc.historical_audit()
    ref=read(HERE/'INPUT_REFERENCE.json')
    need(sha(CONTROL/'SUCCESS.json')==ref['A1S_success_sha256'],'Wrong A1-S identity')
    for tag,path in [('A0',A0),('A1',A1),('A1S',CONTROL),('Exp6D',D)]:
        proof=read(path/'SUCCESS.json')
        for name,h in proof['artifacts_sha256'].items():need(sha(path/name)==h,'Old sealed artifact changed: '+tag+'/'+name)
    need(read(D/'SUCCESS.json')['status']=='EXP6D_HYPEMO_VS_EXP6C_AUDIT_COMPLETE','Exp6D not complete')
    need(read(D/'AUDIT_STAGE.json')['recommendation']=='ONE_PROPOSED_PILOT_A1S_N1','Exp6D did not recommend this pilot')
    p=read(D/'NO_UPDATE_PROOF.json')
    need(p['frozen_before_equals_after'] and all(p['rng'].values()),'Exp6D preservation failed')
    need(read(CONTROL/'checkpoint_selection.json')['checkpoint_sha256']==ref['audited_A1S_checkpoint_sha256'],'Wrong replay checkpoint')

def audit_reference():
    spec=importlib.util.spec_from_file_location('exp6d_frozen_n1_reference',D/'tools/audit_math.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def protected_inventory():
    roots=[REPO/n for n in ['experiments','results','outputs','数据集','assets/pretrained','docs']]
    roots += [sc.a1lib.UPSTREAM]+[sc.a1lib.UPSTREAM.parent/n for n in ['outputs','reports']]
    roots += [Path('/home/zixinfeng/Documents/HypEmo-Exp6B')/n for n in ['stage_a','stage_b','reports']]
    roots += [Path(read(CONTROL/'plan.json')['snapshot'])]
    result={}
    for root in roots:
        if not root.exists():continue
        print('FREEZE_HASH_ROOT',str(root),flush=True)
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[n for n in dirs if n not in {'.git','__pycache__','.pytest_cache','runtime_cache'} and not n.startswith('.venv') and Path(directory)/n not in [HERE,BASE]]
            for n in files:
                p=Path(directory)/n
                if p.is_file() and not n.endswith(('.pyc','.DS_Store')):result[str(p)]=sha(p)
    return result

def audit_unchanged(root):
    need(protected_inventory()==read(root/'protected_before.json'),'Historical source/data/model/results changed')
    need(check_sources()==read(root/'source_manifest.json'),'Pilot or frozen source changed')
    check_old_seals()

def artifact_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and p.name not in {'SUCCESS.json','exit_code.txt'} and not (p.parent==root and (p.name.endswith('.log') or p.name.endswith('_exit_code.txt')))
        and not {'__pycache__','runtime_cache','review'} & set(p.relative_to(root).parts)}

def dataset():
    from data27 import load_dataset
    data,manifest=load_dataset()
    need(manifest==read(A0/'dataset_manifest.json')==read(A1/'dataset_manifest.json')==read(CONTROL/'dataset_manifest.json'),'Frozen canonical dataset differs')
    need([len(data[k]) for k in ['train','validation','test']]==[30587,3834,3821],'Wrong splits')
    del data['test']
    return data,manifest

def historical_validation(root,names):
    directory=root/'historical_validation';directory.mkdir(exist_ok=True)
    ids=gold=None
    for tag,path in [('A0',A0),('A1',A1),('A1S',CONTROL)]:
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['label_names'].tolist()==names,'Historical label order differs')
            metrics,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),names)
            recorded=read(path/'validation_metrics.json')
            need(all(metrics[k]==v for k,v in recorded.items()),'Historical metrics differ')
            need(np.array_equal(pred,z['predictions']),'Historical predictions differ')
            if ids is not None:need(np.array_equal(ids,z['ids']) and np.array_equal(gold,z['targets']),'Unpaired Validation samples')
            ids=z['ids'].copy();gold=z['targets'].copy()
        write(directory/(tag+'_metrics.json'),metrics);table(directory/(tag+'_per_label.csv'),rows)
    write(directory/'PROVENANCE.json',dict(split='validation',new_model_inference=False,
        prediction_arrays_sha256={tag:sha(path/'validation_predictions.npz') for tag,path in [('A0',A0),('A1',A1),('A1S',CONTROL)]},test_results_used=False))
