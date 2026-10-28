"""Independent pilot provenance and validation-only numerical reporting."""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,sys
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'exp6c1_roberta27_hypaux'))
import a1lib
from a1lib import a0,state_hash,runtime
from metrics27 import score,probabilities

BASE=REPO/'results/exp6c1s_scaled_train'
A0=a1lib.A0_RUN
A1=REPO/'results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL'
D=REPO/'results/exp6c1d_hyperbolic_diagnostics/run_20261021T101516Z_cLRnltwT'
S=REPO/'results/exp6c1s_input_scaling_diagnostics/run_20261021T133415Z_3VyOQEzH'
CONFIG=dict(a1lib.CONFIG,experiment='Exp6C-1S-Train seed42 validation-only pilot',
            hyperbolic_input_scale=.03,new_test_inference=False,pilot=True)
STATUS='EXP6C1S_SCALED_SEED42_PILOT_COMPLETE'
def need(ok,message):
    if not ok: raise RuntimeError(message)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def md(p,lines): Path(p).write_text('\n'.join(lines)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvrows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def table(p,rows):
    need(bool(rows),'Empty output table')
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def now(): return a0.now()
def run_path(p):
    p=Path(p).resolve();need(p.parent==BASE.resolve() and p.name.startswith('run_'),'Wrong pilot output root');return p
def source_hashes():
    actual={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    need(actual==read(HERE/'PACKAGE_MANIFEST.json'),'Pilot source changed')
    for name,digest in read(HERE/'INPUT_REFERENCE.json')['reusable_code_sha256'].items():
        need(sha(HERE/name)==digest,'Frozen diagnostic helper was modified: '+name)
    return dict(pilot=actual,frozen=a1lib.check_sources())
def historical_audit():
    expected=read(HERE/'INPUT_REFERENCE.json')
    for tag,old in [('A0',A0),('A1',A1),('S',S)]:
        need(sha(old/'SUCCESS.json')==expected[tag+'_success_sha256'],'Frozen run identity differs: '+tag)
    for name,digest in expected['S_inputs'].items():need(sha(S/name)==digest,'Frozen scaling input differs: '+name)
    for name,digest in expected['D_inputs'].items():need(sha(D/name)==digest,'Frozen diagnostic source differs: '+name)
    # Reuse the previously audited source/protocol/seal extractor read-only.
    sys.path.insert(0,str(D/'tools'))
    spec=importlib.util.spec_from_file_location('frozen_exp6c1d_auditor',D/'tools/run_diagnostics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.sources()
    sp=read(S/'SUCCESS.json')
    for name,digest in sp['artifacts_sha256'].items():need(sha(S/name)==digest,'Old scaling artifact changed: '+name)
    return expected
def protected_inventory():
    roots=[REPO/n for n in ['experiments','results','outputs','数据集','assets/pretrained','docs']]
    roots += [a1lib.UPSTREAM]+[a1lib.UPSTREAM.parent/n for n in ['outputs','reports']]
    roots += [Path('/home/zixinfeng/Documents/HypEmo-Exp6B')/n for n in ['stage_a','stage_b','reports']]
    paths=set()
    for root in roots:
        if not root.exists():continue
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','.pytest_cache'] and not d.startswith('.venv') and Path(directory)/d not in [HERE,BASE]]
            for name in files:
                p=Path(directory)/name
                if p.is_file() and not name.endswith(('.pyc','.DS_Store')):paths.add(p)
    return {str(p):sha(p) for p in sorted(paths)}
def audit_unchanged(root):
    need(protected_inventory()==read(root/'protected_before.json'),'Earlier research inputs/results changed')
    historical_audit()
    need(source_hashes()==read(root/'source_manifest.json'),'Pilot/frozen source changed')
def artifact_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and p.name!='SUCCESS.json' and not (p.parent==root and p.name.endswith(('.log','_exit_code.txt'))) and p.name!='exit_code.txt'
        and not {'runtime_cache','review','__pycache__'} & set(p.relative_to(root).parts)}
def extended_score(gold,probs,names):
    from sklearn.metrics import average_precision_score
    metrics,rows,rare,pred=score(gold,probs,names)
    aps=[]
    for k,row in enumerate(rows):
        positive=gold[:,k]==1;predicted=pred[:,k]==1
        ap=float(average_precision_score(gold[:,k],probs[:,k])) if positive.any() else None
        row.update(average_precision=ap,
            mean_positive_probability=float(probs[positive,k].mean()) if positive.any() else None,
            mean_predicted_positive_probability=float(probs[predicted,k].mean()) if predicted.any() else None)
        if ap is not None:aps.append(ap)
    metrics.update(macro_average_precision=float(np.mean(aps)) if aps else None,
        average_precision_defined_labels=len(aps),mean_positive_probability_definition='mean probability over gold-positive examples per label')
    return metrics,rows,rare,pred
def save_validation(root,stem,logits,gold,ids,names):
    probs=probabilities(logits);metrics,rows,rare,pred=extended_score(gold,probs,names)
    np.savez_compressed(root/(stem+'_predictions.npz'),logits=logits,probabilities=probs,targets=gold,
        ids=np.asarray(ids,dtype=str),label_names=np.asarray(names,dtype=str),predictions=pred,
        predicted_positive_counts=pred.sum(1),true_positive_counts=gold.sum(1))
    write(root/(stem+'_metrics.json'),metrics);table(root/(stem+'_per_label.csv'),rows)
    return metrics,rows
def historical_validation(root,names):
    evidence=root/'historical_validation';evidence.mkdir(exist_ok=True)
    result={}
    for tag,old in [('A0',A0),('A1',A1)]:
        with np.load(old/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['label_names'].tolist()==names,'Historical label order differs')
            probs=probabilities(z['logits']);metrics,rows,_,pred=extended_score(z['targets'],probs,names)
            need(np.array_equal(z['predictions'],pred),'Historical threshold predictions differ')
            recorded=read(old/'validation_metrics.json')
            need(all(metrics[k]==v for k,v in recorded.items()),'Historical Validation metrics differ')
            if result:need(np.array_equal(z['ids'],ids) and np.array_equal(z['targets'],gold),'Unpaired historical Validation arrays')
            ids=z['ids'].copy();gold=z['targets'].copy()
        result[tag]=dict(metrics=metrics,per_label=rows)
        write(evidence/(tag+'_metrics.json'),metrics);table(evidence/(tag+'_per_label.csv'),rows)
    given={'A0':[63.4011,58.1356,55.0248],'A1':[63.9513,56.0209,44.7456]}
    for tag,values in given.items():
        need(np.allclose([100*result[tag]['metrics'][k] for k in ['micro_f1','macro_f1','rare_f1']],values,rtol=0,atol=.000051),'Prompt historical values differ from raw saved Validation')
    write(evidence/'AUDIT.json',dict(raw_files_sha256={tag:sha(old/'validation_predictions.npz') for tag,old in [('A0',A0),('A1',A1)]},
        historical_test_predictions_used_for_comparison=False))
    return result
