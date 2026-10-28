"""Paired seed extension only. Frozen models/losses imported without modification."""
from pathlib import Path
import os,random,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'exp6d1_normalized_distance_bce'))
import ncommon as nc
from ncommon import read,write,sha,table,csvrows,need,md,now,state_hash,a0,extended_score,probabilities,save_validation
from model27 import load_model as build_a0,optimizer_for
from scaled_model import build as build_a1s
from nmodel import build as build_n1
from data27 import Collator,targets
BASE=REPO/'results/exp6d2_paired_multiseed'
KINDS=('A0','A1S','N1');NEW_SEEDS=(123,2026);ALL_SEEDS=(42,123,2026)
OLD=dict(A0=nc.A0,A1S=nc.CONTROL,N1=REPO/'results/exp6d1_normalized_distance_bce/run_20261023T042735Z_T60IQScg')
STATUS='EXP6D2_PAIRED_THREE_SEED_VALIDATION_COMPLETE'

def runtime():
    # The frozen runtime insists on hashseed42. Retain its platform/package/GPU
    # gates here, changing only the allowed preregistered hashseed. Never spoof
    # os.environ or patch the frozen helper to make that old gate pass.
    need(a0.platform.system()=='Linux' and a0.platform.node()=='D3020-A3','GPU work requires Ubuntu D3020-A3')
    need(REPO==Path('/home/zixinfeng/Documents/context-aware-hyperbolic-emotion'),'Wrong Ubuntu repository')
    need(os.environ.get('PYTHONHASHSEED') in {str(s) for s in ALL_SEEDS},'Unregistered process hash seed')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Missing deterministic CUBLAS setting')
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Need exactly one visible CUDA GPU')
    need('4090' in torch.cuda.get_device_name(0),'Expected RTX4090')
    current=dict(host=a0.platform.node(),python=sys.version,executable=sys.executable,
        packages={p:a0.version(p) for p in ['torch','transformers','numpy','scikit-learn','tokenizers','safetensors']},
        pip_freeze=a0.capture([sys.executable,'-m','pip','freeze']),gpu=torch.cuda.get_device_name(0),cuda_runtime=torch.version.cuda,
        nvidia_smi=a0.capture(['nvidia-smi']),git_commit=a0.capture(['git','-C',str(REPO),'rev-parse','HEAD']),
        git_status=a0.capture(['git','-C',str(REPO),'status','--short']),
        env={k:os.environ.get(k) for k in ['PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE']})
    historical=read(OLD['A0']/'plan.json')['runtime']
    for k in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:
        need(current[k]==historical[k],'Runtime differs from frozen A0: '+k)
    return current

def seed_all(seed):
    need(seed in ALL_SEEDS,'Only preregistered seeds')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def make_loader(records,tokenizer,seed,shuffle=False):
    need(seed in ALL_SEEDS,'Unregistered data seed')
    from torch.utils.data import DataLoader
    return DataLoader(records,batch_size=16 if shuffle else 64,shuffle=shuffle,collate_fn=Collator(tokenizer),
        num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed) if shuffle else None)

def planned_orders(ids,seed):
    # Same DataLoader generator consumption as the frozen implementation,
    # including its per-iterator base seed. Tokenization does not affect indices.
    from torch.utils.data import DataLoader
    dl=DataLoader(list(ids),batch_size=16,shuffle=True,collate_fn=list,num_workers=0,pin_memory=True,
        generator=torch.Generator().manual_seed(seed))
    return [[rid for batch in dl for rid in batch] for _ in range(4)]

def config(kind,seed):
    need(kind in KINDS and seed in NEW_SEEDS,'No seed42 training/new variants')
    base={'A0':a0.CONFIG,'A1S':nc.sc.CONFIG,'N1':nc.CONFIG}[kind]
    result=dict(base,experiment='Exp6D-2 paired multi-seed '+kind,seed=seed,variant=kind,
        new_test_inference=False,formal_test_inference_calls=0,training_data_generator_seed=seed,
        selection='full Validation Macro-F1 at .30, earliest tie',paired_seeds=list(ALL_SEEDS))
    if kind=='N1':
        result.update(matched_control='same-seed A1S cell in this Exp6D2 suite',formula_source_control=base['matched_control'],
            comparison='paired Validation-only N1-A1S; N1-A0 secondary')
    return result

def build(kind,snapshot,names):
    need(kind in KINDS,'Unknown variant')
    if kind=='A0':
        m,t,info=build_a0(snapshot);return m,t
    m,t,_,_=({'A1S':build_a1s,'N1':build_n1}[kind])(snapshot,names)
    return m,t

def control(model,kind):return model if kind=='A0' else model.control

def values(out,kind):
    if kind=='A0':
        loss=float(out['loss'].detach());return dict(loss=loss,bce=loss,weighted_bce=loss,auxiliary=0.,weighted_auxiliary=0.,max_ball_norm=0.)
    c=out['components'];b=c['ordinary_bce'] if kind=='N1' else c['classification']
    return dict(loss=float(out['loss'].detach()),bce=float(b.detach()),weighted_bce=float(c['classification'].detach()),
        auxiliary=float(c['auxiliary'].detach()),weighted_auxiliary=.1*float(c['auxiliary'].detach()),max_ball_norm=float(out['z'].detach().norm(dim=-1).max()))

def source_hashes():
    current={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    need(current==read(HERE/'PACKAGE_MANIFEST.json'),'D2 code changed')
    for directory,files in read(HERE/'FROZEN_SOURCE_REFERENCE.json').items():
        need({p.name:sha(p) for p in (HERE.parent/directory).iterdir() if p.is_file()}==files,'Frozen source changed: '+directory)
    return dict(D2=current,frozen=nc.check_sources())

def old_seals():
    source_hashes();nc.check_old_seals()
    p=read(OLD['N1']/'SUCCESS.json');need(p['status']==nc.STATUS and p['seed']==42 and p['new_test_inference_calls']==0,'N1 seed42 incomplete')
    for name,h in p['artifacts_sha256'].items():need(sha(OLD['N1']/name)==h,'N1 seed42 sealed artifact changed: '+name)
    need(read(OLD['N1']/'config.json')==nc.CONFIG,'N1 frozen protocol changed')
    need(read(OLD['N1']/'checkpoint_reload_verification.json')['logits_exact_equal'],'N1 seed42 lacks reload proof')

def protected_inventory(snapshot):
    roots=[REPO/n for n in ['experiments','results','outputs','数据集','assets/pretrained','docs']]
    roots += [nc.sc.a1lib.UPSTREAM]+[nc.sc.a1lib.UPSTREAM.parent/n for n in ['outputs','reports']]
    roots += [Path('/home/zixinfeng/Documents/HypEmo-Exp6B')/n for n in ['stage_a','stage_b','reports']]+[Path(snapshot)]
    result={}
    for root in roots:
        if not root.exists():continue
        print('D2_FREEZE_HASH_ROOT',root,flush=True)
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','runtime_cache','.pytest_cache'} and not d.startswith('.venv') and Path(directory)/d not in [HERE,BASE]]
            for name in files:
                p=Path(directory)/name
                if p.is_file() and not name.endswith(('.pyc','.DS_Store')):result[str(p)]=sha(p)
    return result

def freeze_check(suite):
    p=read(suite/'SUITE_PLAN.json')
    need(source_hashes()==p['source_manifest'],'Suite source changed')
    need(protected_inventory(p['snapshot'])==read(suite/'protected_before.json'),'Historical research changed')
    old_seals()

def dataset():
    data,manifest=nc.dataset()
    need(manifest==read(OLD['N1']/'dataset_manifest.json'),'N1 canonical data differs')
    return data,manifest

def artifacts(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=Path(root)/'SUCCESS.json'
        and not p.name.endswith(('.log','_exit_code.txt')) and p.name!='exit_code.txt'
        and not {'runtime_cache','__pycache__'} & set(p.relative_to(root).parts)}

def run_path(suite,seed,kind):
    suite=Path(suite).resolve();need(suite.parent==BASE.resolve() and suite.name.startswith('run_'),'Wrong suite root')
    need(seed in NEW_SEEDS and kind in KINDS,'Only six authorized cells')
    return suite/('seed'+str(seed))/kind

def historical_seed42(suite,data,manifest):
    directory=suite/'seed42_reused';directory.mkdir()
    expected_ids=[r['id'] for r in data['validation']];gold=targets(data['validation'])
    for kind,path in OLD.items():
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['label_names'].tolist()==manifest['names27'] and z['ids'].tolist()==expected_ids and np.array_equal(z['targets'],gold),'Historical seed42 data/labels not paired')
            m,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
            need(all(m[k]==v for k,v in read(path/'validation_metrics.json').items()),'Historical seed42 metrics differ')
            need(np.array_equal(pred,z['predictions']),'Historical seed42 predictions differ')
        write(directory/(kind+'_metrics.json'),m);table(directory/(kind+'_per_label.csv'),rows)
        write(directory/(kind+'_PROVENANCE.json'),dict(original_run=str(path),success_sha256=sha(path/'SUCCESS.json'),
            validation_predictions_sha256=sha(path/'validation_predictions.npz'),checkpoint_selection=read(path/'checkpoint_selection.json'),
            reused_seed=42,new_training=False,new_model_inference=False,split='validation'))
