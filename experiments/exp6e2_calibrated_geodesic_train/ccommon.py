"""Independent one-pilot paths and frozen calibration/protocol provenance."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'exp6e_geodesic_alignment'))
import gcommon as g
from gcommon import np,torch,sc,read,write,sha,table,csvrows,need,md,now,state_hash,extended_score,probabilities,save_validation
from gcommon import runtime,parameter_groups,seed_everything,load_dataset,loader,Collator,targets,isolated_eval,ScalingTrace,release
from cmodel import GAMMA,GAMMA_HEX,install_weight

BASE=REPO/'results/exp6e2_calibrated_geodesic_train'
A0=g.A0;A1=g.A1;UPSTREAM=g.UPSTREAM
OLD=REPO/'results/exp6e_geodesic_alignment/run_20261027T115121Z_clE7F1kT'
CAL=REPO/'results/exp6e1_gradient_calibration/run_20261028T080408Z_vITsTuCl'
CAL_SOURCE=HERE.parent/'exp6e1_gradient_calibration'
CONFIG=dict(g.CONFIG,experiment='Exp6E-2 A1-G-C calibrated geodesic pilot',
    loss='mean_BCE_plus_frozen_gamma_G_soft_target_CE_and_geodesic_alignment',
    auxiliary_weight=GAMMA,calibrated_gamma_hex=GAMMA_HEX,hyperbolic_input_scale=1.,
    calibration_rule='fresh_train_log_median_shared_encoder_v1',
    comparison='geodesic objective AND whole-aux coefficient change relative to A1; not alignment-only',
    auxiliary_CE_effective_coefficient=GAMMA,reference_A1_auxiliary_CE_coefficient=.1,
    gamma_schedule='constant; never recalibrate during training',new_test_inference=False)
STATUS='EXP6E2_A1GC_SEED42_VALIDATION_PILOT_COMPLETE'
SEVERE_RATIO=10.


def build(snapshot,names,geodesic=True):
    model,tok,loading,identity=g.build(snapshot,names,geodesic=geodesic)
    return (install_weight(model) if geodesic else model),tok,loading,identity


def source_hashes():
    current={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    need(current==read(HERE/'PACKAGE_MANIFEST.json'),'New pilot source changed')
    ref=read(HERE/'INPUT_REFERENCE.json')
    for name,expected in ref['frozen_source_directories'].items():
        path=HERE.parent/name
        need({p.name:sha(p) for p in path.iterdir() if p.is_file()}==expected,'Frozen source changed: '+name)
    g.source_hashes()
    return current


def verify_calibration():
    # Invoke the exact existing stdlib verifier read-only; never rewrite approval.
    import contextlib,io,importlib.util
    sys.path.insert(0,str(CAL_SOURCE))
    spec=importlib.util.spec_from_file_location('frozen_exp6e1_calibration',CAL_SOURCE/'calibrate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):module.verify(CAL)
    proposal=read(CAL/'FROZEN_GAMMA_PROPOSAL.json')
    need(proposal['gamma_G']==GAMMA and proposal['gamma_G_hex']==GAMMA_HEX,'Gamma differs from sealed calibration')
    need((CAL/'exit_code.txt').read_text().strip()=='0','Calibration did not complete')
    return dict(run=str(CAL),success_sha256=sha(CAL/'SUCCESS.json'),proposal_sha256=sha(CAL/'FROZEN_GAMMA_PROPOSAL.json'),
        gamma_G=GAMMA,gamma_G_hex=GAMMA_HEX,training_authorization='current human request; original proposal preserved')


def old_seals():
    g.old_seals()
    return verify_calibration()


def protected_inventory(snapshot):
    # Include original Exp6E and calibration, exclude ONLY this new experiment.
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
    plan=read(root/'PLAN.json')
    need(source_hashes()==plan['source_sha256'],'Plan/source mismatch')
    need(verify_calibration()==plan['calibration'],'Frozen calibration changed')
    need(protected_inventory(plan['snapshot'])==read(root/'protected_before.json'),'Protected historical artifacts changed')
    g.old_seals()


def artifact_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'SUCCESS.json'
        and not p.name.endswith(('.log','_exit_code.txt')) and p.name!='exit_code.txt'
        and not {'runtime_cache','__pycache__'} & set(p.relative_to(root).parts)}


def data27():
    return g.data27()


def diagnostic_values(out):
    c=out['components'];val=lambda x:float(x.detach())
    bce=val(c['classification']);aux=val(c['auxiliary']);ce=val(c['positive_ce'].mean());distance=val(c['positive_distance'].mean())
    need(bce>0 and all(np.isfinite(v) for v in [bce,aux,ce,distance]),'Invalid loss scalar')
    return dict(loss=val(out['loss']),bce=bce,auxiliary=aux,weighted_auxiliary=GAMMA*aux,
        positive_ce_mean=ce,positive_dot_mean=val(c['positive_dot'].mean()),positive_distance_mean=distance,
        weighted_ce=GAMMA*ce,weighted_distance=GAMMA*distance,
        coordinate_auxiliary_same_forward=val(c['coordinate_auxiliary']),weighted_aux_to_BCE=GAMMA*aux/bce,
        max_ball_norm=val(out['z'].norm(dim=-1).max()))


def run_path(path):
    root=Path(path).resolve()
    need(root.parent==BASE.resolve() and root.name.startswith('run_'),'Wrong independent pilot result root')
    return root
