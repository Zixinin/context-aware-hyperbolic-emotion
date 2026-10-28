"""Read sealed fresh-initialization statistics; freeze ONE proposed coefficient.

This is a new CPU extraction/calibration run, not a repeated forward/backward.
Only `run` and read-only `verify` exist. There is no training entry point.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from cal_math import RULE, calibrate, module_contributions, need, number

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO/'results/exp6e1_gradient_calibration'
OLD = REPO/'results/exp6e_geodesic_alignment/run_20261027T115121Z_clE7F1kT'
E_SOURCE = REPO/'experiments/exp6e_geodesic_alignment'
MODULES = {'shared_encoder','classifier','hyperbolic_projection','auxiliary_decoder'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csvrows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def table(path, rows):
    need(bool(rows),'Empty output table')
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_map(root):
    return {p.name:sha(p) for p in sorted(Path(root).iterdir()) if p.is_file()}


def own_source():
    actual=file_map(HERE)
    need(actual.pop('PACKAGE_MANIFEST.json') == sha(HERE/'PACKAGE_MANIFEST.json'),'Missing manifest')
    need(actual == read(HERE/'PACKAGE_MANIFEST.json'),'Calibration source changed')
    return file_map(HERE)


def safe_relative(name):
    p=PurePosixPath(name)
    need(not p.is_absolute() and p.parts and all(v not in ['.','..'] for v in p.parts),'Unsafe sealed path')
    return p


def old_artifacts():
    # Same inclusion/exclusion rules as the actual, pinned Exp6E artifact_hashes.
    result={}
    for p in sorted(OLD.rglob('*')):
        if (not p.is_file() or p in [OLD/'SUCCESS.json',OLD/'DIAGNOSTIC_SUCCESS.json']
            or p.name.endswith(('.log','_exit_code.txt')) or p.name=='exit_code.txt'
            or {'runtime_cache','__pycache__'} & set(p.relative_to(OLD).parts)):
            continue
        need(not p.is_symlink(),'Symlink in old sealed artifacts')
        result[str(p.relative_to(OLD))]=sha(p)
    return result


def inputs():
    reference=read(HERE/'INPUT_REFERENCE.json')
    need(file_map(E_SOURCE)==reference['geodesic_source_files'],'Original Exp6E source changed')
    for name,expected in read(E_SOURCE/'FROZEN_SOURCES.json').items():
        need(file_map(REPO/'experiments'/name)==expected,'Historical source changed: '+name)
    seal=read(OLD/'DIAGNOSTIC_SUCCESS.json')
    for name in seal['artifacts_sha256']:safe_relative(name)
    need(seal['status']=='PRETRAIN_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED','Wrong original diagnostic status')
    need(old_artifacts()==seal['artifacts_sha256'],'Original diagnostic artifact seal mismatch')
    need((OLD/'exit_code.txt').read_text().strip()=='3','Original diagnostic did not stop as reviewed')
    need(not (OLD/'FORMAL_TRAINING_CLAIM.json').exists() and not (OLD/'SMOKE_SUCCESS.json').exists(),'Original scope changed')
    proof=read(OLD/'probes/fresh_initial/NO_UPDATE_PROOF.json')
    initial=read(OLD/'INITIALIZATION_PARITY.json')
    need(proof['state_before']==proof['state_after']==initial['full_initial_state_sha256'],'Fresh model state identity differs')
    need(proof['optimizer_steps']==proof['test_forward_calls']==0 and proof['all_grads_none'],'Original probe mutated model')
    for key in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']:
        need(proof[key] is True,'Missing restored-state proof: '+key)
    need(initial['fresh_pretrained'] is True and initial['no_checkpoint_continuation'] is True and initial['input_scale']==1.,'Not original fresh unscaled A1')
    need(initial['all_initial_forward_tensors_exact'] and initial['constructor_rng_exact'],'Initialization parity incomplete')
    original_a1=REPO/'results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL'
    need(sha(original_a1/'SUCCESS.json')==reference['A1_success_sha256'],'A1 seal identity differs')
    need(sha(original_a1/'INITIALIZATION_PARITY.json')==initial['historical_A1_initialization_sha256'],'Historical initialization changed')
    historic=read(original_a1/'INITIALIZATION_PARITY.json')
    for key in ['shared_state_sha256','hyperbolic_initial_state_sha256']:
        need(initial[key]==historic[key],'Fresh parameters differ from A1')
    need(read(OLD/'PLAN.json')['source_sha256']==read(E_SOURCE/'PACKAGE_MANIFEST.json'),'Source/old plan mismatch')
    # No validation/test dataset or prediction file is read for calibration.
    manifest=read(OLD/'dataset_manifest.json')
    train_path=REPO/'数据集/GoEmotions/processed/train.tsv'
    need(sha(train_path)==manifest['splits']['train']['source_tsv_sha256']==reference['train_tsv_sha256'],'Train TSV changed')
    ids=set(manifest['splits']['train']['retained_ids'])
    need(len(ids)==30587 and len(manifest['names27'])==27 and 'neutral' not in manifest['names27'],'Wrong canonical27 train split')
    cohorts=read(OLD/'FIXED_TRAIN_COHORTS.json')['cohorts']
    need(set(cohorts)==set(reference['fresh_printed_reference']),'Cohort identities changed')
    fixed=[r['id'] for batch in cohorts.values() for r in batch]
    need(len(cohorts)==10 and all(len(batch)==16 for batch in cohorts.values()) and len(fixed)==160 and len(set(fixed))==145,'Wrong fixed train cohort size')
    need(set(fixed)<=ids,'Calibration contains non-train IDs')
    all_rows=csvrows(OLD/'probes/fresh_initial/MODULE_GRADIENTS.csv')
    need(len(all_rows)==40,'Need ALL 10x4 actual module statistics')
    keys={(r['cohort'],r['module']) for r in all_rows}
    need(len(keys)==40 and keys=={(cohort,module) for cohort in cohorts for module in MODULES},'Missing/duplicate gradient rows')
    for r in all_rows:
        need(r['tag']=='fresh_initial','A trained-checkpoint row must not enter calibration')
        b, o, n = (number(r[k]) for k in ['bce_norm','coordinate_aux_norm','geodesic_aux_norm'])
        connected=[int(r[k]) for k in ['bce_connected','coordinate_aux_connected','geodesic_aux_connected']]
        if r['module']=='classifier':
            need(b>0 and o==n==0 and connected[0]>0 and connected[1:]==[0,0],'Classifier auxiliary path wrong')
        elif r['module']=='shared_encoder':
            need(min(b,o,n)>0 and min(connected)>0,'Shared encoder path wrong')
        else:
            need(b==0 and min(o,n)>0 and connected[0]==0 and min(connected[1:])>0,'HNN/decoder BCE path wrong')
    shared=[r for r in all_rows if r['module']=='shared_encoder']
    for r in shared:
        # Independent earlier user-returned printed values, not a gamma target.
        for field,rounded in reference['fresh_printed_reference'][r['cohort']].items():
            need(abs(number(r[field])-rounded)<=5.1e-7,'Full-precision CSV does not reproduce reviewed terminal output')
    lossrows=csvrows(OLD/'probes/fresh_initial/LOSS_SCALES.csv')
    need(len(lossrows)==10 and {r['cohort'] for r in lossrows}==set(cohorts),'Fresh losses missing')
    # Reconstruct global old/new gradient norms from ALL modules to check schema.
    for loss in lossrows:
        rows=[r for r in all_rows if r['cohort']==loss['cohort']]
        for prefix,field in [('coordinate','old_total_gradient_norm'),('geodesic','new_total_gradient_norm')]:
            norm=math.sqrt(sum(number(r[prefix+'_total_norm'])**2 for r in rows))
            need(math.isclose(norm,number(loss[field]),rel_tol=2e-6,abs_tol=1e-9),'Global norm/schema inconsistency')
    used=['DIAGNOSTIC_SUCCESS.json','INITIALIZATION_PARITY.json','PLAN.json','dataset_manifest.json','FIXED_TRAIN_COHORTS.json',
          'probes/fresh_initial/NO_UPDATE_PROOF.json','probes/fresh_initial/MODULE_GRADIENTS.csv','probes/fresh_initial/LOSS_SCALES.csv']
    observed=dict(old_run=str(OLD),read_only_inputs_sha256={str(OLD/p):sha(OLD/p) for p in used},
        original_artifact_count=len(seal['artifacts_sha256']),fresh_state_sha256=proof['state_before'],
        positions=160,unique_train_ids=145,cohorts=10,train_tsv_sha256=sha(train_path),
        geodesic_source_sha256=file_map(E_SOURCE),calibration_source_sha256=own_source())
    return shared,all_rows,observed


def output_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and p.name not in ['SUCCESS.json','exit_code.txt'] and not p.name.endswith('.log') and '__pycache__' not in p.parts}


def interpretation(gamma,shared,module_rows):
    global_rows=[]
    for cohort in [r['cohort'] for r in shared]:
        mm=[r for r in module_rows if r['cohort']==cohort]
        old=math.sqrt(sum(r['old_total_norm']**2 for r in mm))
        new=math.sqrt(sum(r['matched_total_norm']**2 for r in mm))
        global_rows.append(dict(cohort=cohort,old_global_norm=old,matched_global_norm=new,
            matched_to_old_global_norm=new/old,old_would_clip_at_norm1=old>1,matched_would_clip_at_norm1=new>1))
    return dict(gamma_G=gamma,gamma_G_hex=gamma.hex(),coefficient_on_both_CE_and_distance=gamma,
        original_auxiliary_CE_coefficient=.1,CE_coefficient_ratio=gamma/.1,
        log_median_matched_aux_norm_ratio=math.exp(statistics.median(math.log(r['matched_to_old_aux_norm']) for r in shared)),
        matched_to_old_shared_aux_norm_range=[min(r['matched_to_old_aux_norm'] for r in shared),max(r['matched_to_old_aux_norm'] for r in shared)],
        bce_new_aux_cosine_range=[min(r['bce_geodesic_aux_cosine'] for r in shared),max(r['bce_geodesic_aux_cosine'] for r in shared)],
        conflict_cohorts=[r['cohort'] for r in shared if r['geodesic_direction_conflicts_with_bce']],
        global_norm_rows=global_rows,direction_matched=False,training_stability_verified=False,
        geometric_superiority_proven=False,pilot_authorized=False)


def report(result,rows):
    gamma=result['gamma_G']
    lines=['# Exp6E-1：Fresh train-only 梯度尺度校准','',
        '**只完成校准，没有训练。** 本轮CPU只读提取上轮已封存的fresh初始化真实梯度统计；没有重新forward/backward、optimizer、Validation/Test推理或F1选择。',
        '来源：'+str(OLD)+'；仅fresh_initial参与数值校准，old A1-best不参与系数或方向判定。',
        '本轮计算结果由完整精度CSV产生，非六位小数终端表估算；文件完整性、train身份、初始化和无更新证明均复验。','',
        '## 唯一系数及预定规则','',f'`gamma_G = {gamma!r}`（double hex `{gamma.hex()}`）。',
        '令a_b=∇encoder mean(CE−dot)、g_b=∇encoder mean(CE+distance)。每组精确匹配系数r_b=.1||a_b||/||g_b||。',
        'gamma_G=exp(median_b(log(r_b)))；10个原有固定cohorts等权，偶数中位数取两个中央log的平均。没有候选grid、裁剪系数或按方向/性能择优。',
        '保存梯度已经乘.1：实现使用r_b=.1*(saved_old_aux_norm/saved_new_aux_norm)，避免漏乘或重复乘.1。',
        '同batch BCE不变，所以匹配aux范数也匹配aux/BCE范数比；这并不匹配梯度方向，也不保证逐组精确匹配或训练全程匹配。','',
        '|cohort|单组精确系数|固定gamma下aux/旧aux|旧aux/BCE|新aux/BCE|BCE与旧aux cos|BCE与新aux cos|旧与新aux cos|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+r['cohort']+'|'+'|'.join(f'{r[k]:.6f}' for k in ['individual_exact_matching_gamma','matched_to_old_aux_norm','old_aux_to_bce','matched_aux_to_bce','bce_old_aux_cosine','bce_geodesic_aux_cosine','old_new_aux_cosine'])+'|')
    lines+=['','## 方向、边界与局限','',
        '新辅助与BCE出现负cos的组：'+(', '.join(result['conflict_cohorts']) or '无')+'。正系数不会改变这些夹角；没有因方向不好删除组或另挑系数。方向不是正确性证明，也不等于该单标签的梯度/未来F1变化。',
        '固定样本有重叠：160位置、145个独立train IDs，分层覆盖稀有类；等cohort校准不是整套训练分布的无偏估计。eval/dropout关闭下得到的匹配不保证train/dropout与未来更新轨迹匹配。',
        '这是对同一fresh前向的梯度Gram重加权：||BCE+gamma*g||²由范数及cos精确重构。GLOBAL_GRADIENT_IMPLICATIONS.csv中的clip1标记只是局部反事实，不能冒充真实训练clip比例或Adam更新幅度。',
        's=1保持不变，已有首次映射严重截断问题仍存在；降低aux系数不会修复前向截断。本轮没有改标签、曲率、投影、距离、BCE、CE公式或分类头。',
        f'注意：gamma_G乘整个CE+distance，因此辅助CE系数由.1变为{gamma!r}；CE内部公式与targets保持，但实际权重不同。已有统计未分别保存CE与distance的梯度，不能宣称两分量各自匹配。',
        '## 后续仅为待批准协议','',
        '若后续批准，L=BCE+gamma_G*mean(CE+mean_positive d_1)，使用本文件冻结的唯一常数，不逐batch/epoch重估。',
        '命名为A1-G-C：测地目标+整个aux梯度尺度匹配；与A1比较改变了对齐形式和系数，不能称严格仅替换内积。',
        '原预训练snapshot独立初始化；27类原IDs/划分/标签顺序、seed42、s=1、batch16/eval64、4epochs、7648步、warmup765、AdamW5e-5、原WD、clip1、阈值.30、Validation Macro-F1选轮/earliest tie均保持。',
        'Validation报告Macro/Micro/Rare-F1、P/R、Macro AP和全部27标签/固定Rare7；记录CE/距离分量、梯度方向/强度、实际训练clip及首/后续投影截断。无新Test、更多seed、s=.03或系数搜索。',
        'FROZEN_GAMMA_PROPOSAL.json冻结计算候选，不授权训练。需要审阅本报告后决定是否训练；现有原Exp6E暂停记录不修改、不绕门。',
        '若结果不佳保留负结果，不连续搜系数；后续几何独立价值仍需matched Euclidean control。']
    return '\n'.join(lines)+'\n'


def run(root):
    need(not list(root.glob('PLAN.json')),'Existing calibration plan; do not retry')
    write(root/'PLAN.json',dict(created=now(),rule=RULE,source_sha256=own_source(),
        input_run=str(OLD),execution='stdlib-only CPU extraction of sealed fresh REAL gradients',
        model_forward_calls=0,backward_calls=0,optimizer_steps=0,validation_forwards=0,test_forwards=0,
        authorized_scope='calibration only; training requires later review'))
    shared,all_rows,before=inputs()
    write(root/'INPUTS_BEFORE.json',before)
    gamma,rows=calibrate(shared)
    modules=module_contributions(all_rows,gamma)
    result=interpretation(gamma,rows,modules)
    table(root/'SHARED_ENCODER_CALIBRATION.csv',rows)
    table(root/'ALL_MODULE_CONTRIBUTIONS.csv',modules)
    table(root/'GLOBAL_GRADIENT_IMPLICATIONS.csv',result['global_norm_rows'])
    write(root/'RESULT.json',result)
    write(root/'FROZEN_GAMMA_PROPOSAL.json',dict(gamma_G=gamma,gamma_G_hex=gamma.hex(),rule=RULE,
        input_gram_sha256=sha(OLD/'probes/fresh_initial/MODULE_GRADIENTS.csv'),
        plan_sha256=sha(root/'PLAN.json'),model='A1-G-C proposed',
        full_auxiliary_coefficient=True,formal_training_authorized=False,new_Test=False))
    (root/'CALIBRATION_REPORT_CN.md').write_text(report(result,rows),encoding='utf-8')
    _,_,after=inputs()
    need(before==after,'Readonly calibration input hashes changed')
    write(root/'NO_UPDATE_PROOF.json',dict(read_only_inputs_equal=True,inputs_before=before,inputs_after=after,
        model_loaded=False,new_forward_calls=0,new_backward_calls=0,optimizer_created=False,optimizer_steps=0,
        validation_inference_calls=0,test_inference_calls=0,validation_metric_search=False,environment_changed=False))
    write(root/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch_imported='torch' in sys.modules,dependencies_installed=False))
    need('torch' not in sys.modules,'Unexpected model framework import')
    write(root/'SUCCESS.json',dict(status='FRESH_TRAIN_ONLY_GRADIENT_CALIBRATION_COMPLETE_NO_TRAINING',
        artifacts_sha256=output_hashes(root),completed=now()))
    verify(root)


def verify(root):
    seal=read(root/'SUCCESS.json')
    need(seal['status']=='FRESH_TRAIN_ONLY_GRADIENT_CALIBRATION_COMPLETE_NO_TRAINING','Wrong calibration status')
    need(seal['artifacts_sha256']==output_hashes(root),'Calibration artifacts changed')
    plan=read(root/'PLAN.json')
    need(plan['source_sha256']==own_source() and plan['rule']==RULE,'Calibration source/rule drift')
    shared,all_rows,current=inputs()
    need(current==read(root/'INPUTS_BEFORE.json'),'Original inputs changed after calibration')
    gamma,rows=calibrate(shared)
    result=interpretation(gamma,rows,module_contributions(all_rows,gamma))
    need(result==read(root/'RESULT.json'),'Calibrated result cannot be reproduced')
    proposal=read(root/'FROZEN_GAMMA_PROPOSAL.json')
    need(proposal['gamma_G']==gamma and proposal['gamma_G_hex']==gamma.hex() and proposal['rule']==RULE and not proposal['formal_training_authorized'],'Frozen proposal differs')
    need(proposal['input_gram_sha256']==sha(OLD/'probes/fresh_initial/MODULE_GRADIENTS.csv') and proposal['plan_sha256']==sha(root/'PLAN.json'),'Gamma provenance mismatch')
    need((root/'CALIBRATION_REPORT_CN.md').read_text(encoding='utf-8')==report(result,rows),'Report mismatch')
    proof=read(root/'NO_UPDATE_PROOF.json')
    need(proof['inputs_before']==proof['inputs_after']==current and proof['read_only_inputs_equal'],'No-update seal mismatch')
    need(not proof['model_loaded'] and not proof['optimizer_created'] and not proof['environment_changed'] and not proof['validation_metric_search'],'Unexpected scope')
    for key in ['new_forward_calls','new_backward_calls','optimizer_steps','validation_inference_calls','test_inference_calls']:
        need(proof[key]==0,'Unexpected training/inference')
    print('EXP6E1_FRESH_TRAIN_ONLY_CALIBRATION_SAVED_VERIFICATION_PASS',flush=True)
    print((root/'CALIBRATION_REPORT_CN.md').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','verify']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    need(root.parent==BASE.resolve() and root.name.startswith('run_'),'Wrong independent calibration output root')
    try:
        {'run':run,'verify':verify}[a.phase](root)
    except BaseException as error:
        if a.phase=='run':
            write(root/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True,no_training=True))
        raise
