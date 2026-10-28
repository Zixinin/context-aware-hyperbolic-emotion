"""Bind a new training plan to the completed audit and frozen calibration."""
import shutil
from ccommon import *


def prepare(root):
    need(not (root/'PLAN.json').exists(),'Fresh independent run only; no automatic retry')
    source=source_hashes();calibration=old_seals();env=runtime()
    data,manifest=data27();oldplan=read(OLD/'PLAN.json');snapshot=Path(oldplan['snapshot'])
    need(sc.a0.snapshot_inventory(snapshot)==oldplan['snapshot_sha256'],'Pretrained snapshot/tokenizer changed')
    need(sha(OLD/'config.json')==oldplan['config_sha256'],'Original audit config changed')
    need(read(OLD/'AUDIT_SUCCESS.json')['plan_sha256']==sha(OLD/'PLAN.json'),'Prior audit not bound to its plan')
    need(shutil.disk_usage(BASE).free>=8*1024**3,'Need8GiB free; no cleanup or dependency changes performed')
    shutil.copytree(HERE,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(root/'protected_before.json',protected_inventory(snapshot))
    write(root/'dataset_manifest.json',manifest);write(root/'config.json',CONFIG)
    shutil.copyfile(OLD/'fixed_emotion_embeddings.npy',root/'fixed_emotion_embeddings.npy')
    fixed=read(OLD/'FIXED_TRAIN_COHORTS.json')
    train_ids={r['id'] for r in data['train']}
    need(len(fixed['cohorts'])==10 and all(r['id'] in train_ids for rows in fixed['cohorts'].values() for r in rows),'Wrong fixed train probe selection')
    write(root/'FIXED_TRAIN_COHORTS.json',fixed)
    for epoch in range(1,5):
        order=read(OLD/f'planned_train_order_epoch{epoch}.json')
        need(len(order)==30587 and set(order)==train_ids and order==read(A1/f'train_order_epoch{epoch}.json'),'Historical full train order changed')
        write(root/f'planned_train_order_epoch{epoch}.json',order)
    reused=root/'prior_audit_reports';reused.mkdir()
    old_reports={}
    for path in sorted(OLD.glob('0[1-5]_*.md')):
        shutil.copyfile(path,reused/path.name);old_reports[path.name]=sha(path)
    need(len(old_reports)==5,'Missing previous four audits and metric definition')
    write(root/'REUSED_AUDITS.json',dict(source_run=str(OLD),files_sha256=old_reports,
        note='Historical A1 / original gamma=.1 audit preserved as evidence; current gamma/protocol are in config and NEW_PROTOCOL_CN.md'))
    shutil.copyfile(HERE/'PROTOCOL_CN.md',root/'NEW_PROTOCOL_CN.md')
    write(root/'FROZEN_GAMMA.json',dict(gamma_G=GAMMA,gamma_G_hex=GAMMA_HEX,calibration=calibration,
        constant_during_training=True,whole_auxiliary_weight=True,reference_CE_coefficient=.1,new_CE_coefficient=GAMMA))
    write(root/'PLAN.json',dict(created=now(),experiment='A1-G-C calibrated geodesic pilot',seed=42,
        source_sha256=source,calibration=calibration,snapshot=str(snapshot),snapshot_sha256=sc.a0.snapshot_inventory(snapshot),
        runtime=env,config_sha256=sha(root/'config.json'),data_sha256=sha(root/'dataset_manifest.json'),
        gamma_G=GAMMA,gamma_G_hex=GAMMA_HEX,input_scale=1.,epochs=4,formal_optimizer_steps=7648,
        two_discarded_smoke_steps=True,training_runs_authorized=1,new_test_calls=0,
        comparison='seed42 Validation A0/A1/A1-G-C; alignment AND auxiliary coefficient change',
        approval='Human requested next training command/code after reviewing completed calibration',
        pretraining_gates='calibration seal + initialization + actual gradient replay + finite/path/geometry + weighted aux/total norm <=10x on fresh train probes',
        earlier_gamma_point1_scale_gate_preserved=True,no_weight_search=True,no_other_seed=True))
    freeze_check(root)
    write(root/'AUDIT_SUCCESS.json',dict(status='CALIBRATED_PILOT_INPUTS_AND_PROTOCOL_PASS',
        plan_sha256=sha(root/'PLAN.json'),gamma_G=GAMMA,completed=now()))
    print('CALIBRATED_PILOT_INPUTS_PASS — NEXT: ACTUAL MODEL SMOKE, THEN FOUR-EPOCH TRAINING',flush=True)
