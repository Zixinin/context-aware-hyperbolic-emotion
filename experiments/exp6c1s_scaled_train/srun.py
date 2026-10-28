"""Only s=.03 seed42; frozen A1 primitives, validation-only pilot. No Test path."""
from __future__ import annotations
import argparse,gc,json,math,shutil,sys,traceback
from pathlib import Path
import numpy as np
import torch
from scommon import *
from scaled_model import build,unscaled_build,install_scale
from probes import run_probe,isolated_eval
from a1model import train_step,parameter_groups,diagnostic_values
from a1run import initial_parity,formula_and_gradient_case,predict
from data27 import loader,load_dataset,save_audit,targets,Collator
from model27 import seed_everything
from run import compare_reload
from reporting import write_reports

def release():gc.collect();torch.cuda.empty_cache()
def stage(root,name):
    write(root/'state.json',dict(stage=name,time=now()));print('STAGE',name,flush=True)
def dataset():
    data,manifest=load_dataset()
    need(manifest==read(A0/'dataset_manifest.json')==read(A1/'dataset_manifest.json'),'Canonical dataset changed')
    # Keep split sizes in the manifest, but remove Test records from every live
    # training/evaluation function. No predictor accepts a Test split argument.
    del data['test']
    return data,manifest
def evaluate(model,tokenizer,data,smoke=False):
    records=data['validation'][:64] if smoke else data['validation']
    result=predict(model,loader(records,tokenizer),'smoke Validation' if smoke else 'Validation')
    need(np.array_equal(result[2],[r['id'] for r in records]) and np.array_equal(result[1],targets(records)),'Noncanonical Validation evaluation')
    return result
def smoke_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SMOKE_VERIFIED.json'}
def verify_smoke(root):
    proof=read(root/'smoke/SMOKE_VERIFIED.json')
    need(proof['status']=='EXP6C1S_SMOKE_PASS' and proof['artifacts_sha256']==smoke_hashes(root/'smoke'),'Smoke artifacts differ')
    need(proof['plan_sha256']==sha(root/'plan.json') and proof['source_sha256']==source_hashes(),'Smoke plan/source differs')
    need(proof['initialization_sha256']==sha(root/'INITIALIZATION_PARITY.json') and proof['scaled_initialization_sha256']==sha(root/'SCALED_INITIALIZATION_PARITY.json'),'Smoke initialization proofs differ')
    need(read(root/'config.json')==CONFIG,'Pilot protocol changed')
    print('EXP6C1S_SAVED_SMOKE_VERIFICATION_PASS',flush=True)

def scaled_initial_parity(model,x,root):
    state=state_hash(model.state_dict());cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        def reset():torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        reset();model.train();old_train=model(**x)['logits'].clone()
        train_cpu=torch.get_rng_state().clone();train_gpu=torch.cuda.get_rng_state().clone()
        reset();model.eval();old_eval=model(**x)['logits'].clone()
        reset();install_scale(model)
        need(torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'Scale installation consumed RNG')
        model.train();new_train=model(**x)['logits'].clone()
        need(torch.equal(train_cpu,torch.get_rng_state()) and torch.equal(train_gpu,torch.cuda.get_rng_state()),'Scale changed train-forward RNG consumption')
        reset();model.eval();new_eval=model(**x)['logits'].clone()
    need(torch.equal(old_train,new_train) and torch.equal(old_eval,new_eval),'Scaling changed initial classification logits')
    need(state_hash(model.state_dict())==state,'Scale changed parameters or state-dict keys')
    write(root/'SCALED_INITIALIZATION_PARITY.json',dict(state_before=state,state_after=state,
        classification_train_logits_exact=True,classification_eval_logits_exact=True,
        max_abs_train_error=float((new_train-old_train).abs().max()),max_abs_eval_error=float((new_eval-old_eval).abs().max()),
        initialization_rng_unchanged=True,initial_train_forward_rng_consumption_equal=True,
        scalar_multiplication_count_at_aux_entry=1,classifier_input_scaled=False,parameters_added=0,
        training_historical_masks_note='Recorded initialization and probe RNG parity only; no claim of per-step historical dropout mask inspection.'))

def replay_diagnosis(root,snapshot,names,data,selection):
    stage(root,'read_only_previous_A1_best_geometry_replay_not_training_initialization')
    seed_everything();model,tokenizer,_,_=unscaled_build(snapshot,names)
    model.load_state_dict(torch.load(A1/'best_checkpoint/model.pt',map_location='cpu',weights_only=True),strict=True)
    model.to('cuda').eval();need(state_hash(model.state_dict())==read(S/'NO_UPDATE_PROOF.json')['model_state_sha256_before'],'Not previous diagnostic checkpoint')
    by_id={r['id']:r for r in data['train']}
    with isolated_eval(model),torch.no_grad():
        for cohort,rows in selection['cohorts'].items():
            batch=Collator(tokenizer)([by_id[r['id']] for r in rows]);ids=batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
            out=model(**x)
            with np.load(S/'cache'/f'{cohort}_s1p0.npz',allow_pickle=False) as old:
                need(np.array_equal(old['ids'],ids) and np.array_equal(old['classification_logits'],out['logits'].cpu()),'s=1 classification replay differs')
                need(np.array_equal(old['z'],out['z'].cpu()) and np.array_equal(old['decoder_logits'],out['decoder_logits'].cpu()),'s=1 original A1 forward differs')
            del out,x,batch
    install_scale(model)
    pooled,_=run_probe(root/'smoke','A1_best_scaled',model,tokenizer,data,selection)
    old_loss=csvrows(S/'03_losses_and_gradients.csv');new_loss=csvrows(root/'smoke/geometry/A1_best_scaled/losses.csv')
    old_grad=csvrows(S/'03_gradient_summary.csv');new_grad=csvrows(root/'smoke/geometry/A1_best_scaled/gradients.csv')
    for r in new_loss:
        old=next(v for v in old_loss if v['cohort']==r['cohort'] and float(v['scale'])==.03)
        for k in ['bce','auxiliary','shared_aux_to_BCE_gradient_norm','shared_gradient_cosine']:
            need(np.isclose(float(r[k]),float(old[k]),rtol=1e-6,atol=1e-8),'s=.03 old loss/gradient replay differs: '+k)
    for r in new_grad:
        old=next(v for v in old_grad if v['cohort']==r['cohort'] and v['module']==r['module'] and float(v['scale'])==.03)
        for k in ['bce_norm','weighted_aux_norm','cosine']:
            if r[k]!='':need(np.isclose(float(r[k]),float(old[k]),rtol=1e-6,atol=1e-9),'s=.03 module gradient replay differs')
    need(pooled['first_clip_fraction']==pooled['later_clip_fraction']==0,'Prior s=.03 geometric diagnosis not reproduced')
    for cohort in selection['cohorts']:
        with np.load(S/'cache'/f'{cohort}_s0p03.npz',allow_pickle=False) as old,np.load(root/'smoke/geometry/A1_best_scaled'/(cohort+'.npz'),allow_pickle=False) as new:
            need(np.array_equal(old['classification_logits'],new['logits']) and np.array_equal(old['z'],new['z']) and np.array_equal(old['decoder_logits'],new['decoder_logits']),'s=.03 exact saved-vector replay differs')
    write(root/'smoke/DIAGNOSTIC_REPLAY.json',dict(s1_original_forward_exact=True,s003_vectors_exact=True,
        loss_and_gradients_within_predeclared_tolerance=True,first_and_later_clips_zero=True,
        historical_checkpoint_used_for_read_only_diagnosis_only=True,formal_training_from_this_checkpoint=False))
    del model;release()

def smoke(root):
    from transformers import get_linear_schedule_with_warmup
    root=run_path(root);need(sys.flags.utf8_mode==1,'Use -X utf8')
    need(not (root/'plan.json').exists(),'Do not repeat smoke into an old run')
    stage(root,'frozen_inputs_protocol_environment_audit')
    source=source_hashes();historical_audit();env=runtime();data,manifest=dataset()
    protected=protected_inventory();write(root/'protected_before.json',protected)
    write(root/'source_manifest.json',source);write(root/'config.json',CONFIG);save_audit(root,manifest)
    snapshot=Path(read(A1/'plan.json')['snapshot'])
    need(a0.snapshot_inventory(snapshot)==read(A1/'plan.json')['snapshot_files_sha256'],'Raw snapshot changed')
    selection=read(S/'FIXED_TRAIN_COHORTS.json');write(root/'FIXED_TRAIN_COHORTS.json',selection)
    historical_validation(root,manifest['names27'])
    shutil.copytree(HERE,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(HERE/'IMPLEMENTATION_AUDIT.md',root/'EXP6C1S_TRAIN_IMPLEMENTATION_AUDIT.md')
    write(root/'plan.json',dict(config=CONFIG,source_sha256=source,runtime=env,snapshot=str(snapshot),
        snapshot_files_sha256=a0.snapshot_inventory(snapshot),created=now(),
        scope='s=.03 seed42 4epochs Validation-only; no new Test; stop after pilot',smoke_timeout_minutes=60,training_timeout_hours=8))
    directory=root/'smoke';directory.mkdir(exist_ok=False)
    stage(root,'same_raw_A0_A1_encoder_classifier_HNN_decoder_initialization')
    parity=initial_parity(snapshot,manifest['names27'],data,root)
    old=read(A1/'INITIALIZATION_PARITY.json')
    for k in ['shared_state_sha256','hyperbolic_initial_state_sha256','initial_train_logits_sha256']:
        need(parity[k]==old[k],'Fresh initialization differs from frozen A1: '+k)
    replay_diagnosis(root,snapshot,manifest['names27'],data,selection)
    seed_everything();model,tokenizer,_,_=unscaled_build(snapshot,manifest['names27']);model.to('cuda')
    first=next(iter(loader(data['train'],tokenizer,shuffle=True)));first_ids=first.pop('ids');x={k:v.to('cuda') for k,v in first.items()}
    need(first_ids==read(A1/'epoch1_first_batch.json')['ids'],'Fresh train order differs')
    scaled_initial_parity(model,x,root);del x,first
    np.save(root/'fixed_emotion_embeddings.npy',model.emotion_embeddings.detach().cpu().numpy())
    need(np.array_equal(np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False),np.load(A1/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Fixed label vectors changed')
    neutral=manifest['neutral_original_index']
    cases={'single_positive':[r for r in data['train'] if len(r['labels'])==1][:16],
        'multi_positive':[r for r in data['train'] if len(r['labels'])>1][:16],
        'mixed_neutral_retained':[r for r in data['train'] if neutral in r['original_labels']][:16]}
    reports={name:formula_and_gradient_case(model,Collator(tokenizer)(records)) for name,records in cases.items()}
    write(directory/'FORMULA_AND_GRADIENT_CASES.json',reports)
    initial,_=run_probe(directory,'fresh_scaled_initialization',model,tokenizer,data,selection)
    need(initial['first_clip_fraction']==0,'Scaled fresh input still clips at first mapping')
    del model;release()
    stage(root,'two_smoke_updates_and_smoke_Validation_reload')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.to('cuda').train()
    need(state_hash(model.control.state_dict())==parity['shared_state_sha256'] and state_hash(model.hyper.state_dict())==parity['hyperbolic_initial_state_sha256'],'Scaled smoke initialization drift')
    optimizer,groups=parameter_groups(model);scheduler=get_linear_schedule_with_warmup(optimizer,765,7648)
    initial_hyp=state_hash(model.hyper.state_dict());steps=[];iterations=iter(loader(data['train'],tokenizer,shuffle=True))
    for i in range(2):
        lr=optimizer.param_groups[0]['lr'];step=train_step(model,next(iterations),optimizer,scheduler,detailed=True)
        steps.append(dict(step,lr_used=lr,lr_after=optimizer.param_groups[0]['lr']))
    need(steps[0]['lr_used']==0 and steps[1]['lr_used']>0 and state_hash(model.hyper.state_dict())!=initial_hyp,'Smoke optimizer did not update')
    before=evaluate(model,tokenizer,data,smoke=True);torch.save(model.state_dict(),directory/'smoke_weights.pt')
    del model,optimizer,scheduler,iterations;release()
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27'])
    model.load_state_dict(torch.load(directory/'smoke_weights.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    after=evaluate(model,tokenizer,data,smoke=True);reload=compare_reload(before,after,manifest['names27'])
    need(np.array_equal(before[0],after[0]),'Smoke reload logits not exactly equal')
    del model;release()
    audit_unchanged(root)
    write(directory/'SMOKE_REPORT.json',dict(status='PASS',new_test_calls=0,formal_training_started=False,
        new_training_initialization='fresh raw snapshot; never A1 best or smoke checkpoint',
        initialization=parity,formula_gradient_cases=reports,two_smoke_optimizer_steps=steps,reload=reload,
        fresh_geometry=initial,completed=now()))
    write(directory/'SMOKE_VERIFIED.json',dict(status='EXP6C1S_SMOKE_PASS',artifacts_sha256=smoke_hashes(directory),
        plan_sha256=sha(root/'plan.json'),source_sha256=source,initialization_sha256=sha(root/'INITIALIZATION_PARITY.json'),
        scaled_initialization_sha256=sha(root/'SCALED_INITIALIZATION_PARITY.json')))
    verify_smoke(root);stage(root,'smoke_pass_seed42_formal_training_authorized_by_prompt')

def assert_plan(root):
    verify_smoke(root);need((root/'smoke_exit_code.txt').read_text(encoding='utf-8').strip()=='0','Smoke process did not exit zero')
    plan=read(root/'plan.json');env=runtime()
    for k in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[k]==plan['runtime'][k],'Runtime changed since smoke')
    data,manifest=dataset();need(manifest==read(root/'dataset_manifest.json'),'Dataset changed since smoke')
    audit_unchanged(root)
    snapshot=Path(plan['snapshot']);need(a0.snapshot_inventory(snapshot)==plan['snapshot_files_sha256'],'Snapshot changed')
    return data,manifest,plan,snapshot

def train(root):
    from transformers import get_linear_schedule_with_warmup,AutoTokenizer
    root=run_path(root);data,manifest,plan,snapshot=assert_plan(root)
    need(not (root/'FAILURE.json').exists(),'Prior failure; no automatic retry')
    with (root/'FORMAL_TRAINING_CLAIM.json').open('x',encoding='utf-8') as f:
        json.dump(dict(seed=42,scale=.03,epochs=4,new_test=False,no_automatic_retry=True,started=now()),f)
    env=dict(plan['runtime'],status='TRAINING',start_time=now(),completed_epochs=0,completed_optimizer_steps=0,formal_test_inference_calls=0)
    write(root/'environment.json',env);stage(root,'fresh_seed42_scaled_formal_training')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);parity=read(root/'INITIALIZATION_PARITY.json')
    need(state_hash(model.control.state_dict())==parity['shared_state_sha256'] and state_hash(model.hyper.state_dict())==parity['hyperbolic_initial_state_sha256'],'Formal initialization mismatch')
    model.to('cuda');optimizer,groups=parameter_groups(model)
    training=loader(data['train'],tokenizer,shuffle=True);validation=loader(data['validation'],tokenizer)
    need(len(training)*4==7648 and math.ceil(len(training)*4*.1)==765,'Wrong steps/warmup')
    scheduler=get_linear_schedule_with_warmup(optimizer,765,7648)
    write(root/'optimizer_protocol.json',dict(groups=groups,total_steps=7648,warmup_steps=765,max_grad_norm=1.,
        optimizer='frozen A1 parameter_groups',train_step='frozen a1model.train_step; unchanged normalization and call order'))
    checkpoint=root/'best_checkpoint';checkpoint.mkdir(exist_ok=False)
    selection=read(root/'FIXED_TRAIN_COHORTS.json');run_probe(root,'initial',model,tokenizer,data,selection)
    history=[];best=-1.;global_step=0
    fields=['epoch','batch','step','batch_size','loss','bce','auxiliary','weighted_auxiliary','weighted_auxiliary_to_bce',
        'absolute_weighted_auxiliary_to_bce','positive_ce_mean','positive_dot_mean','max_ball_norm','unclipped_grad_norm','lr_used','lr_after']
    with (root/'batch_log.csv').open('x',encoding='utf-8',newline='') as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,5):
            stage(root,f'training_epoch_{epoch}');model.train();optimizer.zero_grad(set_to_none=True)
            sums={};count=0;order=[];norms=[];maximum_ball=0.
            for index,batch in enumerate(training,1):
                lr=optimizer.param_groups[0]['lr'];r=train_step(model,batch,optimizer,scheduler,detailed=index==1)
                if index==1:
                    need(r['ids']==read(A1/f'epoch{epoch}_first_batch.json')['ids'],'Epoch-first train order differs from A1')
                    write(root/f'epoch{epoch}_first_batch.json',r)
                global_step+=1;size=r['batch_size'];count+=size;order.extend(r['ids']);norms.append(r['unclipped_grad_norm'])
                maximum_ball=max(maximum_ball,r['max_ball_norm'])
                for k in ['loss','bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean']:sums[k]=sums.get(k,0.)+r[k]*size
                writer.writerow(dict(epoch=epoch,batch=index,step=global_step,lr_used=lr,lr_after=optimizer.param_groups[0]['lr'],**{k:r[k] for k in fields if k in r}))
                if index%50==0 or index==len(training):
                    f.flush();print(f'A1-S seed42 epoch={epoch}/4 batch={index}/{len(training)} step={global_step}/7648 BCE={r["bce"]:.6f} aux={r["auxiliary"]:.6f} total={r["loss"]:.6f}',flush=True)
            need(count==30587 and len(set(order))==30587 and order==read(A1/f'train_order_epoch{epoch}.json'),'Full training order differs from frozen A1')
            write(root/f'train_order_epoch{epoch}.json',order)
            val=predict(model,validation,f'epoch{epoch} Validation')
            need(np.array_equal(val[2],[r['id'] for r in data['validation']]) and np.array_equal(val[1],targets(data['validation'])),'Validation IDs/gold differ')
            metrics=extended_score(val[1],probabilities(val[0]),manifest['names27'])[0]
            entry=dict(epoch=epoch,global_step=global_step,training_examples=count,
                **{'training_'+k:v/count for k,v in sums.items()},max_training_ball_norm=maximum_ball,
                validation_loss=val[3],validation_bce=val[4]['bce'],validation_auxiliary=val[4]['auxiliary'],
                validation_weighted_auxiliary=val[4]['weighted_auxiliary'],validation_macro_f1=metrics['macro_f1'],
                validation_micro_f1=metrics['micro_f1'],validation_rare_f1=metrics['rare_f1'],validation_macro_average_precision=metrics['macro_average_precision'],
                learning_rate=optimizer.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(np.max(norms)),
                gradient_clip_fraction=float(np.mean(np.array(norms)>1)))
            history.append(entry);table(root/'training_log.csv',history)
            write(root/f'epoch{epoch}_validation_metrics.json',metrics);write(root/f'epoch{epoch}_validation_loss_diagnostics.json',val[4])
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];torch.save(model.state_dict(),checkpoint/'model.pt')
                tokenizer.save_pretrained(checkpoint/'tokenizer');model.control.encoder.config.save_pretrained(checkpoint/'encoder_config')
                save_validation(root,'validation_selected',*val[:3],manifest['names27'])
                write(root/'validation_selected_loss_diagnostics.json',val[4])
                write(root/'checkpoint_selection.json',dict(epoch=epoch,global_step=global_step,validation_macro_f1=best,threshold=.30,
                    selection_split='validation',tie_break='earliest via strict greater-than',checkpoint_sha256=sha(checkpoint/'model.pt')))
            run_probe(root,f'epoch{epoch}',model,tokenizer,data,selection)
            env.update(completed_epochs=epoch,completed_optimizer_steps=global_step);write(root/'environment.json',env)
            print('EPOCH_RESULT',entry,flush=True)
    need(global_step==7648 and scheduler.get_last_lr()[0]==0,'Incomplete scheduled budget')
    del model,optimizer,scheduler,training,validation,val;release()
    stage(root,'full_Validation_reload_no_Test')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27'])
    model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer'),local_files_only=True,use_fast=True)
    reloaded=evaluate(model,tokenizer,data)
    with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as z:previous=z['logits'].copy(),z['targets'].copy(),z['ids'].copy()
    reload=compare_reload(previous,reloaded,manifest['names27']);need(np.array_equal(previous[0],reloaded[0]),'Full reload logits must be exactly equal')
    reload['logits_exact_equal']=True;write(root/'checkpoint_reload_verification.json',reload)
    for k,v in reloaded[4].items():need(np.isclose(v,read(root/'validation_selected_loss_diagnostics.json')[k],rtol=1e-6,atol=1e-6),'Reload loss diagnostics differ')
    save_validation(root,'validation',*reloaded[:3],manifest['names27']);write(root/'validation_loss_diagnostics.json',reloaded[4])
    need(np.array_equal(model.emotion_embeddings.detach().cpu().numpy(),np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Label vectors updated')
    run_probe(root,'best_reload',model,tokenizer,data,selection)
    del model;release()
    stage(root,'frozen_inputs_recheck_and_saved_Validation_verification');audit_unchanged(root)
    write(root/'FROZEN_POSTCHECK.json',dict(status='PASS',all_historical_files_rehashed_unchanged=True,
        protected_file_count=len(read(root/'protected_before.json')),protected_manifest_sha256=sha(root/'protected_before.json'),completed=now()))
    env.update(status='COMPLETE',end_time=now(),formal_test_inference_calls=0,old_research_files_unchanged=True);write(root/'environment.json',env)
    write_reports(root);verify(root,sealed=False);stage(root,'COMPLETE')
    write(root/'SUCCESS.json',dict(status=STATUS,seed=42,scale=.03,auxiliary_coefficient=.1,epochs=4,new_test_inference_calls=0,
        artifacts_sha256=artifact_hashes(root),source_sha256=source_hashes(),completed=now()))
    verify(root);print('TRAIN_AND_VALIDATION_RELOAD_COMPLETE_NO_TEST',flush=True)

def verify(root,sealed=True):
    root=run_path(root)
    if sealed:
        proof=read(root/'SUCCESS.json');need(proof['status']==STATUS and proof['artifacts_sha256']==artifact_hashes(root),'Pilot success seal differs')
    source_hashes();need(read(root/'config.json')==CONFIG,'Config differs')
    env=read(root/'environment.json')
    need(env['completed_epochs']==4 and env['completed_optimizer_steps']==7648 and env['formal_test_inference_calls']==0,'Pilot budget/Test count differs')
    need(not list(root.glob('test*')) and not (root/'FORMAL_TEST_CLAIM.json').exists(),'Unexpected Test output')
    history=csvrows(root/'training_log.csv');need([int(r['global_step']) for r in history]==[1912,3824,5736,7648],'Wrong epoch history')
    selected=max(history,key=lambda r:float(r['validation_macro_f1']));choice=read(root/'checkpoint_selection.json')
    need(int(selected['epoch'])==choice['epoch'] and float(selected['validation_macro_f1'])==choice['validation_macro_f1'],'Wrong Validation checkpoint/earliest tie')
    need(sha(root/'best_checkpoint/model.pt')==choice['checkpoint_sha256'],'Checkpoint bytes differ')
    opt=read(root/'optimizer_protocol.json');need(opt['total_steps']==7648 and opt['warmup_steps']==765 and opt['max_grad_norm']==1,'Wrong optimizer protocol')
    batches=csvrows(root/'batch_log.csv');need([int(r['step']) for r in batches]==list(range(1,7649)),'Incomplete batch log')
    need(float(batches[-1]['lr_after'])==0,'Scheduler incomplete')
    for r in batches:
        need(all(np.isfinite(float(v)) for v in r.values()),'Nonfinite batch diagnostics')
        need(float(r['max_ball_norm'])<1 and abs(float(r['loss'])-float(r['bce'])-.1*float(r['auxiliary']))<2e-6,'Bad ball/loss composition')
    manifest=read(root/'dataset_manifest.json');names=manifest['names27']
    data,current_manifest=dataset();need(current_manifest==manifest,'Canonical dataset differs during CPU verification')
    with np.load(root/'validation_predictions.npz',allow_pickle=False) as z:
        need(z['label_names'].tolist()==names and np.array_equal(z['ids'],manifest['splits']['validation']['retained_ids']),'Validation labels/IDs differ')
        need(np.array_equal(z['targets'],targets(data['validation'])),'Saved Validation gold differs from canonical TSV')
        probs=probabilities(z['logits']);metrics,rows,_,pred=extended_score(z['targets'],probs,names)
        need(np.array_equal(probs,z['probabilities']) and np.array_equal(pred,z['predictions']),'Saved predictions differ')
        need(metrics==read(root/'validation_metrics.json'),'Saved metric/AP recomputation differs')
        need(metrics['average_precision_defined_labels']==27,'All27 Validation AP values must be defined')
        recorded=csvrows(root/'validation_per_label.csv')
        need(len(recorded)==len(rows)==27,'Need27 labels')
        for a,b in zip(recorded,rows):
            for k,v in b.items():need(a[k]=='' if v is None else a[k]==v if isinstance(v,str) else np.isclose(float(a[k]),v,rtol=0,atol=1e-12),'Per-label saved metrics differ')
        with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as previous:
            for k in ['ids','targets','logits','predictions']:need(np.array_equal(previous[k],z[k]),'Saved/reloaded Validation arrays differ')
    need(metrics['macro_f1']==choice['validation_macro_f1'],'Final metrics do not match selection')
    reload=read(root/'checkpoint_reload_verification.json')
    need(reload['num_examples']==3834 and reload['logits_exact_equal'] and reload['max_abs_logit_error']==0,'Missing exact full Validation reload')
    for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        p=read(root/'geometry'/tag/'NO_UPDATE_PROOF.json')
        need(p['state_before']==p['state_after'] and p['optimizer_steps']==p['test_calls']==0,'Geometry probe updated state')
        need(all(p[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode changed')
    for epoch in range(1,5):
        need(read(root/f'train_order_epoch{epoch}.json')==read(A1/f'train_order_epoch{epoch}.json'),'Training order differs from A1')
    post=read(root/'FROZEN_POSTCHECK.json');need(post['all_historical_files_rehashed_unchanged'] and post['protected_manifest_sha256']==sha(root/'protected_before.json'),'Missing historical freeze recheck')
    print('EXP6C1S_SAVED_ARTIFACT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train','verify','verify-smoke']);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    try:
        {'smoke':smoke,'train':train,'verify':verify,'verify-smoke':verify_smoke}[a.phase](a.run_root)
    except BaseException as e:
        if a.phase in ['smoke','train'] and a.run_root.exists() and not (a.run_root/'SUCCESS.json').exists() and not (a.run_root/'FAILURE.json').exists():
            write(a.run_root/'FAILURE.json',dict(phase=a.phase,error=repr(e),traceback=traceback.format_exc(),time=now(),new_test=False))
        raise
