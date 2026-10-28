"""One fresh seed42 N1 pilot. Train/Validation only; no Test prediction path."""
import argparse
import gc
import math
import shutil
import sys
import traceback
import numpy as np
import torch
from ncommon import *
from nmodel import build,control_build,install_n1,values,vectors,describe,train_step
from nprobe import run_probe,formula_check
from probes import isolated_eval
from a1model import parameter_groups
from a1run import initial_parity
from model27 import seed_everything
from data27 import Collator,loader,targets,save_audit
from run import compare_reload

def release():gc.collect();torch.cuda.empty_cache()
def stage(root,name):write(root/'state.json',dict(stage=name,time=now()));print('STAGE',name,flush=True)

@torch.no_grad()
def predict_validation(model,tokenizer,records,description):
    need(len(records) in [64,3834],'Only fixed smoke64/full3834 Validation allowed')
    model.eval();logits=[];gold=[];ids=[];totals={};count=0;maxradius=0.
    for index,batch in enumerate(loader(records,tokenizer),1):
        record_ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x);v=values(out);size=len(record_ids)
        for k,value in v.items():totals[k]=totals.get(k,0.)+value*size
        count+=size;maxradius=max(maxradius,v['max_ball_norm']);ids.extend(record_ids)
        logits.append(out['logits'].cpu().numpy());gold.append(x['labels'].cpu().numpy())
        if index%25==0:print(description,'batches',index,'examples',count,flush=True)
    need(all(not m.training for m in model.modules()),'Validation dropout active')
    stats={k:v/count for k,v in totals.items()};stats['max_ball_norm']=maxradius
    stats['weighted_bce_to_bce']=stats['weighted_bce']/stats['bce']
    need(ids==[r['id'] for r in records] and np.array_equal(np.concatenate(gold),targets(records)),'Noncanonical Validation IDs/gold')
    return np.concatenate(logits),np.concatenate(gold),np.array(ids),stats['loss'],stats

def smoke_hashes(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='SMOKE_VERIFIED.json'}

def verify_smoke(root):
    p=read(root/'smoke/SMOKE_VERIFIED.json')
    need(p['status']=='EXP6D1_SMOKE_PASS' and p['artifacts_sha256']==smoke_hashes(root/'smoke'),'Smoke failed/changed')
    need(p['plan_sha256']==sha(root/'plan.json') and p['source_manifest']==check_sources(),'Smoke source/plan changed')
    need(read(root/'config.json')==CONFIG,'Protocol changed')
    for n in ['INITIALIZATION_PARITY.json','N1_INITIALIZATION_PARITY.json']:
        need(p['initialization_sha256'][n]==sha(root/n),'Initialization proof changed')
    print('EXP6D1_SAVED_SMOKE_VERIFICATION_PASS',flush=True)

def initialization(root,snapshot,names,data):
    parity=initial_parity(snapshot,names,data,root)
    old=read(CONTROL/'INITIALIZATION_PARITY.json')
    for k in ['shared_state_sha256','hyperbolic_initial_state_sha256','initial_train_logits_sha256']:
        need(parity[k]==old[k],'Fresh shared/HNN initialization differs from A1-S: '+k)
    seed_everything();model,tokenizer,_,_=control_build(snapshot,names);model.to('cuda')
    batch=next(iter(loader(data['train'],tokenizer,shuffle=True)));ids=batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
    need(ids==parity['first_canonical_training_batch_ids'],'Initial batch differs')
    state=state_hash(model.state_dict());cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        def reset():torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        reset();model.train();oldtrain=model(**x);after_cpu=torch.get_rng_state().clone();after_gpu=torch.cuda.get_rng_state().clone()
        reset();model.eval();oldeval=model(**x)
        reset();install_n1(model);need(torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'N1 installation changed RNG')
        model.train();newtrain=model(**x)
        need(torch.equal(after_cpu,torch.get_rng_state()) and torch.equal(after_gpu,torch.cuda.get_rng_state()),'N1 changed train-forward RNG consumption')
        reset();model.eval();neweval=model(**x)
        for oldout,newout in [(oldtrain,newtrain),(oldeval,neweval)]:
            for k in ['logits','z','decoder_logits']:need(torch.equal(oldout[k],newout[k]),'N1 changed model forward: '+k)
            need(torch.equal(oldout['components']['auxiliary'],newout['components']['auxiliary']),'Auxiliary changed')
    need(state_hash(model.state_dict())==state,'N1 changed state keys/parameters/buffers')
    write(root/'N1_INITIALIZATION_PARITY.json',dict(state_before=state,state_after=state,new_parameters=0,
        shared_and_hyper_initialization_match_A1S=True,train_and_eval_classification_logits_exact=True,
        train_and_eval_z_decoder_exact=True,auxiliary_exact=True,initial_forward_rng_consumption_equal=True,
        trained_A1S_weights_used_for_initialization=False,fresh_pretrained_start=True))
    del model,x,oldtrain,oldeval,newtrain,neweval;release()
    return parity

def smoke(root):
    from transformers import get_linear_schedule_with_warmup
    root=run_path(root);need(sys.flags.utf8_mode==1,'Use -X utf8');need(not (root/'plan.json').exists(),'No reuse of smoke run')
    stage(root,'source_protocol_frozen_seals_and_runtime')
    source=check_sources();check_old_seals();env=runtime();data,manifest=dataset();snapshot=Path(read(CONTROL/'plan.json')['snapshot'])
    need(a0.snapshot_inventory(snapshot)==read(CONTROL/'plan.json')['snapshot_files_sha256'],'Raw snapshot changed')
    write(root/'protected_before.json',protected_inventory());write(root/'source_manifest.json',source);write(root/'config.json',CONFIG);save_audit(root,manifest)
    selection=read(CONTROL/'FIXED_TRAIN_COHORTS.json');need(selection==read(D/'FIXED_TRAIN_COHORTS.json'),'Fixed cohorts differ from Exp6D')
    by_id={r['id']:r for r in data['train']}
    for records in selection['cohorts'].values():
        for r in records:need([manifest['names27'][k] for k in by_id[r['id']]['labels']]==r['labels'],'Canonical train probe label mapping differs')
    write(root/'FIXED_TRAIN_COHORTS.json',selection);historical_validation(root,manifest['names27'])
    shutil.copytree(HERE,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(HERE/'IMPLEMENTATION_AUDIT.md',root/'EXP6D1_IMPLEMENTATION_AUDIT.md')
    write(root/'plan.json',dict(config=CONFIG,runtime=env,source_manifest=source,snapshot=str(snapshot),
        snapshot_files_sha256=a0.snapshot_inventory(snapshot),control=str(CONTROL),completed_audit=str(D),created=now(),
        scope='N1 seed42, 4epochs, Validation only; no rawF, no extra seeds, no new Test',smoke_timeout_minutes=90,training_timeout_hours=8))
    (root/'smoke').mkdir()
    stage(root,'fresh_raw_snapshot_shared_classifier_HNN_decoder_initialization')
    parity=initialization(root,snapshot,manifest['names27'],data)
    stage(root,'completed_Exp6D_A1S_best_no_update_N1_loss_and_parameter_gradient_replay')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27'])
    model.load_state_dict(torch.load(CONTROL/'best_checkpoint/model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    need(state_hash(model.state_dict())==next(p for p in read(D/'NO_UPDATE_PROOF.json')['models'] if p['model']=='A1S')['state_before'],
         'Loaded replay state differs from completed Exp6D')
    run_probe(root/'smoke','A1S_best_replay',model,tokenizer,data,selection,replay=True)
    del model;release()
    stage(root,'fresh_N1_fixed_train_probes_and_short_batch_formula')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.to('cuda')
    need(state_hash(model.control.state_dict())==parity['shared_state_sha256'] and state_hash(model.hyper.state_dict())==parity['hyperbolic_initial_state_sha256'],'Fresh smoke state mismatch')
    np.save(root/'fixed_emotion_embeddings.npy',model.emotion_embeddings.cpu().numpy())
    need(np.array_equal(np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False),np.load(CONTROL/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Fixed label vectors changed')
    run_probe(root/'smoke','fresh_initial',model,tokenizer,data,selection)
    cases={'batch1':data['train'][:1],'single_positive':[r for r in data['train'] if len(r['labels'])==1][:16],
        'multi_positive':[r for r in data['train'] if len(r['labels'])>1][:16],'short_batch11':data['train'][-11:]}
    reports={};ref=audit_reference()
    with isolated_eval(model):
        for tag,records in cases.items():
            b=Collator(tokenizer)(records);ids=b.pop('ids');x={k:v.to('cuda') for k,v in b.items()};out=model(**x)
            reports[tag]=dict(formula_check(out,x['labels'],model.emotion_embeddings,ref),ids=ids,batch_size=len(ids))
            if len(ids)==1:need(torch.allclose(out['components']['ordinary_bce'],out['components']['weighted_bce'],atol=1e-7,rtol=1e-6),'B=1 must normalize to ordinary BCE')
            del out,x,b
    write(root/'smoke/SHORT_AND_MULTIPOSITIVE_FORMULA.json',reports)
    del model;release()
    stage(root,'two_discarded_smoke_updates_and_64_Validation_exact_reload')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.to('cuda').train()
    optimizer,groups=parameter_groups(model);scheduler=get_linear_schedule_with_warmup(optimizer,765,7648)
    iterator=iter(loader(data['train'],tokenizer,shuffle=True));steps=[]
    for i in range(2):
        lr=optimizer.param_groups[0]['lr'];r,v=train_step(model,next(iterator),optimizer,scheduler)
        steps.append(dict(r,lr_used=lr,lr_after=optimizer.param_groups[0]['lr'],**describe(v)))
    need(steps[0]['lr_used']==0 and steps[1]['lr_used']>0,'Smoke scheduler mismatch')
    before=predict_validation(model,tokenizer,data['validation'][:64],'Smoke Validation')
    torch.save(model.state_dict(),root/'smoke/weights.pt');del model,optimizer,scheduler,iterator;release()
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.load_state_dict(torch.load(root/'smoke/weights.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    after=predict_validation(model,tokenizer,data['validation'][:64],'Reload Smoke Validation');reload=compare_reload(before,after,manifest['names27'])
    need(np.array_equal(before[0],after[0]),'Smoke reload logits not exact')
    del model;release();audit_unchanged(root)
    write(root/'smoke/SMOKE_REPORT.json',dict(status='PASS',formal_training_started=False,formal_test_calls=0,
        discarded_optimizer_steps=steps,reload=reload,initialization=parity,short_batch_and_multilabel=reports,completed=now()))
    write(root/'smoke/SMOKE_VERIFIED.json',dict(status='EXP6D1_SMOKE_PASS',artifacts_sha256=smoke_hashes(root/'smoke'),
        plan_sha256=sha(root/'plan.json'),source_manifest=source,
        initialization_sha256={n:sha(root/n) for n in ['INITIALIZATION_PARITY.json','N1_INITIALIZATION_PARITY.json']}))
    verify_smoke(root);stage(root,'smoke_pass_formal_training_authorized_by_user_prompt')

def train(root):
    from transformers import get_linear_schedule_with_warmup,AutoTokenizer
    root=run_path(root);verify_smoke(root)
    need((root/'smoke_exit_code.txt').read_text(encoding='utf-8').strip()=='0','Smoke process failed')
    need(not (root/'FAILURE.json').exists(),'Prior failure; no automatic retry')
    plan=read(root/'plan.json');env=runtime()
    for k in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[k]==plan['runtime'][k],'Runtime changed')
    audit_unchanged(root);data,manifest=dataset();snapshot=Path(plan['snapshot'])
    need(a0.snapshot_inventory(snapshot)==plan['snapshot_files_sha256'],'Snapshot changed since smoke')
    with (root/'FORMAL_TRAINING_CLAIM.json').open('x',encoding='utf-8') as f:
        import json
        json.dump(dict(seed=42,epochs=4,normalization='N1',input_scale=.03,new_test=False,started=now(),no_automatic_retry=True),f)
    env.update(status='TRAINING',start_time=now(),completed_epochs=0,completed_optimizer_steps=0,formal_test_inference_calls=0)
    write(root/'environment.json',env);stage(root,'fresh_seed42_N1_formal_training')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.to('cuda');parity=read(root/'INITIALIZATION_PARITY.json')
    need(state_hash(model.control.state_dict())==parity['shared_state_sha256'] and state_hash(model.hyper.state_dict())==parity['hyperbolic_initial_state_sha256'],'Formal run not same fresh initialization')
    optimizer,groups=parameter_groups(model);training=loader(data['train'],tokenizer,shuffle=True)
    need(len(training)==1912 and len(training)*4==7648 and math.ceil(len(training)*4*.1)==765,'Schedule mismatch')
    scheduler=get_linear_schedule_with_warmup(optimizer,765,7648)
    write(root/'optimizer_protocol.json',dict(groups=groups,optimizer='frozen A1 parameter_groups AdamW',learning_rate=5e-5,total_steps=7648,warmup_steps=765,max_grad_norm=1.,
        backward='loss*batch_size; backward; divide grads by batch_size; clip1; optimizer.step; scheduler.step; zero_grad',normalization='actual current batch, including last11'))
    checkpoint=root/'best_checkpoint';checkpoint.mkdir();selection=read(root/'FIXED_TRAIN_COHORTS.json')
    run_probe(root,'initial',model,tokenizer,data,selection)
    best=-1.;history=[];weight_stats=[];step=0
    fields=['epoch','batch','step','batch_size','loss','bce','weighted_bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean',
        'weighted_bce_to_bce','max_ball_norm','unclipped_grad_norm','lr_used','lr_after','distance_mean','distance_std','distance_median','distance_p95','distance_max',
        'weight_mean','weight_std','weight_min','weight_p95','weight_max','weight_vs_bce_correlation']
    import csv
    with (root/'batch_log.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,5):
            stage(root,f'training_epoch_{epoch}');model.train();optimizer.zero_grad(set_to_none=True)
            sums={};count=0;order=[];norms=[];vectors_epoch={k:[] for k in ['distance','weight','bce','positive_count']}
            for index,batch in enumerate(training,1):
                lr=optimizer.param_groups[0]['lr'];r,v=train_step(model,batch,optimizer,scheduler);step+=1;size=r['batch_size'];count+=size
                if index==1:
                    need(r['ids']==read(CONTROL/f'epoch{epoch}_first_batch.json')['ids'],'Epoch-first train IDs differ')
                    write(root/f'epoch{epoch}_first_batch.json',r)
                if index==1912:need(size==11 and abs(float(v['weight'].mean())-1)<2e-6,'Last11 normalization incorrect')
                order.extend(r['ids']);norms.append(r['unclipped_grad_norm'])
                for k in ['loss','bce','weighted_bce','auxiliary','weighted_auxiliary']:sums[k]=sums.get(k,0.)+r[k]*size
                for k in vectors_epoch:vectors_epoch[k].append(v[k])
                logged=dict(epoch=epoch,batch=index,step=step,lr_used=lr,lr_after=optimizer.param_groups[0]['lr'],**{k:val for k,val in r.items() if k!='ids'})
                logged.update(describe(v))
                writer.writerow(logged)
                if index%50==0 or index==1912:
                    f.flush();print(f'N1 seed42 epoch={epoch}/4 batch={index}/1912 step={step}/7648 BCE={r["bce"]:.6f} N1={r["weighted_bce"]:.6f} aux={r["auxiliary"]:.6f}',flush=True)
            need(count==30587 and len(set(order))==30587 and order==read(CONTROL/f'train_order_epoch{epoch}.json'),'Full train order differs from A1-S')
            write(root/f'train_order_epoch{epoch}.json',order)
            vec={k:np.concatenate(v) for k,v in vectors_epoch.items()};np.savez_compressed(root/f'train_n1_epoch{epoch}.npz',ids=np.array(order),**vec)
            for group,mask in [('all',np.ones(count,dtype=bool)),('single_positive',vec['positive_count']==1),('multi_positive',vec['positive_count']>1)]:
                weight_stats.append(dict(epoch=epoch,group=group,examples=int(mask.sum()),**describe({k:v[mask] for k,v in vec.items()})))
            table(root/'EXP6D1_DISTANCE_WEIGHT_DIAGNOSTICS.csv',weight_stats)
            val=predict_validation(model,tokenizer,data['validation'],f'epoch{epoch} Validation')
            metrics=extended_score(val[1],probabilities(val[0]),manifest['names27'])[0]
            entry=dict(epoch=epoch,global_step=step,training_examples=count,**{'training_'+k:v/count for k,v in sums.items()},
                training_total_loss=sums['loss']/count,validation_macro_f1=metrics['macro_f1'],validation_micro_f1=metrics['micro_f1'],
                validation_rare_f1=metrics['rare_f1'],validation_macro_ap=metrics['macro_average_precision'],
                validation_loss=val[3],validation_bce=val[4]['bce'],validation_weighted_bce=val[4]['weighted_bce'],validation_auxiliary=val[4]['auxiliary'],
                learning_rate=optimizer.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(np.max(norms)),
                gradient_clip_fraction=float(np.mean(np.array(norms)>1)),**describe(vec))
            history.append(entry);table(root/'training_log.csv',history)
            write(root/f'epoch{epoch}_validation_metrics.json',metrics);write(root/f'epoch{epoch}_validation_loss_diagnostics.json',val[4])
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];torch.save(model.state_dict(),checkpoint/'model.pt');tokenizer.save_pretrained(checkpoint/'tokenizer');model.control.encoder.config.save_pretrained(checkpoint/'encoder_config')
                save_validation(root,'validation_selected',*val[:3],manifest['names27']);write(root/'validation_selected_loss_diagnostics.json',val[4])
                write(root/'checkpoint_selection.json',dict(epoch=epoch,global_step=step,validation_macro_f1=best,threshold=.30,selection_split='validation',tie_break='earliest strict greater-than',checkpoint_sha256=sha(checkpoint/'model.pt')))
            run_probe(root,f'epoch{epoch}',model,tokenizer,data,selection)
            env.update(completed_epochs=epoch,completed_optimizer_steps=step);write(root/'environment.json',env);print('EPOCH_RESULT',entry,flush=True)
    need(step==7648 and scheduler.get_last_lr()[0]==0,'Incomplete schedule')
    del model,optimizer,scheduler,training,val;release();stage(root,'full3834_Validation_reload_NO_Test')
    seed_everything();model,tokenizer,_,_=build(snapshot,manifest['names27']);model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer'),local_files_only=True,use_fast=True)
    after=predict_validation(model,tokenizer,data['validation'],'Reload Validation')
    with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as z:before=z['logits'].copy(),z['targets'].copy(),z['ids'].copy()
    reload=compare_reload(before,after,manifest['names27']);need(np.array_equal(before[0],after[0]),'Full3834 reload logits not exact');reload['logits_exact_equal']=True
    write(root/'checkpoint_reload_verification.json',reload)
    for k,v in after[4].items():need(np.isclose(v,read(root/'validation_selected_loss_diagnostics.json')[k],atol=1e-6,rtol=1e-6),'Reload loss differs')
    save_validation(root,'validation',*after[:3],manifest['names27']);write(root/'validation_loss_diagnostics.json',after[4])
    need(np.array_equal(model.emotion_embeddings.cpu(),np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Fixed labels updated')
    run_probe(root,'best_reload',model,tokenizer,data,selection);del model;release()
    stage(root,'historical_freeze_recheck_and_saved_data_verification');audit_unchanged(root)
    write(root/'FROZEN_POSTCHECK.json',dict(all_historical_files_rehashed_unchanged=True,protected_file_count=len(read(root/'protected_before.json')),protected_manifest_sha256=sha(root/'protected_before.json'),completed=now()))
    env.update(status='COMPLETE',end_time=now(),formal_test_inference_calls=0);write(root/'environment.json',env)
    from nreport import write_reports
    write_reports(root);verify(root,sealed=False);stage(root,'COMPLETE')
    write(root/'SUCCESS.json',dict(status=STATUS,seed=42,epochs=4,scale=.03,auxiliary_coefficient=.1,new_test_inference_calls=0,normalization='N1',artifacts_sha256=artifact_hashes(root),completed=now()))
    verify(root);print((root/'EXP6D1_TRAINING_SUMMARY.md').read_text(encoding='utf-8'),flush=True)

def verify(root,sealed=True):
    root=run_path(root)
    if sealed:
        proof=read(root/'SUCCESS.json');need(proof['status']==STATUS and proof['artifacts_sha256']==artifact_hashes(root),'Success seal differs')
    need(read(root/'config.json')==CONFIG and check_sources()==read(root/'source_manifest.json'),'Source/config differs')
    env=read(root/'environment.json');need(env['completed_epochs']==4 and env['completed_optimizer_steps']==7648 and env['formal_test_inference_calls']==0,'Budget/Test count differs')
    need(not list(root.glob('test*')) and not (root/'FORMAL_TEST_CLAIM.json').exists(),'Unexpected Test artifacts')
    data,manifest=dataset();need(manifest==read(root/'dataset_manifest.json'),'Dataset changed')
    names=manifest['names27'];history=csvrows(root/'training_log.csv');need([int(r['global_step']) for r in history]==[1912,3824,5736,7648],'Wrong epoch steps')
    best=max(history,key=lambda r:float(r['validation_macro_f1']));selection=read(root/'checkpoint_selection.json')
    need(int(best['epoch'])==selection['epoch'] and float(best['validation_macro_f1'])==selection['validation_macro_f1'],'Wrong checkpoint/tie')
    need(sha(root/'best_checkpoint/model.pt')==selection['checkpoint_sha256'],'Checkpoint bytes changed')
    opt=read(root/'optimizer_protocol.json');need(opt['total_steps']==7648 and opt['warmup_steps']==765 and opt['max_grad_norm']==1,'Optimizer mismatch')
    batches=csvrows(root/'batch_log.csv');need([int(r['step']) for r in batches]==list(range(1,7649)) and float(batches[-1]['lr_after'])==0,'Incomplete batch/scheduler log')
    for r in batches:
        need(all(v=='' or np.isfinite(float(v)) for v in r.values()),'Nonfinite batch log')
        need(float(r['max_ball_norm'])<1 and abs(float(r['weight_mean'])-1)<2e-6,'Invalid ball/weight')
        need(abs(float(r['loss'])-float(r['weighted_bce'])-.1*float(r['auxiliary']))<2e-6,'Wrong total loss')
    for epoch in range(1,5):
        need(read(root/f'train_order_epoch{epoch}.json')==read(CONTROL/f'train_order_epoch{epoch}.json'),'Train order changed')
        with np.load(root/f'train_n1_epoch{epoch}.npz',allow_pickle=False) as z:
            need(z['ids'].tolist()==read(root/f'train_order_epoch{epoch}.json') and len(z['ids'])==30587,'Incomplete per-sample diagnostics')
            # Recheck batch normalization including the final 11 using recorded raw distances.
            for start in range(0,len(z['ids']),16):
                d=z['distance'][start:start+16];w=z['weight'][start:start+16]
                need(np.allclose(w,d/d.mean(),atol=1e-7,rtol=1e-6),'Saved weights do not match current-batch N1')
            stats=describe({k:z[k] for k in ['distance','weight','bce','positive_count']})
            for k,v in stats.items():
                saved=history[epoch-1][k];need(saved=='' if v is None else np.isclose(float(saved),v,atol=1e-10,rtol=1e-10),'Epoch N1 distribution differs')
            for group,mask in [('all',np.ones(len(z['ids']),dtype=bool)),('single_positive',z['positive_count']==1),('multi_positive',z['positive_count']>1)]:
                records=csvrows(root/'EXP6D1_DISTANCE_WEIGHT_DIAGNOSTICS.csv')
                recorded=next(r for r in records if int(r['epoch'])==epoch and r['group']==group)
                need(int(recorded['examples'])==int(mask.sum()),'Single/multi diagnostic count differs')
                actual=describe({k:z[k][mask] for k in ['distance','weight','bce','positive_count']})
                for k,v in actual.items():need(recorded[k]=='' if v is None else np.isclose(float(recorded[k]),v,atol=1e-10,rtol=1e-10),'Single/multi distribution differs')
    with np.load(root/'validation_predictions.npz',allow_pickle=False) as z:
        need(z['label_names'].tolist()==names and z['ids'].tolist()==[r['id'] for r in data['validation']] and np.array_equal(z['targets'],targets(data['validation'])),'Validation label/ID/gold mismatch')
        probs=probabilities(z['logits']);metrics,rows,rare,pred=extended_score(z['targets'],probs,names)
        need(metrics==read(root/'validation_metrics.json') and np.array_equal(pred,z['predictions']) and np.array_equal(probs,z['probabilities']),'Metric/AP/probability mismatch')
        need(metrics['average_precision_defined_labels']==27 and metrics['macro_f1']==selection['validation_macro_f1'],'Wrong selected metric')
        recorded=csvrows(root/'validation_per_label.csv');need(len(recorded)==27,'Need27 per-label rows')
        for a,b in zip(recorded,rows):
            for k,v in b.items():need(a[k]=='' if v is None else a[k]==v if isinstance(v,str) else np.isclose(float(a[k]),v,atol=1e-12,rtol=0),'Per-label metrics differ')
        with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as old:
            for k in ['logits','ids','targets','predictions']:need(np.array_equal(old[k],z[k]),'Selected/reloaded arrays differ')
    reload=read(root/'checkpoint_reload_verification.json');need(reload['num_examples']==3834 and reload['logits_exact_equal'] and reload['max_abs_logit_error']==0,'Missing full exact reload')
    for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        p=read(root/'geometry'/tag/'NO_UPDATE_PROOF.json')
        need(p['state_before']==p['state_after'] and p['optimizer_steps']==p['test_calls']==0 and p['all_grads_none'],'Probe mutation')
        need(all(p[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode not restored')
    post=read(root/'FROZEN_POSTCHECK.json');need(post['all_historical_files_rehashed_unchanged'] and post['protected_manifest_sha256']==sha(root/'protected_before.json'),'Historical preservation proof absent')
    rare=read(root/'rare_label_metrics.json');need(rare['labels']==a0.RARE and rare['rare_f1']==metrics['rare_f1'],'Rare label definition/metric changed')
    for name in ['EXP6D1_IMPLEMENTATION_AUDIT.md','EXP6D1_N1_LOSS_VERIFICATION.md','EXP6D1_TRAINING_SUMMARY.md','EXP6D1_VALIDATION_COMPARISON.csv','EXP6D1_PER_LABEL_COMPARISON.csv','EXP6D1_DISTANCE_WEIGHT_DIAGNOSTICS.csv','EXP6D1_GEOMETRY_REPORT.csv','EXP6D1_VERIFICATION_REPORT.md','rare_label_metrics.json']:
        need((root/name).is_file(),'Required deliverable missing: '+name)
    print('EXP6D1_SAVED_ARTIFACT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','train','verify','verify-smoke']);p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    try:{'smoke':smoke,'train':train,'verify':verify,'verify-smoke':verify_smoke}[args.phase](args.run_root)
    except BaseException as e:
        if args.phase in ['smoke','train'] and args.run_root.exists() and not (args.run_root/'SUCCESS.json').exists() and not (args.run_root/'FAILURE.json').exists():
            write(args.run_root/'FAILURE.json',dict(phase=args.phase,error=repr(e),traceback=traceback.format_exc(),time=now(),new_test=False,automatic_retry=False))
        raise
