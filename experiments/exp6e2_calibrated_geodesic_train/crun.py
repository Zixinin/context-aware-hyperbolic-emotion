"""One calibrated A1-G-C seed42 training run; no Test or coefficient search."""
import argparse,csv,traceback
from ccommon import *
from cprobe import run_probe,PREFIXES
from cgeometry import ProjectionMonitor

@torch.no_grad()
def predict(model,tok,records,description):
    need(len(records) in [64,3834],'Only smoke/full Validation allowed')
    model.eval();ids=[];zs=[];ys=[];sums={};count=0;maximum=0.
    for i,batch in enumerate(loader(records,tok),1):
        record_ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x);n=len(record_ids)
        need(bool(torch.isfinite(out['loss']) and torch.isfinite(out['logits']).all()),'Nonfinite Validation forward')
        v=diagnostic_values(out)
        for k,value in v.items():
            if k!='max_ball_norm':sums[k]=sums.get(k,0.)+value*n
        maximum=max(maximum,v['max_ball_norm']);count+=n;ids+=record_ids;zs.append(out['logits'].cpu().numpy());ys.append(x['labels'].cpu().numpy())
        if i%25==0:print(description,i,count,flush=True)
    logits=np.concatenate(zs);gold=np.concatenate(ys)
    need(ids==[r['id'] for r in records] and np.array_equal(gold,targets(records)),'Validation canonical data mismatch')
    need(all(not m.training for m in model.modules()),'Validation dropout active')
    result={k:v/count for k,v in sums.items()};result['max_ball_norm']=maximum
    result['weighted_aux_to_BCE']=result['weighted_auxiliary']/result['bce']
    return logits,gold,np.array(ids),result

def step(model,batch,optimizer,scheduler,detailed=False):
    ids=batch.pop('ids');device=next(model.parameters()).device
    x={k:v.to(device,non_blocking=True) for k,v in batch.items()}
    with ProjectionMonitor(model,len(ids)) as monitor:
        out=model(**x);geometry=monitor.values(out['z'])
    need(bool(torch.isfinite(out['logits']).all()),'Nonfinite classifier logits')
    values=diagnostic_values(out);n=len(ids);(out['loss']*n).backward()
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(n)
    need(model.emotion_embeddings.grad is None,'Fixed labels got gradients')
    gradients={}
    if detailed:
        for group,prefix in PREFIXES.items():
            ps=[p for name,p in model.named_parameters() if name.startswith(prefix)]
            need(ps and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in ps),'Missing/invalid training module gradient')
            norm=sum(float(p.grad.detach().double().square().sum()) for p in ps)**.5;need(norm>0,'Zero training module gradient')
            gradients[group]=dict(norm=norm,parameter_tensors=len(ps))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return dict(values,ids=ids,batch_size=n,unclipped_grad_norm=float(norm),gradients=gradients,**geometry)

def parity(root,data,manifest,snapshot):
    names=manifest['names27'];old=read(A1/'INITIALIZATION_PARITY.json')
    seed_everything();reference,tok,_,_=build(snapshot,names,geodesic=False);reference.to('cuda')
    shared=state_hash(reference.control.state_dict());hyper=state_hash(reference.hyper.state_dict());full=state_hash(reference.state_dict())
    need(shared==old['shared_state_sha256'] and hyper==old['hyperbolic_initial_state_sha256'],'Historical A1 initialization cannot be matched; stop, do not silently train another control')
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    batch=next(iter(loader(data['train'],tok,True)));ids=batch.pop('ids');need(ids==old['first_canonical_training_batch_ids'],'Historical first training batch differs')
    x={k:v.to('cuda') for k,v in batch.items()};expected={}
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        for mode in [True,False]:
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);reference.train(mode);out=reference(**x)
            expected[str(mode)]={k:out[k].cpu() for k in ['logits','z','decoder_logits']}
    import hashlib
    need(hashlib.sha256(expected['True']['logits'].numpy().tobytes()).hexdigest()==old['initial_train_logits_sha256'],'Historical initial train logits hash differs')
    del reference,out;release()
    seed_everything();model,tok,_,identity=build(snapshot,names);need(state_hash(model.state_dict())==full,'Loss installation changed initial parameters/buffers')
    need(torch.equal(torch.get_rng_state(),cpu) and torch.equal(torch.cuda.get_rng_state(),cuda),'Loss installation changed constructor RNG')
    # Every actual curvature attribute must agree; constants are not new params.
    cs=[]
    for name,m in model.hyper.named_modules():
        for key in ['c','c_in','c_out']:
            if hasattr(m,key):
                value=getattr(m,key);number=float(value.detach().cpu()) if isinstance(value,torch.Tensor) else float(value)
                need(number==1.,'Inconsistent geometric curvature');cs.append(dict(module=name,attribute=key,c=number))
    model.to('cuda')
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        for mode in [True,False]:
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);model.train(mode)
            with ProjectionMonitor(model,len(ids)) as monitor:
                out=model(**x);observed_geometry=monitor.values(out['z'])
            need(torch.equal(out['loss'],out['components']['classification']+GAMMA*out['components']['auxiliary']),'Calibrated forward coefficient wrong')
            for key in ['logits','z','decoder_logits']:need(torch.equal(out[key].cpu(),expected[str(mode)][key]),'A1/A1-G initial forward not exactly equal: '+key)
    opt,groups=parameter_groups(model);need(groups==read(A1/'optimizer_protocol.json')['groups'],'Original A1 optimizer grouping changed')
    need(np.array_equal(model.emotion_embeddings.cpu(),np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Model label order/coordinates differ')
    proof=dict(shared_state_sha256=shared,hyperbolic_initial_state_sha256=hyper,full_initial_state_sha256=full,
        historical_A1_initialization_sha256=sha(A1/'INITIALIZATION_PARITY.json'),all_initial_forward_tensors_exact=True,
        constructor_rng_exact=True,optimizer_groups_exact=True,no_parameter_or_buffer_added=True,curvature_attributes=cs,
        input_scale=1.,fresh_pretrained=True,no_checkpoint_continuation=True,identity=identity,
        gamma_G=GAMMA,projection_monitor_forward_exact=True)
    write(root/'INITIALIZATION_PARITY.json',proof);del model,opt,out,x,batch;release();return proof

def replay_fresh_calibration(root):
    saved=csvrows(OLD/'probes/fresh_initial/MODULE_GRADIENTS.csv')
    actual=csvrows(root/'probes/fresh_initial/MODULE_GRADIENTS.csv')
    need(len(saved)==len(actual)==40,'Fresh replay needs all modules/cohorts')
    need(len({(r['cohort'],r['module']) for r in actual})==40 and {(r['cohort'],r['module']) for r in saved}=={(r['cohort'],r['module']) for r in actual},'Fresh replay cohort/module rows differ')
    errors=[]
    for r in actual:
        old=next(v for v in saved if v['cohort']==r['cohort'] and v['module']==r['module'])
        for key,factor in [('bce_norm',1.),('coordinate_aux_norm',1.),('geodesic_aux_norm',GAMMA/.1)]:
            measured=float(r[key]);expected=float(old[key])*factor
            need(np.isclose(measured,expected,rtol=3e-5,atol=3e-8),'Fresh calibrated gradient replay differs: '+r['cohort']+'/'+r['module']+'/'+key)
            errors.append(abs(measured-expected))
        for key in ['bce_geodesic_aux_cosine','old_new_aux_cosine']:
            need((r[key]=='')==(old[key]==''),'Fresh gradient connectivity differs')
            if r[key]!='':need(np.isclose(float(r[key]),float(old[key]),rtol=3e-5,atol=3e-6),'Fresh direction replay differs')
    write(root/'FRESH_CALIBRATION_REPLAY.json',dict(full_precision_source_sha256=sha(OLD/'probes/fresh_initial/MODULE_GRADIENTS.csv'),
        calibrated_proposal_sha256=sha(CAL/'FROZEN_GAMMA_PROPOSAL.json'),gamma_G=GAMMA,all40_module_rows_match=True,
        max_absolute_norm_error=max(errors),tolerance_rtol=3e-5,tolerance_atol=3e-8,optimizer_steps=0))


def smoke(root):
    from transformers import get_linear_schedule_with_warmup
    need((root/'AUDIT_SUCCESS.json').exists() and not (root/'smoke').exists(),'Audit required; do not repeat smoke')
    freeze_check(root);runtime();data,manifest=data27();plan=read(root/'PLAN.json');snapshot=Path(plan['snapshot'])
    parity(root,data,manifest,snapshot);directory=root/'smoke';directory.mkdir()
    selection=read(root/'FIXED_TRAIN_COHORTS.json')
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27']);model.to('cuda')
    run_probe(root,'fresh_initial',model,tok,data,selection,gate=True)
    del model;release();replay_fresh_calibration(root)
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27']);model.to('cuda').train()
    opt,groups=parameter_groups(model);scheduler=get_linear_schedule_with_warmup(opt,765,7648)
    iterator=iter(loader(data['train'],tok,True));steps=[]
    for i in range(2):
        lr=opt.param_groups[0]['lr'];r=step(model,next(iterator),opt,scheduler,True)
        if i==0:need(r['ids']==read(root/'planned_train_order_epoch1.json')[:16],'Smoke train order differs')
        steps.append(dict(r,lr_used=lr,lr_after=opt.param_groups[0]['lr']))
    need(steps[0]['lr_used']==0. and steps[1]['lr_used']>0.,'Warmup scheduler differs')
    before=predict(model,tok,data['validation'][:64],'Smoke64 Validation')
    torch.save(model.state_dict(),directory/'discarded_weights.pt')
    del model,opt,scheduler,iterator;release()
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27'])
    model.load_state_dict(torch.load(directory/'discarded_weights.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    after=predict(model,tok,data['validation'][:64],'Smoke64 reload')
    need(all(np.array_equal(a,b) for a,b in zip(before[:3],after[:3])),'Smoke reload not exact')
    write(directory/'SMOKE.json',dict(discarded_optimizer_steps=steps,reload_examples=64,logits_ids_gold_exact=True,
        no_Test=True,gamma_G=GAMMA,formal_initialization_is_independent=True))
    del model;release();freeze_check(root)
    write(root/'SMOKE_SUCCESS.json',dict(status='CALIBRATED_GEODESIC_SMOKE_PASS',plan_sha256=sha(root/'PLAN.json'),
        smoke_artifacts_sha256=artifact_hashes(directory),pretraining_probe_tags=['fresh_initial'],
        calibration_replay_sha256=sha(root/'FRESH_CALIBRATION_REPLAY.json'),completed=now()))
    print('CALIBRATED_GEODESIC_SMOKE_PASS — STARTING AUTHORIZED FOUR-EPOCH TRAINING NEXT',flush=True)


def check_smoke(root):
    proof=read(root/'SMOKE_SUCCESS.json')
    need(proof['status']=='CALIBRATED_GEODESIC_SMOKE_PASS' and proof['plan_sha256']==sha(root/'PLAN.json'),'Smoke/plan mismatch')
    need(proof['smoke_artifacts_sha256']==artifact_hashes(root/'smoke'),'Smoke artifacts changed')
    need(proof['pretraining_probe_tags']==['fresh_initial'] and not read(root/'probes/fresh_initial/SUMMARY.json')['severe_findings'],'Calibrated preflight not passed')
    need(proof['calibration_replay_sha256']==sha(root/'FRESH_CALIBRATION_REPLAY.json'),'Calibration replay proof changed')
    need(read(root/'FRESH_CALIBRATION_REPLAY.json')['all40_module_rows_match'],'Fresh actual gradient replay missing')


def train(root):
    from transformers import get_linear_schedule_with_warmup,AutoTokenizer
    check_smoke(root);freeze_check(root);env=runtime();data,manifest=data27();plan=read(root/'PLAN.json')
    for k in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[k]==plan['runtime'][k],'Runtime drift')
    need(not (root/'FORMAL_TRAINING_CLAIM.json').exists() and not list(root.glob('FAILURE_*.json')),'Prior training/failure exists; no automatic retry')
    write(root/'FORMAL_TRAINING_CLAIM.json',dict(seed=42,epochs=4,started=now(),runs=1,retries=False,new_Test=False,gamma_G=GAMMA,input_scale=1.,
        source='current human authorization for one calibrated seed42 pilot'))
    print(f'FORMAL_TRAINING_START A1-G-C seed42 epochs=4 steps=7648 gamma_G={GAMMA!r} s=1 NO_TEST',flush=True)
    seed_everything();snapshot=Path(plan['snapshot']);model,tok,_,_=build(snapshot,manifest['names27']);initial=read(root/'INITIALIZATION_PARITY.json')
    need(state_hash(model.state_dict())==initial['full_initial_state_sha256'],'Formal initial parameters differ from original A1/fresh smoke')
    model.to('cuda');opt,groups=parameter_groups(model);training=loader(data['train'],tok,True)
    scheduler=get_linear_schedule_with_warmup(opt,765,7648);need(len(training)==1912,'Wrong training budget')
    write(root/'optimizer_protocol.json',dict(groups=groups,total_steps=7648,warmup_steps=765,optimizer='original A1 AdamW5e-5',clip=1.))
    checkpoint=root/'best_checkpoint';checkpoint.mkdir();selection=read(root/'FIXED_TRAIN_COHORTS.json');history=[];stepno=0;best=-1.
    env.update(status='TRAINING',completed_epochs=0,completed_optimizer_steps=0,new_test_inference_calls=0);write(root/'environment.json',env)
    fields=['epoch','batch','step','batch_size','loss','bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean','positive_distance_mean',
        'weighted_ce','weighted_distance','coordinate_auxiliary_same_forward','weighted_aux_to_BCE','max_ball_norm',
        'first_text_clip_count','later_text_clip_count','final_radius_above099_count','final_radius_mean','final_radius_min','final_radius_max','text_projection_calls',
        'unclipped_grad_norm','lr_used','lr_after']
    with (root/'batch_log.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,5):
            model.train();opt.zero_grad(set_to_none=True);sums={};order=[];norms=[];geometry_counts=dict(first=0,later=0,near=0);radius_sum=0.;radius_max=0.;radius_min=1.
            for i,batch in enumerate(training,1):
                lr=opt.param_groups[0]['lr'];r=step(model,batch,opt,scheduler,detailed=i==1);stepno+=1;n=r['batch_size'];order+=r['ids'];norms.append(r['unclipped_grad_norm'])
                if i==1:write(root/f'epoch{epoch}_first_batch.json',r)
                for k in ['loss','bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean','positive_distance_mean','weighted_ce','weighted_distance']:sums[k]=sums.get(k,0.)+r[k]*n
                geometry_counts['first']+=r['first_text_clip_count'];geometry_counts['later']+=r['later_text_clip_count'];geometry_counts['near']+=r['final_radius_above099_count']
                radius_sum+=n*r['final_radius_mean'];radius_max=max(radius_max,r['final_radius_max']);radius_min=min(radius_min,r['final_radius_min'])
                writer.writerow(dict(epoch=epoch,batch=i,step=stepno,lr_used=lr,lr_after=opt.param_groups[0]['lr'],**{k:v for k,v in r.items() if k not in ['ids','gradients']}))
                if i%50==0 or i==1912:f.flush();print(f'A1-G-C epoch{epoch}/4 batch{i}/1912 total={r["loss"]:.6f} BCE={r["bce"]:.6f} distance={r["positive_distance_mean"]:.6f}',flush=True)
            need(len(order)==30587 and order==read(root/f'planned_train_order_epoch{epoch}.json'),'Full training order differs from historical A1')
            write(root/f'train_order_epoch{epoch}.json',order)
            val=predict(model,tok,data['validation'],f'epoch{epoch} Validation');metrics=extended_score(val[1],probabilities(val[0]),manifest['names27'])[0]
            record=dict(epoch=epoch,global_step=stepno,training_examples=len(order),**{'training_'+k:v/len(order) for k,v in sums.items()},
                **{'validation_'+k:v for k,v in val[3].items()},validation_micro_f1=metrics['micro_f1'],validation_macro_f1=metrics['macro_f1'],
                validation_rare_f1=metrics['rare_f1'],validation_macro_average_precision=metrics['macro_average_precision'],
                training_first_text_clip_fraction=geometry_counts['first']/len(order),training_later_text_clip_fraction=geometry_counts['later']/len(order),
                training_final_fraction_above099=geometry_counts['near']/len(order),training_final_radius_mean=radius_sum/len(order),
                training_final_radius_min=radius_min,training_final_radius_max=radius_max,
                gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(max(norms)),global_gradient_clip_fraction=float(np.mean(np.array(norms)>1.)),learning_rate=opt.param_groups[0]['lr'])
            history.append(record);table(root/'training_log.csv',history);write(root/f'epoch{epoch}_validation_metrics.json',metrics)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];torch.save(model.state_dict(),checkpoint/'model.pt');tok.save_pretrained(checkpoint/'tokenizer');model.control.encoder.config.save_pretrained(checkpoint/'encoder_config')
                save_validation(root,'validation_selected',*val[:3],manifest['names27']);write(root/'selected_loss_diagnostics.json',val[3])
                write(root/'checkpoint_selection.json',dict(epoch=epoch,step=stepno,validation_macro_f1=best,threshold=.3,tie_break='earliest',checkpoint_sha256=sha(checkpoint/'model.pt')))
            # Finite/geometry/path gates remain active; old-coordinate objective
            # may approach zero on a geodesic-trained model, so the pretraining
            # ratio gate is descriptive after training begins, never gamma tuning.
            run_probe(root,f'epoch{epoch}',model,tok,data,selection,gate=False)
            env.update(completed_epochs=epoch,completed_optimizer_steps=stepno);write(root/'environment.json',env);print('EPOCH_RESULT',record,flush=True)
    need(stepno==7648 and opt.param_groups[0]['lr']==0.,'Incomplete training schedule')
    del model,opt,scheduler,training,val;release()
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27']);model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    tok=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer'),local_files_only=True,use_fast=True)
    val=predict(model,tok,data['validation'],'Full3834 Validation reload')
    with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as saved:
        need(all(np.array_equal(saved[k],v) for k,v in zip(['logits','targets','ids'],val[:3])),'Full Validation reload not exact')
    for k,v in read(root/'selected_loss_diagnostics.json').items():need(np.isclose(v,val[3][k],atol=1e-6,rtol=1e-6),'Reload loss differs')
    need(np.array_equal(model.emotion_embeddings.cpu(),np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Fixed label vectors changed')
    save_validation(root,'validation',*val[:3],manifest['names27']);write(root/'validation_loss_diagnostics.json',val[3])
    write(root/'checkpoint_reload_verification.json',dict(examples=3834,logits_exact=True,ids_equal=True,targets_equal=True,max_abs_logit_error=0.))
    run_probe(root,'best_reload',model,tok,data,selection,gate=False);del model;release();freeze_check(root)
    env.update(status='COMPLETE',completed=now());write(root/'environment.json',env)
    from creport import report
    report(root,manifest);write(root/'FROZEN_POSTCHECK.json',dict(unchanged=True,source_and_historical_artifacts=True,completed=now()))
    verify(root,sealed=False)
    write(root/'SUCCESS.json',dict(status=STATUS,seed=42,epochs=4,optimizer_steps=7648,new_test_inference_calls=0,artifacts_sha256=artifact_hashes(root),completed=now()))
    verify(root);print((root/'GEODESIC_ALIGNMENT_SUMMARY_CN.md').read_text(encoding='utf-8'),flush=True)

def verify(root,sealed=True):
    need(source_hashes()==read(root/'PLAN.json')['source_sha256'],'Current verifier/source differs from frozen run')
    if sealed:
        p=read(root/'SUCCESS.json');need(p['status']==STATUS and p['artifacts_sha256']==artifact_hashes(root),'Saved artifact seal mismatch')
    need(read(root/'config.json')==CONFIG,'Protocol mismatch');check_smoke(root)
    need(not list(root.glob('test*')),'Unexpected new Test artifacts')
    data,manifest=data27();env=read(root/'environment.json')
    need(env['status']=='COMPLETE' and env['completed_epochs']==4 and env['completed_optimizer_steps']==7648 and env['new_test_inference_calls']==0,'Pilot incomplete')
    history=csvrows(root/'training_log.csv');need([int(r['global_step']) for r in history]==[1912,3824,5736,7648],'Epoch budget wrong')
    expected=max(history,key=lambda r:float(r['validation_macro_f1']));selection=read(root/'checkpoint_selection.json')
    need(selection['epoch']==int(expected['epoch']) and selection['validation_macro_f1']==float(expected['validation_macro_f1']),'Checkpoint selection not highest Validation Macro-F1 earliest tie')
    need(sha(root/'best_checkpoint/model.pt')==selection['checkpoint_sha256'],'Checkpoint bytes changed')
    logs=csvrows(root/'batch_log.csv');need([int(r['step']) for r in logs]==list(range(1,7649)) and float(logs[-1]['lr_after'])==0.,'Missing steps/scheduler completion')
    for r in logs:
        need(all(np.isfinite(float(v)) for v in r.values()),'Nonfinite saved training scalar')
        need(abs(float(r['weighted_auxiliary'])-GAMMA*float(r['auxiliary']))<1e-9,'Frozen gamma changed in saved training log')
        need(abs(float(r['weighted_ce'])-GAMMA*float(r['positive_ce_mean']))<1e-9 and abs(float(r['weighted_distance'])-GAMMA*float(r['positive_distance_mean']))<1e-9,'Component coefficients differ')
        need(abs(float(r['loss'])-float(r['bce'])-float(r['weighted_auxiliary']))<3e-6,'BCE/aux composition wrong')
        need(abs(float(r['auxiliary'])-float(r['positive_ce_mean'])-float(r['positive_distance_mean']))<5e-6,'Geodesic auxiliary reduction wrong')
        n=int(r['batch_size'])
        need(n in [16,11] and int(r['text_projection_calls'])>=2,'Wrong actual training batch/projection observation')
        for key in ['first_text_clip_count','later_text_clip_count','final_radius_above099_count']:
            need(0<=int(r[key])<=n,'Invalid actual training geometry count')
        need(0<=float(r['final_radius_min'])<=float(r['final_radius_mean'])+1e-7 and float(r['final_radius_mean'])<=float(r['final_radius_max'])+1e-7 and float(r['final_radius_max'])<1.,'Invalid saved ball radii')
    for record in history:
        e=int(record['epoch']);batches=[r for r in logs if int(r['epoch'])==e]
        count=sum(int(r['batch_size']) for r in batches)
        need(len(batches)==1912 and count==30587,'Wrong epoch training examples')
        for out,key in [('training_first_text_clip_fraction','first_text_clip_count'),('training_later_text_clip_fraction','later_text_clip_count'),('training_final_fraction_above099','final_radius_above099_count')]:
            need(abs(float(record[out])-sum(int(r[key]) for r in batches)/count)<1e-12,'Training geometry aggregate mismatch')
        need(abs(float(record['global_gradient_clip_fraction'])-float(np.mean([float(r['unclipped_grad_norm'])>1 for r in batches])))<1e-12,'Actual clip-frequency aggregate mismatch')
    for e in range(1,5):need(read(root/f'train_order_epoch{e}.json')==read(A1/f'train_order_epoch{e}.json'),'Historical A1 data order changed')
    with np.load(root/'validation_predictions.npz',allow_pickle=False) as z:
        need(z['ids'].tolist()==[r['id'] for r in data['validation']] and z['label_names'].tolist()==manifest['names27'] and np.array_equal(z['targets'],targets(data['validation'])),'Canonical saved Validation mismatch')
        m,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
        need(m==read(root/'validation_metrics.json') and np.array_equal(pred,z['predictions']) and np.array_equal(probabilities(z['logits']),z['probabilities']),'Saved metrics/probabilities disagree')
        recorded=csvrows(root/'validation_per_label.csv');need(len(recorded)==27,'Missing27 per-label metrics')
        for a,b in zip(recorded,rows):
            need(a['label']==b['label'],'Per-label order changed')
            for k in ['precision','recall','f1','average_precision','tp','fp','fn','support']:need(abs(float(a[k])-b[k])<1e-12,'Per-label metric changed')
        with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as old:
            for k in ['logits','targets','ids','predictions']:need(np.array_equal(z[k],old[k]),'Selected/reloaded arrays not identical')
    reload=read(root/'checkpoint_reload_verification.json');need(reload['examples']==3834 and reload['logits_exact'] and reload['max_abs_logit_error']==0,'Full reload proof missing')
    for tag in ['fresh_initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        p=read(root/'probes'/tag/'NO_UPDATE_PROOF.json')
        need(p['state_before']==p['state_after'] and p['optimizer_steps']==p['test_forward_calls']==0 and p['all_grads_none'],'Probe state/gradient mutation')
        need(all(p[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode not restored')
    need(read(root/'PLAN.json')['calibration']==verify_calibration(),'Frozen calibration changed')
    need(read(root/'FORMAL_TRAINING_CLAIM.json')['gamma_G']==GAMMA,'Formal coefficient mismatch')
    need(read(root/'FROZEN_POSTCHECK.json')['unchanged'],'Historical protection proof missing')
    print('EXP6E2_A1GC_SAVED_ARTIFACT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','smoke','train','verify']);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=run_path(args.root)
    try:
        if args.phase=='prepare':
            from cprepare import prepare
            prepare(root)
        else:{'smoke':smoke,'train':train,'verify':verify}[args.phase](root)
    except BaseException as error:
        if args.phase!='verify':write(root/f'FAILURE_{args.phase}_{__import__("time").time_ns()}.json',
            dict(phase=args.phase,error=repr(error),traceback=traceback.format_exc(),gamma_G=GAMMA,new_Test=False,no_automatic_retry=True))
        raise
