"""One seed42 A1-G smoke/pilot, serial, no new Test or automatic retries."""
import argparse,csv,traceback
from gcommon import *
from gprobe import run_probe,PREFIXES,ScaleReviewRequired

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
    ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x)
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
    return dict(values,ids=ids,batch_size=n,unclipped_grad_norm=float(norm),gradients=gradients)

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
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);model.train(mode);out=model(**x)
            for key in ['logits','z','decoder_logits']:need(torch.equal(out[key].cpu(),expected[str(mode)][key]),'A1/A1-G initial forward not exactly equal: '+key)
    opt,groups=parameter_groups(model);need(groups==read(A1/'optimizer_protocol.json')['groups'],'Original A1 optimizer grouping changed')
    need(np.array_equal(model.emotion_embeddings.cpu(),np.load(root/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Model label order/coordinates differ')
    proof=dict(shared_state_sha256=shared,hyperbolic_initial_state_sha256=hyper,full_initial_state_sha256=full,
        historical_A1_initialization_sha256=sha(A1/'INITIALIZATION_PARITY.json'),all_initial_forward_tensors_exact=True,
        constructor_rng_exact=True,optimizer_groups_exact=True,no_parameter_or_buffer_added=True,curvature_attributes=cs,
        input_scale=1.,fresh_pretrained=True,no_checkpoint_continuation=True,identity=identity)
    write(root/'INITIALIZATION_PARITY.json',proof);del model,opt,out,x,batch;release();return proof

def smoke(root):
    from transformers import get_linear_schedule_with_warmup
    need((root/'AUDIT_SUCCESS.json').exists() and not (root/'smoke').exists(),'Audit required; do not repeat smoke')
    freeze_check(root);runtime();data,manifest=data27();plan=read(root/'PLAN.json');snapshot=Path(plan['snapshot'])
    parity(root,data,manifest,snapshot);directory=root/'smoke';directory.mkdir()
    selection=read(root/'FIXED_TRAIN_COHORTS.json')
    summaries=[]
    for tag,checkpoint in [('fresh_initial',None),('original_A1_best',A1/'best_checkpoint/model.pt')]:
        seed_everything();model,tok,_,_=build(snapshot,manifest['names27'])
        if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.to('cuda');summaries.append(run_probe(root,tag,model,tok,data,selection,gate=False));del model;release()
    # Complete BOTH no-update timepoints before reporting the scale stop. No
    # optimizer.step occurs unless both gates pass. Parity may instantiate an
    # optimizer only to inspect its grouping, without a state update.
    if any(s['severe_findings'] for s in summaries):
        lines=['# 无更新诊断完成：原权重下尺度门触发，正式pilot暂停','',
            '预登记门：训练前同forward的aux绝对尺度或total梯度norm超过原目标10倍。没有改gamma=.1、scale=1、标签向量或公式。没有执行optimizer.step、正式训练或Test。','',
            '|时点|cohort|旧aux|新aux|aux比例|total梯度比例|旧shared aux norm|新shared aux norm|BCE-new aux cos|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for s in summaries:
            directory=root/'probes'/s['tag'];grads=csvrows(directory/'MODULE_GRADIENTS.csv')
            for r in csvrows(directory/'LOSS_SCALES.csv'):
                g=next(g for g in grads if g['cohort']==r['cohort'] and g['module']=='shared_encoder')
                lines.append('|'+s['tag']+'|'+r['cohort']+'|'+'|'.join(f'{float(v):.6f}' for v in [r['coordinate_auxiliary_same_forward'],r['auxiliary'],r['geodesic_to_coordinate_aux_absolute_scale'],r['total_gradient_norm_ratio'],g['coordinate_aux_norm'],g['geodesic_aux_norm'],g['bce_geodesic_aux_cosine']])+'|')
        lines+=['','新距离和旧坐标内积的单位/符号/梯度不同。接近球边界时坐标梯度可很大；z处的坐标梯度不能替代共享encoder梯度，更不能直接推出Adam实际参数更新的倍数。','',
            '建议的单独敏感性设计：先在固定train-only fresh模型探针上定义一个全局gamma_G，使共享参数weighted-aux/BCE范数比接近原A1；系数在训练前固定，不能按Validation/Test结果择优，也不逐batch归一化。它与原gamma=.1构成另一个权重实验，不能伪装成当前仅替换alignment的消融。本轮不计算或训练这些候选权重。',
            '当前A1-G性能未知；数学/梯度检查通过不等于Macro-F1改善。保留原权重单因素pilot代码，但停止门不会自动绕过。']
        md(root/'SCALE_REVIEW_STOP.md',lines);write(root/'PRETRAIN_DIAGNOSTICS_STATUS.json',dict(status='SCALE_REVIEW_REQUIRED',summaries=summaries,optimizer_step_calls=0,new_test_forward_calls=0))
        raise ScaleReviewRequired('Pretraining scale gate triggered; both train-only no-update timepoints completed, no optimizer steps')
    # Independent two-step numerical smoke, discarded before the formal model.
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27']);model.to('cuda').train()
    opt,groups=parameter_groups(model);scheduler=get_linear_schedule_with_warmup(opt,765,7648);iterator=iter(loader(data['train'],tok,True));steps=[]
    for i in range(2):
        lr=opt.param_groups[0]['lr'];r=step(model,next(iterator),opt,scheduler,True)
        if i==0:need(r['ids']==read(root/'planned_train_order_epoch1.json')[:16],'Smoke train order differs')
        steps.append(dict(r,lr_used=lr,lr_after=opt.param_groups[0]['lr']))
    need(steps[0]['lr_used']==0. and steps[1]['lr_used']>0.,'Warmup scheduler differs')
    before=predict(model,tok,data['validation'][:64],'Smoke64 Validation');torch.save(model.state_dict(),directory/'discarded_weights.pt')
    del model,opt,scheduler,iterator;release()
    seed_everything();model,tok,_,_=build(snapshot,manifest['names27']);model.load_state_dict(torch.load(directory/'discarded_weights.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    after=predict(model,tok,data['validation'][:64],'Smoke64 reload')
    need(all(np.array_equal(a,b) for a,b in zip(before[:3],after[:3])),'Smoke reload not exact')
    write(directory/'SMOKE.json',dict(discarded_optimizer_steps=steps,reload_examples=64,logits_ids_gold_exact=True,no_Test=True))
    del model;release();freeze_check(root)
    write(root/'SMOKE_SUCCESS.json',dict(status='GEODESIC_ALIGNMENT_SMOKE_PASS',plan_sha256=sha(root/'PLAN.json'),
        smoke_artifacts_sha256=artifact_hashes(directory),pretraining_probe_tags=['fresh_initial','original_A1_best'],completed=now()))
    print('GEODESIC_ALIGNMENT_SMOKE_PASS — READY FOR ONE AUTHORIZED FRESH PILOT',flush=True)

def check_smoke(root):
    proof=read(root/'SMOKE_SUCCESS.json');need(proof['status']=='GEODESIC_ALIGNMENT_SMOKE_PASS' and proof['plan_sha256']==sha(root/'PLAN.json'),'Smoke not bound to this plan')
    need(proof['smoke_artifacts_sha256']==artifact_hashes(root/'smoke'),'Smoke artifacts changed')
    for tag in proof['pretraining_probe_tags']:
        p=read(root/'probes'/tag/'SUMMARY.json');need(not p['severe_findings'],'Pretraining severe scale gate not passed')

def train(root):
    from transformers import get_linear_schedule_with_warmup,AutoTokenizer
    check_smoke(root);freeze_check(root);env=runtime();data,manifest=data27();plan=read(root/'PLAN.json')
    for k in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[k]==plan['runtime'][k],'Runtime drift')
    need(not (root/'FORMAL_TRAINING_CLAIM.json').exists() and not list(root.glob('FAILURE_*.json')),'Prior training/failure exists; no automatic retry')
    write(root/'FORMAL_TRAINING_CLAIM.json',dict(seed=42,epochs=4,started=now(),runs=1,retries=False,new_Test=False))
    seed_everything();snapshot=Path(plan['snapshot']);model,tok,_,_=build(snapshot,manifest['names27']);initial=read(root/'INITIALIZATION_PARITY.json')
    need(state_hash(model.state_dict())==initial['full_initial_state_sha256'],'Formal initial parameters differ from original A1/fresh smoke')
    model.to('cuda');opt,groups=parameter_groups(model);training=loader(data['train'],tok,True)
    scheduler=get_linear_schedule_with_warmup(opt,765,7648);need(len(training)==1912,'Wrong training budget')
    write(root/'optimizer_protocol.json',dict(groups=groups,total_steps=7648,warmup_steps=765,optimizer='original A1 AdamW5e-5',clip=1.))
    checkpoint=root/'best_checkpoint';checkpoint.mkdir();selection=read(root/'FIXED_TRAIN_COHORTS.json');history=[];stepno=0;best=-1.
    env.update(status='TRAINING',completed_epochs=0,completed_optimizer_steps=0,new_test_inference_calls=0);write(root/'environment.json',env)
    fields=['epoch','batch','step','batch_size','loss','bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean','positive_distance_mean',
        'coordinate_auxiliary_same_forward','weighted_aux_to_BCE','max_ball_norm','unclipped_grad_norm','lr_used','lr_after']
    with (root/'batch_log.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,5):
            model.train();opt.zero_grad(set_to_none=True);sums={};order=[];norms=[]
            for i,batch in enumerate(training,1):
                lr=opt.param_groups[0]['lr'];r=step(model,batch,opt,scheduler,detailed=i==1);stepno+=1;n=r['batch_size'];order+=r['ids'];norms.append(r['unclipped_grad_norm'])
                if i==1:write(root/f'epoch{epoch}_first_batch.json',r)
                for k in ['loss','bce','auxiliary','weighted_auxiliary','positive_ce_mean','positive_dot_mean','positive_distance_mean']:sums[k]=sums.get(k,0.)+r[k]*n
                writer.writerow(dict(epoch=epoch,batch=i,step=stepno,lr_used=lr,lr_after=opt.param_groups[0]['lr'],**{k:v for k,v in r.items() if k not in ['ids','gradients']}))
                if i%50==0 or i==1912:f.flush();print(f'A1-G epoch{epoch}/4 batch{i}/1912 total={r["loss"]:.6f} BCE={r["bce"]:.6f} distance={r["positive_distance_mean"]:.6f}',flush=True)
            need(len(order)==30587 and order==read(root/f'planned_train_order_epoch{epoch}.json'),'Full training order differs from historical A1')
            write(root/f'train_order_epoch{epoch}.json',order)
            val=predict(model,tok,data['validation'],f'epoch{epoch} Validation');metrics=extended_score(val[1],probabilities(val[0]),manifest['names27'])[0]
            record=dict(epoch=epoch,global_step=stepno,training_examples=len(order),**{'training_'+k:v/len(order) for k,v in sums.items()},
                **{'validation_'+k:v for k,v in val[3].items()},validation_micro_f1=metrics['micro_f1'],validation_macro_f1=metrics['macro_f1'],
                validation_rare_f1=metrics['rare_f1'],validation_macro_average_precision=metrics['macro_average_precision'],
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
    from greport import report
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
        need(abs(float(r['loss'])-float(r['bce'])-float(r['weighted_auxiliary']))<3e-6,'BCE/aux composition wrong')
        need(abs(float(r['auxiliary'])-float(r['positive_ce_mean'])-float(r['positive_distance_mean']))<5e-6,'Geodesic auxiliary reduction wrong')
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
    for tag in ['fresh_initial','original_A1_best','epoch1','epoch2','epoch3','epoch4','best_reload']:
        p=read(root/'probes'/tag/'NO_UPDATE_PROOF.json')
        need(p['state_before']==p['state_after'] and p['optimizer_steps']==p['test_forward_calls']==0 and p['all_grads_none'],'Probe state/gradient mutation')
        need(all(p[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode not restored')
    need(read(root/'FROZEN_POSTCHECK.json')['unchanged'],'Historical protection proof missing')
    print('GEODESIC_ALIGNMENT_SAVED_ARTIFACT_VERIFICATION_PASS',flush=True)

def inspect_diagnostics(root):
    need(source_hashes()==read(root/'PLAN.json')['source_sha256'],'Current verifier/source differs from diagnostic run')
    p=read(root/'DIAGNOSTIC_SUCCESS.json')
    need(p['status']=='PRETRAIN_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED' and p['artifacts_sha256']==artifact_hashes(root),'Diagnostic seal mismatch')
    need(not (root/'FORMAL_TRAINING_CLAIM.json').exists() and not (root/'SMOKE_SUCCESS.json').exists(),'Unexpected optimizer/training scope')
    for tag in ['fresh_initial','original_A1_best']:
        proof=read(root/'probes'/tag/'NO_UPDATE_PROOF.json')
        need(proof['state_before']==proof['state_after'] and proof['optimizer_steps']==proof['test_forward_calls']==0,'Probe mutated state')
        need(all(proof[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode not restored')
    print('GEODESIC_NO_UPDATE_DIAGNOSTIC_SAVED_VERIFICATION_PASS',flush=True)
    print((root/'SCALE_REVIEW_STOP.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['audit','smoke','train','verify','inspect']);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=run_path(args.root)
    try:
        if args.phase=='audit':
            from gaudit import prepare
            prepare(root)
        else:{'smoke':smoke,'train':train,'verify':verify,'inspect':inspect_diagnostics}[args.phase](root)
    except ScaleReviewRequired as error:
        freeze_check(root)
        write(root/'REVIEW_REQUIRED.json',dict(reason=str(error),phase=args.phase,optimizer_step_calls=0,new_test_forward_calls=0,completed=now()))
        write(root/'DIAGNOSTIC_SUCCESS.json',dict(status='PRETRAIN_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED',artifacts_sha256=artifact_hashes(root),completed=now()))
        print('GEODESIC_PRETRAIN_DIAGNOSTICS_COMPLETE_SCALE_REVIEW_REQUIRED — NO TRAINING / NO TEST',flush=True)
        raise SystemExit(3)
    except BaseException as error:
        if args.phase not in ['verify','inspect']:write(root/f'FAILURE_{args.phase}_{__import__("time").time_ns()}.json',dict(phase=args.phase,error=repr(error),traceback=traceback.format_exc(),new_Test=False,no_automatic_retry=True))
        raise
