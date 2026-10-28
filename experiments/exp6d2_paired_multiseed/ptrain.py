"""Frozen architecture/loss training adapter with a caller-specified paired seed."""
import gc
import numpy as np
import torch
from pcommon import *
from probes import isolated_eval,run_probe as probe_a1s
from nprobe import run_probe as probe_n1
from nmodel import vectors as n1_vectors,describe as describe_n1

def release():gc.collect();torch.cuda.empty_cache()

def probe(root,tag,model,kind,tokenizer,data,selection):
    if kind=='A1S':return probe_a1s(root,tag,model,tokenizer,data,selection)
    if kind=='N1':return probe_n1(root,tag,model,tokenizer,data,selection)
    directory=root/'geometry'/tag;directory.mkdir(parents=True,exist_ok=False)
    named=list(model.named_parameters());params=[p for n,p in named];before=state_hash(model.state_dict());by_id={r['id']:r for r in data['train']}
    rows=[];need(all(p.grad is None for p in params),'A0 probe accumulated grad')
    with isolated_eval(model) as isolation:
        for cohort,records in selection['cohorts'].items():
            batch=Collator(tokenizer)([by_id[r['id']] for r in records]);ids=batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
            out=model(**x);need(bool(torch.isfinite(out['loss'])),'Nonfinite A0 probe')
            grad=torch.autograd.grad(out['loss'],params,allow_unused=False)
            norms={}
            for group,prefix in [('shared_encoder','encoder.'),('classification_head','classifier_')]:
                selected=[g for (n,p),g in zip(named,grad) if n.startswith(prefix)]
                need(selected and all(bool(torch.isfinite(g).all()) for g in selected),'Invalid A0 probe gradients')
                norms[group]=sum(float(g.detach().double().square().sum()) for g in selected)**.5
            rows.append(dict(cohort=cohort,bce=float(out['loss'].detach()),**norms))
            del out,grad,x,batch
    after=state_hash(model.state_dict());need(before==after and all(p.grad is None for p in params),'A0 probe changed state')
    table(directory/'gradients.csv',rows)
    write(directory/'pooled.json',dict(tag=tag,geometry_applicable=False,reason='A0 has no hyperbolic branch',cohorts=10,positions=160))
    write(directory/'NO_UPDATE_PROOF.json',dict(state_before=before,state_after=after,optimizer_steps=0,test_calls=0,**isolation))

@torch.no_grad()
def predict(model,kind,tokenizer,records,seed,description):
    need(len(records) in [64,3834],'Only smoke/full Validation accepted')
    model.eval();logits=[];gold=[];ids=[];sums={};count=0;maximum=0.
    for index,batch in enumerate(make_loader(records,tokenizer,seed),1):
        record_ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x)
        need(bool(torch.isfinite(out['loss'])) and bool(torch.isfinite(out['logits']).all()),'Nonfinite Validation')
        v=values(out,kind);n=len(record_ids)
        for k in ['loss','bce','weighted_bce','auxiliary','weighted_auxiliary']:sums[k]=sums.get(k,0.)+v[k]*n
        maximum=max(maximum,v['max_ball_norm']);count+=n;ids.extend(record_ids)
        logits.append(out['logits'].cpu().numpy());gold.append(x['labels'].cpu().numpy())
        if index%25==0:print(description,index,count,flush=True)
    need(ids==[r['id'] for r in records] and np.array_equal(np.concatenate(gold),targets(records)),'Validation IDs/gold not canonical')
    result={k:v/count for k,v in sums.items()};result['max_ball_norm']=maximum if kind!='A0' else None
    need(all(not m.training for m in model.modules()),'Validation dropout active')
    return np.concatenate(logits),np.concatenate(gold),np.array(ids),result

def update(model,kind,batch,optimizer,scheduler):
    ids=batch.pop('ids');x={k:v.to('cuda',non_blocking=True) for k,v in batch.items()};out=model(**x)
    need(bool(torch.isfinite(out['loss'])) and bool(torch.isfinite(out['logits']).all()),'Nonfinite training')
    v=values(out,kind);vec=n1_vectors(out) if kind=='N1' else None;n=len(ids)
    (out['loss']*n).backward()
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(n)
    if kind!='A0':need(model.emotion_embeddings.grad is None,'Fixed label vectors received gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return dict(v,ids=ids,batch_size=n,gradient_norm=float(norm)),vec

def verify_run(suite,seed,kind,sealed=True):
    root=run_path(suite,seed,kind)
    if sealed:
        success=read(root/'SUCCESS.json');need(success['status']=='EXP6D2_CELL_COMPLETE' and success['artifacts_sha256']==artifacts(root),'Cell seal mismatch')
    need(read(root/'config.json')==config(kind,seed),'Cell protocol mismatch')
    env=read(root/'environment.json');need(env['completed_epochs']==4 and env['completed_optimizer_steps']==7648 and env['new_test_inference_calls']==0,'Incomplete cell')
    need(not list(root.glob('test*')),'Unexpected Test artifacts')
    data,manifest=dataset();need(manifest==read(root/'dataset_manifest.json'),'Cell dataset changed')
    hist=csvrows(root/'training_log.csv');need([int(r['step']) for r in hist]==[1912,3824,5736,7648],'Incomplete epoch budget')
    expected=max(hist,key=lambda r:float(r['validation_macro_f1']));choice=read(root/'checkpoint_selection.json')
    need(choice['epoch']==int(expected['epoch']) and choice['validation_macro_f1']==float(expected['validation_macro_f1']),'Wrong selected checkpoint')
    need(sha(root/'best_checkpoint/model.pt')==choice['checkpoint_sha256'],'Changed checkpoint bytes')
    logs=csvrows(root/'batch_log.csv');need([int(r['step']) for r in logs]==list(range(1,7649)) and float(logs[-1]['lr_after'])==0.,'Wrong steps/scheduler')
    for r in logs:
        need(all(v=='' or np.isfinite(float(v)) for v in r.values()),'Nonfinite batch log')
        need(abs(float(r['loss'])-float(r['weighted_bce'])-float(r['weighted_auxiliary']))<2e-6,'Wrong loss composition')
        if kind=='N1':need(abs(float(r['weight_mean'])-1.)<2e-6,'N1 mean weight changed')
    for epoch in range(1,5):
        need(read(root/f'train_order_epoch{epoch}.json')==read(suite/'planned_orders'/f'seed{seed}_epoch{epoch}.json'),'Not paired train order')
        if kind=='N1':
            with np.load(root/f'train_weights_epoch{epoch}.npz',allow_pickle=False) as z:
                need(z['ids'].tolist()==read(root/f'train_order_epoch{epoch}.json'),'N1 diagnostic IDs differ')
                for i in range(0,len(z['ids']),16):
                    d=z['distance'][i:i+16];need(np.allclose(z['weight'][i:i+16],d/d.mean(),rtol=1e-6,atol=1e-7),'N1 last/current batch normalization differs')
    with np.load(root/'validation_predictions.npz',allow_pickle=False) as z:
        need(z['label_names'].tolist()==manifest['names27'] and z['ids'].tolist()==[r['id'] for r in data['validation']] and np.array_equal(z['targets'],targets(data['validation'])),'Saved Validation not canonical')
        m,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
        need(m==read(root/'validation_metrics.json') and np.array_equal(pred,z['predictions']) and np.array_equal(probabilities(z['logits']),z['probabilities']),'Saved metrics/probabilities changed')
        per=csvrows(root/'validation_per_label.csv');need(len(per)==27,'Need27 labels')
        for a,b in zip(per,rows):
            for k,v in b.items():need(a[k]=='' if v is None else a[k]==v if isinstance(v,str) else np.isclose(float(a[k]),v,atol=1e-12,rtol=0),'Per-label F1/AP/count mismatch')
        with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as selected:
            for k in ['ids','targets','logits','predictions']:need(np.array_equal(selected[k],z[k]),'Reload arrays not exact')
    r=read(root/'checkpoint_reload_verification.json');need(r['examples']==3834 and r['logits_exact'] and r['max_abs_logit_error']==0,'Missing full reload proof')
    for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        p=read(root/'geometry'/tag/'NO_UPDATE_PROOF.json')
        need(p['state_before']==p['state_after'] and p['optimizer_steps']==p['test_calls']==0,'Probe state changed')
        need(all(p[k] for k in ['python_rng_restored','numpy_rng_restored','cpu_rng_restored','cuda_rng_restored','module_modes_restored']),'Probe RNG/mode changed')
    need(read(root/'FROZEN_POSTCHECK.json')['unchanged'],'No frozen input proof')
    print('EXP6D2_CELL_SAVED_VERIFICATION_PASS',seed,kind,flush=True)

def train_one(suite,seed,kind):
    from transformers import get_linear_schedule_with_warmup,AutoTokenizer
    need(os.environ.get('PYTHONHASHSEED')==str(seed),'Formal subprocess hash seed differs')
    root=run_path(suite,seed,kind)
    if (root/'SUCCESS.json').exists():verify_run(suite,seed,kind);return
    need(not root.exists(),'Incomplete/failed cell exists; preserve it and request review, do not overwrite/retry')
    need((suite/'SMOKE_SUCCESS.json').exists(),'Paired smoke incomplete')
    plan=read(suite/'SUITE_PLAN.json');source_hashes();freeze_check(suite);data,manifest=dataset()
    need(plan['source_manifest']==source_hashes(),'Suite code changed')
    env=runtime()
    for key in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[key]==plan['runtime'][key],'Runtime differs')
    root.mkdir(parents=True);write(root/'FORMAL_TRAINING_CLAIM.json',dict(seed=seed,variant=kind,started=now(),fresh_pretrained=True,new_test=False))
    write(root/'config.json',config(kind,seed));write(root/'dataset_manifest.json',manifest)
    seed_all(seed);model,tokenizer=build(kind,Path(plan['snapshot']),manifest['names27']);model.to('cuda')
    initial=read(suite/'smoke'/f'seed{seed}'/'INITIALIZATION.json')[kind]
    need(state_hash(control(model,kind).state_dict())==initial['shared_state'],'Fresh shared initialization changed')
    if kind!='A0':need(state_hash(model.hyper.state_dict())==initial['hyper_state'],'Fresh HNN/decoder initialization changed')
    write(root/'INITIALIZATION.json',initial)
    optimizer,groups=optimizer_for(model);scheduler=get_linear_schedule_with_warmup(optimizer,765,7648)
    training=make_loader(data['train'],tokenizer,seed,True);need(len(training)==1912,'Wrong loader budget')
    write(root/'optimizer_protocol.json',dict(groups=groups,total_steps=7648,warmup_steps=765,lr=5e-5,clip=1.,optimizer='unchanged frozen AdamW builder'))
    selection=read(suite/'FIXED_TRAIN_COHORTS.json');probe(root,'initial',model,kind,tokenizer,data,selection)
    checkpoint=root/'best_checkpoint';checkpoint.mkdir();best=-1.;step=0;history=[];distribution=[]
    env.update(completed_epochs=0,completed_optimizer_steps=0,new_test_inference_calls=0,status='TRAINING');write(root/'environment.json',env)
    import csv
    with (root/'batch_log.csv').open('x',encoding='utf-8',newline='') as f:
        fields=['epoch','batch','step','batch_size','loss','bce','weighted_bce','auxiliary','weighted_auxiliary','max_ball_norm','gradient_norm','lr_used','lr_after']
        if kind=='N1':fields+=list(describe_n1(dict(distance=np.array([1.,2.]),weight=np.array([.9,1.1]),bce=np.array([.2,.3]),positive_count=np.array([1,2]))))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,5):
            model.train();optimizer.zero_grad(set_to_none=True);sums={};order=[];norms=[];weights={k:[] for k in ['distance','weight','bce','positive_count']}
            for batch_index,batch in enumerate(training,1):
                lr=optimizer.param_groups[0]['lr'];r,v=update(model,kind,batch,optimizer,scheduler);step+=1;n=r['batch_size'];order.extend(r['ids']);norms.append(r['gradient_norm'])
                for k in ['loss','bce','weighted_bce','auxiliary','weighted_auxiliary']:sums[k]=sums.get(k,0.)+r[k]*n
                logged=dict(epoch=epoch,batch=batch_index,step=step,lr_used=lr,lr_after=optimizer.param_groups[0]['lr'],**{k:v for k,v in r.items() if k!='ids'})
                if kind=='N1':
                    logged.update(describe_n1(v))
                    for k in weights:weights[k].append(v[k])
                writer.writerow(logged)
                if batch_index==1:write(root/f'epoch{epoch}_first_batch.json',dict(ids=r['ids'],seed=seed))
                if batch_index==1912:need(n==11,'Final batch must have11 records')
                if batch_index%50==0 or batch_index==1912:f.flush();print(f'D2 seed={seed} model={kind} epoch={epoch}/4 batch={batch_index}/1912 loss={r["loss"]:.6f}',flush=True)
            need(len(order)==30587 and order==read(suite/'planned_orders'/f'seed{seed}_epoch{epoch}.json'),'Paired full train order mismatch')
            write(root/f'train_order_epoch{epoch}.json',order)
            if kind=='N1':
                vec={k:np.concatenate(v) for k,v in weights.items()};np.savez_compressed(root/f'train_weights_epoch{epoch}.npz',ids=np.array(order),**vec)
                for group,mask in [('all',np.ones(len(order),dtype=bool)),('single_positive',vec['positive_count']==1),('multi_positive',vec['positive_count']>1)]:
                    distribution.append(dict(epoch=epoch,group=group,count=int(mask.sum()),**describe_n1({k:v[mask] for k,v in vec.items()})))
                table(root/'N1_DISTRIBUTION.csv',distribution)
            val=predict(model,kind,tokenizer,data['validation'],seed,f'{kind} seed{seed} epoch{epoch} Validation')
            metrics=extended_score(val[1],probabilities(val[0]),manifest['names27'])[0]
            record=dict(epoch=epoch,step=step,training_examples=len(order),**{'training_'+k:v/len(order) for k,v in sums.items()},
                validation_micro_f1=metrics['micro_f1'],validation_macro_f1=metrics['macro_f1'],validation_rare_f1=metrics['rare_f1'],validation_macro_ap=metrics['macro_average_precision'],
                learning_rate=optimizer.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(np.max(norms)),gradient_clip_fraction=float(np.mean(np.array(norms)>1)))
            history.append(record);table(root/'training_log.csv',history);write(root/f'epoch{epoch}_validation_metrics.json',metrics)
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];torch.save(model.state_dict(),checkpoint/'model.pt');tokenizer.save_pretrained(checkpoint/'tokenizer');control(model,kind).encoder.config.save_pretrained(checkpoint/'encoder_config')
                save_validation(root,'validation_selected',*val[:3],manifest['names27']);write(root/'selected_loss_diagnostics.json',val[3])
                write(root/'checkpoint_selection.json',dict(epoch=epoch,step=step,validation_macro_f1=best,threshold=.30,tie_break='earliest',checkpoint_sha256=sha(checkpoint/'model.pt')))
            probe(root,f'epoch{epoch}',model,kind,tokenizer,data,selection)
            env.update(completed_epochs=epoch,completed_optimizer_steps=step);write(root/'environment.json',env);print('D2_EPOCH_RESULT',seed,kind,record,flush=True)
    need(step==7648 and optimizer.param_groups[0]['lr']==0,'Incomplete optimizer schedule')
    del model,optimizer,scheduler,training,val;release()
    seed_all(seed);model,tokenizer=build(kind,Path(plan['snapshot']),manifest['names27'])
    model.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer'),local_files_only=True,use_fast=True)
    after=predict(model,kind,tokenizer,data['validation'],seed,'Full3834 reload')
    with np.load(root/'validation_selected_predictions.npz',allow_pickle=False) as z:
        need(np.array_equal(z['logits'],after[0]) and np.array_equal(z['targets'],after[1]) and np.array_equal(z['ids'],after[2]),'Full reload not exact')
    for k,v in read(root/'selected_loss_diagnostics.json').items():need(after[3][k] is None if v is None else np.isclose(v,after[3][k],atol=1e-6,rtol=1e-6),'Reload loss diagnostics differ')
    save_validation(root,'validation',*after[:3],manifest['names27'])
    write(root/'checkpoint_reload_verification.json',dict(examples=3834,logits_exact=True,ids_equal=True,gold_equal=True,max_abs_logit_error=0.))
    if kind!='A0':need(np.array_equal(model.emotion_embeddings.cpu(),np.load(OLD['N1']/'fixed_emotion_embeddings.npy',allow_pickle=False)),'Label vectors changed')
    probe(root,'best_reload',model,kind,tokenizer,data,selection);del model;release();freeze_check(suite)
    write(root/'FROZEN_POSTCHECK.json',dict(unchanged=True,protected_manifest_sha256=sha(suite/'protected_before.json'),completed=now()))
    env.update(status='COMPLETE',new_test_inference_calls=0,end_time=now());write(root/'environment.json',env)
    verify_run(suite,seed,kind,sealed=False)
    write(root/'SUCCESS.json',dict(status='EXP6D2_CELL_COMPLETE',seed=seed,variant=kind,epochs=4,steps=7648,new_test_inference_calls=0,artifacts_sha256=artifacts(root),completed=now()))
    verify_run(suite,seed,kind)
