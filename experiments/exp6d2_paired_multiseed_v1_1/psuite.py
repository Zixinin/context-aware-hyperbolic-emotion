"""Serial six-cell suite, preflight/smoke gates, conservative resume, no Test."""
import argparse,gc,shutil,sys,traceback
import numpy as np
import torch
from pcommon import *
from ptrain import train_one,verify_run,probe,predict,update,release

def prepare(suite):
    need(suite.parent==BASE.resolve() and not (suite/'SUITE_PLAN.json').exists(),'Fresh suite required')
    source=source_hashes();old_seals();recovery=recovery_inputs();data,manifest=dataset();env=runtime()
    snapshot=Path(read(OLD['A1S']/'plan.json')['snapshot'])
    need(a0.snapshot_inventory(snapshot)==read(OLD['A1S']/'plan.json')['snapshot_files_sha256'],'Raw model/tokenizer changed')
    need(shutil.disk_usage(BASE).free>=12*1024**3,'Need at least12GiB free for six checkpoints/smoke/logs; no cleanup performed')
    shutil.copytree(HERE,suite/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(suite/'RECOVERY_INPUTS_SHA256.json',recovery)
    write(suite/'protected_before.json',protected_inventory(snapshot));write(suite/'dataset_manifest.json',manifest)
    selection=read(OLD['A1S']/'FIXED_TRAIN_COHORTS.json');need(selection==read(OLD['N1']/'FIXED_TRAIN_COHORTS.json'),'Fixed geometry cohorts differ')
    write(suite/'FIXED_TRAIN_COHORTS.json',selection);historical_seed42(suite,data,manifest)
    orders=suite/'planned_orders';orders.mkdir();ids=[r['id'] for r in data['train']]
    seed42_orders={}
    for seed in ALL_SEEDS:
        for epoch,order in enumerate(planned_orders(ids,seed),1):
            need(len(order)==len(set(order))==30587,'Invalid planned order')
            if seed==42:seed42_orders[epoch]=order
            write(orders/f'seed{seed}_epoch{epoch}.json',order)
    replay=frozen_seed42_order_replay(data['train'])
    need(replay==[seed42_orders[e] for e in range(1,5)],'Frozen A0 DataLoader replay differs from new seed42 planner')
    order_audit=historical_order_audit(seed42_orders)
    order_audit['frozen_A0_loader_full_order_replay_matches_planner']=True
    write(suite/'SEED42_ORDER_AUDIT.json',order_audit)
    write(suite/'SUITE_PLAN.json',dict(created=now(),new_seeds=list(NEW_SEEDS),all_seeds=list(ALL_SEEDS),models=list(KINDS),
        execution_order=[dict(seed=s,variant=k) for s in NEW_SEEDS for k in KINDS],new_training_cells=6,reuse_seed42=True,
        source_manifest=source,runtime=env,snapshot=str(snapshot),snapshot_sha256=a0.snapshot_inventory(snapshot),
        dataset_manifest_sha256=sha(suite/'dataset_manifest.json'),historical_provenance={k:sha(suite/'seed42_reused'/(k+'_PROVENANCE.json')) for k in KINDS},
        epochs=4,total_steps=7648,warmup_steps=765,train_batch=16,eval_batch=64,threshold=.30,
        selection='Validation Macro-F1 earliest tie',data_generator='per-seed Torch Generator advances four epochs; A1S/N1 saved full seed42 orders checked; A0 saved first batches plus frozen loader reconstruction',
        seed42_order_audit_sha256=sha(suite/'SEED42_ORDER_AUDIT.json'),recovery_reference=read(HERE/'RECOVERY_REFERENCE.json'),
        new_test_inference=False,no_raw_F=True,no_euclidean_control=True,paired_primary='N1-A1S Macro-F1',paired_secondary='N1-A0 Macro-F1',
        reporting='n=3 seeds, mean and sample SD ddof1; paired per-seed deltas; no significance/generalization claim'))
    print('EXP6D2_PLAN_FROZEN_SEED42_REUSED_SIX_NEW_CELLS',flush=True)

def smoke(suite,only_seed=None):
    from transformers import get_linear_schedule_with_warmup
    plan=read(suite/'SUITE_PLAN.json');need(not (suite/'SMOKE_SUCCESS.json').exists(),'Smoke already sealed; use resume')
    need(only_seed in NEW_SEEDS,'Run one smoke seed per correctly seeded process')
    need(os.environ.get('PYTHONHASHSEED')==str(only_seed),'Smoke subprocess hash seed differs')
    freeze_check(suite);env=runtime()
    for key in ['host','python','executable','packages','pip_freeze','gpu','cuda_runtime']:need(env[key]==plan['runtime'][key],'Smoke runtime differs')
    data,manifest=dataset();selection=read(suite/'FIXED_TRAIN_COHORTS.json');directory=suite/'smoke';directory.mkdir(exist_ok=True)
    for seed in [only_seed]:
        seed_dir=directory/f'seed{seed}'
        if (seed_dir/'SUCCESS.json').exists():
            need(read(seed_dir/'SUCCESS.json')['artifacts_sha256']==artifacts(seed_dir),'Completed seed smoke changed');continue
        seed_dir.mkdir();initial={};ref_logits={};ref_hyper={};ref_groups=None
        ids=read(suite/'planned_orders'/f'seed{seed}_epoch1.json')[:16];by_id={r['id']:r for r in data['train']}
        for kind in KINDS:
            print('D2_PAIRED_INITIALIZATION_SMOKE',seed,kind,flush=True)
            seed_all(seed);model,tokenizer=build(kind,Path(plan['snapshot']),manifest['names27']);model.to('cuda')
            shared=state_hash(control(model,kind).state_dict());hyper=None if kind=='A0' else state_hash(model.hyper.state_dict())
            if initial:need(shared==initial['A0']['shared_state'],'Paired shared initialization differs')
            if kind=='N1':need(hyper==initial['A1S']['hyper_state'],'A1S/N1 hyper initial parameters differ')
            batch=Collator(tokenizer)([by_id[r] for r in ids]);batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
            if kind=='A0':base_cpu=cpu.clone();base_cuda=cuda.clone()
            else:need(torch.equal(cpu,base_cpu) and torch.equal(cuda,base_cuda),'Branch constructor perturbed shared RNG')
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                for training in [True,False]:
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);model.train(training);out=model(**x)
                    key='train' if training else 'eval';logits=out['logits'].cpu()
                    if kind=='A0':ref_logits[key]=logits
                    else:need(torch.equal(logits,ref_logits[key]),'Paired initial classification logits differ')
                    if kind=='A1S':ref_hyper[key]={k:out[k].cpu() for k in ['z','decoder_logits']}
                    if kind=='N1':
                        for k in ['z','decoder_logits']:need(torch.equal(out[k].cpu(),ref_hyper[key][k]),'A1S/N1 initial branch forward differs')
            optimizer,groups=optimizer_for(model)
            shared_groups=[dict(weight_decay=g['weight_decay'],names=[n.removeprefix('control.') for n in g['parameter_names'] if kind=='A0' or n.startswith('control.')]) for g in groups]
            if kind=='A0':ref_groups=shared_groups
            else:need(shared_groups==ref_groups,'Shared optimizer grouping differs')
            initial[kind]=dict(seed=seed,shared_state=shared,hyper_state=hyper,paired_initial_train_eval_logits_exact=True,
                initial_training_logits_sha256=__import__('hashlib').sha256(ref_logits['train'].numpy().tobytes()).hexdigest(),fresh_pretrained=True)
            del optimizer,model,x,batch,out;release()
        write(seed_dir/'INITIALIZATION.json',initial)
        for kind in KINDS:
            print('D2_MODEL_SMOKE',seed,kind,flush=True)
            root=seed_dir/kind;root.mkdir();seed_all(seed);model,tokenizer=build(kind,Path(plan['snapshot']),manifest['names27']);model.to('cuda')
            need(state_hash(control(model,kind).state_dict())==initial[kind]['shared_state'],'Smoke initialization changed')
            probe(root,'initial',model,kind,tokenizer,data,selection)
            optimizer,groups=optimizer_for(model);scheduler=get_linear_schedule_with_warmup(optimizer,765,7648);model.train()
            iterator=iter(make_loader(data['train'],tokenizer,seed,True));steps=[]
            for i in range(2):
                lr=optimizer.param_groups[0]['lr'];r,v=update(model,kind,next(iterator),optimizer,scheduler)
                if i==0:need(r['ids']==ids,'Dedicated per-seed loader differs from preregistered order')
                steps.append(dict(r,lr_used=lr,lr_after=optimizer.param_groups[0]['lr']))
            need(steps[0]['lr_used']==0 and steps[1]['lr_used']>0,'Smoke scheduler differs')
            before=predict(model,kind,tokenizer,data['validation'][:64],seed,'smoke64 Validation')
            torch.save(model.state_dict(),root/'weights.pt');del model,optimizer,scheduler,iterator;release()
            seed_all(seed);model,tokenizer=build(kind,Path(plan['snapshot']),manifest['names27'])
            model.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True),strict=True);model.to('cuda')
            after=predict(model,kind,tokenizer,data['validation'][:64],seed,'smoke64 reload')
            need(all(np.array_equal(a,b) for a,b in zip(before[:3],after[:3])),'Smoke logits/IDs/gold reload not exact')
            write(root/'SMOKE.json',dict(seed=seed,variant=kind,discarded_smoke_updates=steps,reload_examples=64,reload_logits_exact=True,new_test=False))
            del model;release()
        write(seed_dir/'SUCCESS.json',dict(seed=seed,status='PAIRED_SEED_SMOKE_PASS',artifacts_sha256=artifacts(seed_dir)))
    if not all((directory/f'seed{s}'/'SUCCESS.json').exists() for s in NEW_SEEDS):
        print('EXP6D2_ONE_SEED_SMOKE_PASS',only_seed,'remaining smoke seed pending',flush=True);return
    for seed in NEW_SEEDS:
        need(read(directory/f'seed{seed}'/'SUCCESS.json')['artifacts_sha256']==artifacts(directory/f'seed{seed}'),'Seed smoke seal differs')
    freeze_check(suite)
    write(suite/'SMOKE_SUCCESS.json',dict(status='EXP6D2_ALL_SIX_SMOKES_PASS',cells=6,source_manifest=source_hashes(),
        plan_sha256=sha(suite/'SUITE_PLAN.json'),artifacts_sha256=artifacts(directory),completed=now()))
    print('EXP6D2_ALL_SIX_SMOKES_PASS',flush=True)

def verify_smoke(suite):
    p=read(suite/'SMOKE_SUCCESS.json')
    need(p['status']=='EXP6D2_ALL_SIX_SMOKES_PASS' and p['artifacts_sha256']==artifacts(suite/'smoke'),'Smoke changed or incomplete')
    need(p['plan_sha256']==sha(suite/'SUITE_PLAN.json') and p['source_manifest']==source_hashes(),'Plan/source changed since smoke')

def summarize(suite):
    from preport import write_reports
    verify_smoke(suite)
    for seed in NEW_SEEDS:
        for kind in KINDS:verify_run(suite,seed,kind)
    freeze_check(suite);write_reports(suite)
    write(suite/'FROZEN_POSTCHECK.json',dict(unchanged=True,manifest_sha256=sha(suite/'protected_before.json'),completed=now()))
    write(suite/'SUCCESS.json',dict(status=STATUS,new_training_cells=6,reused_seed42_cells=3,seeds=list(ALL_SEEDS),new_test_inference_calls=0,
        artifacts_sha256=artifacts(suite),completed=now()))
    verify(suite);print((suite/'EXP6D2_SUMMARY_CN.md').read_text(encoding='utf-8'),flush=True)

def verify(suite):
    p=read(suite/'SUCCESS.json');need(p['status']==STATUS and p['artifacts_sha256']==artifacts(suite),'Suite seal differs')
    verify_smoke(suite)
    for seed in NEW_SEEDS:
        for kind in KINDS:verify_run(suite,seed,kind)
    need(read(suite/'FROZEN_POSTCHECK.json')['unchanged'],'No final freeze proof')
    need(len(csvrows(suite/'EXP6D2_NINE_RUN_VALIDATION.csv'))==9,'Need all nine seed/model cells')
    print('EXP6D2_SUITE_SAVED_ARTIFACT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','smoke','train-one','summarize','verify']);p.add_argument('--suite',required=True,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--variant',choices=KINDS);a=p.parse_args();suite=a.suite.resolve()
    try:
        if a.phase=='train-one':verify_smoke(suite);train_one(suite,a.seed,a.variant)
        elif a.phase=='smoke':smoke(suite,a.seed)
        else:{'prepare':prepare,'summarize':summarize,'verify':verify}[a.phase](suite)
    except BaseException as e:
        if a.phase!='verify':
            failure=suite/f'FAILURE_{a.phase}_{a.seed}_{a.variant}_{__import__("time").time_ns()}.json'
            write(failure,dict(phase=a.phase,seed=a.seed,variant=a.variant,error=repr(e),traceback=traceback.format_exc(),new_test=False,no_automatic_retry=True))
        raise
