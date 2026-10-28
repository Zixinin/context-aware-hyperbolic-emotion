"""Fixed train-only eval/autograd probes, isolated from training RNG/mode/state."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from geometry_core import ScalingTrace,scale_summary,radial_decomposition
from diagnostic_core import gradient_statistics
from scommon import need,write,table,state_hash
from geometry_core import summary
from data27 import Collator

@contextmanager
def isolated_eval(model):
    """Also restore Python/NumPy RNG; no claim about unrecorded historical masks."""
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state().clone()
    cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    modes=[(m,m.training) for m in model.modules()]
    proof={}
    try:
        model.eval();yield proof
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
        for m,flag in modes:m.training=flag
        proof.update(python_rng_restored=random.getstate()==py,
            numpy_rng_restored=np.array_equal(np.random.get_state()[1],npstate[1]) and np.random.get_state()[2:]==npstate[2:],
            cpu_rng_restored=torch.equal(torch.get_rng_state(),cpu),
            cuda_rng_restored=all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())) if cuda else True,
            module_modes_restored=all(m.training==flag for m,flag in modes))
        need(all(proof.values()),'Probe failed to restore RNG or module modes')

def groups():
    def between(n,a,b):
        p='control.encoder.encoder.layer.'
        return n.startswith(p) and a<=int(n[len(p):].split('.')[0])<=b
    g={'shared_encoder':lambda n:n.startswith('control.encoder.'),
       'embedding':lambda n:n.startswith('control.encoder.embeddings.'),
       'early_layers_0_3':lambda n:between(n,0,3),'middle_layers_4_7':lambda n:between(n,4,7),
       'final_layers_8_11':lambda n:between(n,8,11),'classification_head':lambda n:n.startswith('control.classifier_'),
       'hyperbolic_projection':lambda n:n.startswith('hyper.encoder.'),'auxiliary_decoder':lambda n:n.startswith('hyper.decoder.')}
    for i in range(12):g[f'layer_{i}']=lambda n,i=i:between(n,i,i)
    return g

def run_probe(root,tag,model,tokenizer,data,selection):
    directory=root/'geometry'/tag;directory.mkdir(parents=True,exist_ok=False)
    named=list(model.named_parameters());params=[p for _,p in named]
    need(all(p.grad is None for p in params),'Probe must enter with .grad=None')
    before=state_hash(model.state_dict());by_id={r['id']:r for r in data['train']}
    device=next(model.parameters()).device
    samples=[];stages=[];gradients=[];losses=[];georows=[]
    with isolated_eval(model) as isolation:
        for cohort,rows in selection['cohorts'].items():
            records=[by_id[r['id']] for r in rows]
            batch=Collator(tokenizer)(records);ids=batch.pop('ids');x={k:v.to(device) for k,v in batch.items()}
            # Observe an ALREADY scaled model. The trace itself is s=1 identity;
            # it never multiplies the auxiliary input a second time.
            with ScalingTrace(model,1.,ids,cohort) as trace:out=model(**x)
            need(trace.initial_projected is not None,'Initial mapping was not observed')
            first=next(e for e in trace.events if e['module']=='encoder' and e['operation']=='proj')
            exp=next(e for e in trace.events if e['module']=='encoder' and e['operation']=='expmap0')
            raw_norm=trace.raw_h.detach().norm(dim=-1).cpu().numpy()
            rg=torch.autograd.grad(trace.initial_projected.norm(dim=-1).sum(),trace.raw_h,retain_graph=True)[0]
            gb=torch.autograd.grad(out['components']['classification'],params,retain_graph=True,allow_unused=True)
            ga=torch.autograd.grad(.1*out['components']['auxiliary'],params,allow_unused=True)
            gs=gradient_statistics(named,gb,ga,groups())
            for row in gs:
                need(row['num_parameters']>0,'Unexpected parameter layout')
                if row['module']=='classification_head':need(row['bce_norm']>0 and row['aux_connected_tensors']==0,'Classifier gradient path differs')
                elif row['module'] in ['hyperbolic_projection','auxiliary_decoder']:
                    need(row['bce_connected_tensors']==0 and row['weighted_aux_norm']>0,'Auxiliary gradient path differs')
                else:need(row['bce_norm']>0 and row['weighted_aux_norm']>0,'Shared gradient missing or zero')
                row.update(cohort=cohort,tag=tag)
            shared=next(r for r in gs if r['module']=='shared_encoder')
            derivative=radial_decomposition(trace.raw_h,rg)['radial'];radius=out['z'].detach().norm(dim=-1).cpu().numpy()
            for i,record_id in enumerate(ids):
                r=next(r for r in trace.rows if r['id']==record_id and r['stage']==first['stage'])
                ex=next(r for r in trace.rows if r['id']==record_id and r['stage']==exp['stage'])
                need(np.isclose(ex['norm_before'],.03*raw_norm[i],rtol=2e-6,atol=2e-7),'Scaling is missing, repeated or at wrong entry')
                later=[r for r in trace.rows if r['id']==record_id and r['role']=='text' and r['operation']=='proj' and r['stage']!=first['stage']]
                samples.append(dict(cohort=cohort,scale=.03,id=record_id,raw_h_norm=float(raw_norm[i]),
                    scaled_input_norm=ex['norm_before'],first_expmap_norm=ex['norm_after'],
                    initial_radius=r['norm_after'],initial_clipped=r['actual_proj_clipped'],
                    downstream_any_clipped=any(r['actual_proj_clipped'] for r in later),final_radius=float(radius[i]),
                    initial_radial_derivative=float(derivative[i])))
            p=out['components'];bce=float(p['classification'].detach());aux=float(p['auxiliary'].detach())
            losses.append(dict(cohort=cohort,tag=tag,bce=bce,auxiliary=aux,weighted_auxiliary=.1*aux,total=bce+.1*aux,
                weighted_aux_to_BCE_loss=.1*aux/bce,shared_aux_to_BCE_gradient_norm=shared['weighted_aux_to_bce_norm'],
                shared_gradient_cosine=shared['cosine']))
            st=summary(radius)
            georows.append(dict(cohort=cohort,tag=tag,raw_h_mean_norm=float(raw_norm.mean()),
                scaled_input_mean_norm=float(np.mean([r['scaled_input_norm'] for r in samples[-len(ids):]])),
                first_expmap_mean_norm=float(np.mean([r['first_expmap_norm'] for r in samples[-len(ids):]])),
                first_clip_fraction=float(np.mean([r['initial_clipped'] for r in samples[-len(ids):]])),
                later_clip_fraction=float(np.mean([r['downstream_any_clipped'] for r in samples[-len(ids):]])),
                final_mean_norm=st['mean'],final_min_norm=st['minimum'],final_max_norm=st['maximum'],
                final_norm_std=st['population_std'],final_norm_IQR=st['IQR'],final_fraction_above_099=float(np.mean(radius>.99)),
                shared_aux_to_BCE_gradient_norm=shared['weighted_aux_to_bce_norm'],shared_gradient_cosine=shared['cosine']))
            np.savez_compressed(directory/(cohort+'.npz'),ids=np.array(ids),targets=x['labels'].detach().cpu().numpy(),
                logits=out['logits'].detach().cpu().numpy(),z=out['z'].detach().cpu().numpy(),decoder_logits=out['decoder_logits'].detach().cpu().numpy())
            for row in trace.rows:row['scale']=.03
            stages+=trace.rows;gradients+=gs
            del out,gb,ga,gs,x,batch,trace,rg
    after=state_hash(model.state_dict())
    need(before==after and all(p.grad is None for p in params) and model.emotion_embeddings.grad is None,'Probe changed weights/buffers or accumulated gradients')
    table(directory/'samples.csv',samples);table(directory/'stages.csv',stages)
    table(directory/'gradients.csv',gradients);table(directory/'losses.csv',losses);table(directory/'cohorts.csv',georows)
    # Reuse the frozen extractor with only the actual s=.03 rows populated.
    # Its other scale slots are not needed; summarize unique IDs directly.
    unique={}
    for row in samples:unique.setdefault(row['id'],row)
    values=list(unique.values());radius=summary([r['final_radius'] for r in values])
    pooled=dict(tag=tag,unique_ids=len(values),cohort_positions=len(samples),
        first_clip_fraction=float(np.mean([r['initial_clipped'] for r in values])),
        later_clip_fraction=float(np.mean([r['downstream_any_clipped'] for r in values])),
        final_fraction_above_099=float(np.mean([r['final_radius']>.99 for r in values])),
        final_mean_norm=radius['mean'],final_min_norm=radius['minimum'],final_max_norm=radius['maximum'],
        final_norm_std=radius['population_std'],final_norm_IQR=radius['IQR'],
        median_shared_aux_to_BCE_gradient_norm=float(np.median([r['shared_aux_to_BCE_gradient_norm'] for r in losses])),
        median_shared_gradient_cosine=float(np.median([r['shared_gradient_cosine'] for r in losses])))
    write(directory/'NO_UPDATE_PROOF.json',dict(state_before=before,state_after=after,all_grads_none=True,
        optimizer_steps=0,split='train only',test_calls=0,scale_applied_by_model_once=True,trace_scale_is_identity=True,**isolation))
    write(directory/'pooled.json',pooled)
    print('GEOMETRY',tag,pooled,flush=True)
    return pooled,georows
