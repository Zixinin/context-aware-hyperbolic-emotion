"""Fixed train-only geometry and four separate parameter-gradient measurements."""
import gc
import numpy as np
import torch
from ncommon import *
from nmodel import values,vectors,describe
from probes import isolated_eval
from geometry_core import ScalingTrace,summary
from data27 import Collator

def cpu_grad(loss,params,retain=True):
    g=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    result=[]
    for x in g:
        need(x is None or bool(torch.isfinite(x).all()),'Nonfinite parameter gradient')
        result.append(None if x is None else x.detach().cpu())
    return result

def pair(a,b,indices):
    aa=bb=ab=0.;na=nb=0
    for i in indices:
        x,y=a[i],b[i]
        if x is not None:xx=x.double();aa+=float(xx.square().sum());na+=1
        if y is not None:yy=y.double();bb+=float(yy.square().sum());nb+=1
        if x is not None and y is not None:ab+=float((xx*yy).sum())
    return dict(a_norm=aa**.5,b_norm=bb**.5,a_connected=na,b_connected=nb,
        cosine=ab/(aa*bb)**.5 if aa>0 and bb>0 else None,sum_norm=max(0.,aa+bb+2*ab)**.5)

def formula_check(out,y,e,reference):
    c=out['components'];r=reference.candidates(out['logits'],y,out['z'],out['decoder_logits'],e)
    for new,old in [('positive_distance','dbar'),('normalized_weight','N_weight'),('ordinary_bce','bce'),('weighted_bce','N_classification'),('auxiliary','aux'),('total','N')]:
        need(torch.allclose(c[new],r[old],atol=1e-7,rtol=1e-6),'Mismatch with completed Exp6D N1 audit: '+new)
    need(not c['normalized_weight'].requires_grad and c['normalized_weight'].grad_fn is None,'Weights not detached')
    leak=torch.autograd.grad(c['weighted_bce'],[out['z'],out['decoder_logits']],allow_unused=True,retain_graph=True)
    need(all(v is None for v in leak),'Distance weighting backpropagates to hyperbolic branch')
    dg=torch.autograd.grad(c['weighted_bce'],out['logits'],retain_graph=True)[0]
    expected=c['normalized_weight'][:,None]*(out['logits'].sigmoid()-y)/(len(y)*27)
    need(torch.allclose(dg,expected,atol=1e-8,rtol=1e-5),'Independent multi-hot BCE derivative failed')
    need(bool((dg[y==1]<=0).all() and (dg[y==0]>=0).all()),'An additional positive became negative')
    for i in range(len(y)):
        need(torch.allclose(c['positive_distance'][i],c['distance_matrix'][i,y[i]==1].mean(),atol=1e-10,rtol=1e-10),'Positive-label mean differs')
    return dict(reference_formula=True,all_positives_included=True,weights_detached=True,
        BCE_target_derivative=True,weight_mean=float(c['normalized_weight'].mean()),**values(out))

def run_probe(root,tag,model,tokenizer,data,selection,replay=False):
    directory=root/'geometry'/tag;directory.mkdir(parents=True,exist_ok=False)
    named=list(model.named_parameters());params=[p for _,p in named]
    groups={key:[i for i,(n,p) in enumerate(named) if n.startswith(prefix)] for key,prefix in {
        'shared_encoder':'control.encoder.','classification_head':'control.classifier_',
        'hyperbolic_projection':'hyper.encoder.','auxiliary_decoder':'hyper.decoder.'}.items()}
    need(sum(map(len,groups.values()))==len(params) and all(groups.values()),'Unexpected module layout')
    need(all(p.grad is None for p in params),'Probe entered with accumulated gradients')
    before=state_hash(model.state_dict());by_id={r['id']:r for r in data['train']};reference=audit_reference()
    samples=[];stages=[];gradient_rows=[];losses=[];cases=[];replay_checks=[]
    old_grad=csvrows(D/'MODULE_GRADIENTS.csv') if replay else []
    with isolated_eval(model) as isolation:
        for cohort,records in selection['cohorts'].items():
            print('N1_NO_UPDATE_PROBE',tag,cohort,flush=True)
            batch=Collator(tokenizer)([by_id[r['id']] for r in records]);ids=batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
            with ScalingTrace(model,1.,ids,cohort) as trace:out=model(**x)
            c=out['components'];case=formula_check(out,x['labels'],model.emotion_embeddings,reference);case.update(cohort=cohort);cases.append(case)
            if replay:
                with np.load(D/'numeric/A1S'/(cohort+'.npz'),allow_pickle=False) as old:
                    need(old['ids'].tolist()==ids and np.array_equal(old['targets'],x['labels'].cpu()),'Exp6D replay IDs/gold differ')
                    for k in ['logits','z','decoder_logits']:
                        need(np.allclose(old[k],out[k].detach().cpu().numpy(),rtol=1e-6,atol=1e-6 if k=='z' else 1e-5),'Exp6D frozen forward replay differs')
            gb=cpu_grad(c['ordinary_bce'],params);gn=cpu_grad(c['weighted_bce'],params)
            ga=cpu_grad(.1*c['auxiliary'],params);gt=cpu_grad(c['total'],params,retain=False)
            for i,p in enumerate(params):
                expected=(gn[i] if gn[i] is not None else torch.zeros_like(gt[i])) if gt[i] is not None else None
                if ga[i] is not None:expected=expected+ga[i]
                need(expected is not None and gt[i] is not None and torch.allclose(expected,gt[i],atol=1e-6,rtol=1e-5),'Combined gradient is not N1+0.1aux')
            global2=0.;base2=0.
            for group,indices in groups.items():
                b_n=pair(gb,gn,indices);n_a=pair(gn,ga,indices);t=pair(gt,gt,indices);b_a=pair(gb,ga,indices)
                if group=='classification_head':need(b_n['a_norm']>0 and n_a['b_connected']==0,'Classifier auxiliary path differs')
                elif group in ['hyperbolic_projection','auxiliary_decoder']:need(b_n['a_connected']==b_n['b_connected']==0 and n_a['b_norm']>0,'Hyperbolic classification path differs')
                else:need(b_n['a_norm']>0 and b_n['b_norm']>0 and n_a['b_norm']>0,'Shared gradient missing')
                ratio=b_n['b_norm']/b_n['a_norm'] if b_n['a_norm'] else None
                need(ratio is None or ratio<10.,'Abnormal >=10x N1 classification gradient amplification; no formula changes')
                r=dict(tag=tag,cohort=cohort,module=group,ordinary_BCE_norm=b_n['a_norm'],N1_class_norm=b_n['b_norm'],
                    weighted_aux_norm=n_a['b_norm'],combined_norm=t['a_norm'],N1_to_BCE_norm=ratio,
                    N1_vs_BCE_cosine=b_n['cosine'],N1_vs_aux_cosine=n_a['cosine'],
                    BCE_connected=b_n['a_connected'],N1_connected=b_n['b_connected'],aux_connected=n_a['b_connected'])
                gradient_rows.append(r);global2+=t['a_norm']**2;base2+=b_a['sum_norm']**2
                if replay:
                    old=next(v for v in old_grad if v['model']=='A1S' and v['option']=='N' and v['cohort']==cohort and v['module']==group)
                    for key,oldkey in [('N1_class_norm','weighted_class_norm'),('ordinary_BCE_norm','ordinary_BCE_norm'),('weighted_aux_norm','weighted_aux_norm'),('combined_norm','combined_norm')]:
                        need(np.isclose(r[key],float(old[oldkey]),atol=1e-7,rtol=1e-5),'Exp6D actual module gradient replay differs: '+key)
            first=next(ev for ev in trace.events if ev['module']=='encoder' and ev['operation']=='proj')
            exp=next(ev for ev in trace.events if ev['module']=='encoder' and ev['operation']=='expmap0')
            raw_norm=trace.raw_h.detach().norm(dim=-1).cpu().numpy();radii=out['z'].detach().norm(dim=-1).cpu().numpy();v=vectors(out)
            for i,rid in enumerate(ids):
                f=next(r for r in trace.rows if r['id']==rid and r['stage']==first['stage'])
                ex=next(r for r in trace.rows if r['id']==rid and r['stage']==exp['stage'])
                need(np.isclose(ex['norm_before'],.03*raw_norm[i],rtol=2e-6,atol=2e-7),'Input scale not exactly once at .03')
                later=[r for r in trace.rows if r['id']==rid and r['role']=='text' and r['operation']=='proj' and r['stage']!=first['stage']]
                samples.append(dict(tag=tag,cohort=cohort,id=rid,raw_h_norm=float(raw_norm[i]),initial_clipped=f['actual_proj_clipped'],
                    downstream_any_clipped=any(r['actual_proj_clipped'] for r in later),final_radius=float(radii[i]),
                    **{k:float(a[i]) for k,a in v.items()}))
            stages+=trace.rows
            shared=gradient_rows[-4]
            lossrow=dict(tag=tag,cohort=cohort,**values(out),
                shared_N1_to_BCE_gradient_norm=shared['N1_to_BCE_norm'],
                shared_aux_to_N1_gradient_norm=shared['weighted_aux_norm']/shared['N1_class_norm'],
                shared_N1_aux_cosine=shared['N1_vs_aux_cosine'],combined_gradient_norm=global2**.5,
                baseline_combined_gradient_norm=base2**.5,would_clip_at_1=global2>1)
            lossrow.update(describe(v));losses.append(lossrow)
            np.savez_compressed(directory/(cohort+'.npz'),ids=np.array(ids),targets=x['labels'].detach().cpu().numpy(),
                **{k:out[k].detach().cpu().numpy() for k in ['logits','z','decoder_logits']},**v)
            del out,c,gb,gn,ga,gt,x,batch,trace;gc.collect()
    after=state_hash(model.state_dict())
    need(before==after and all(p.grad is None for p in params) and model.emotion_embeddings.grad is None,'Probe mutated state/grad')
    unique={}
    for r in samples:unique.setdefault(r['id'],r)
    u=list(unique.values());radius=summary([r['final_radius'] for r in u])
    pooled=dict(tag=tag,unique_ids=len(u),cohort_positions=len(samples),
        first_clip_fraction=float(np.mean([r['initial_clipped'] for r in u])),later_clip_fraction=float(np.mean([r['downstream_any_clipped'] for r in u])),
        final_fraction_above_099=float(np.mean([r['final_radius']>.99 for r in u])),final_mean_norm=radius['mean'],
        final_min_norm=radius['minimum'],final_max_norm=radius['maximum'],final_norm_std=radius['population_std'],final_norm_IQR=radius['IQR'],
        median_shared_N1_to_BCE_gradient_norm=float(np.median([r['shared_N1_to_BCE_gradient_norm'] for r in losses])),
        median_shared_aux_to_N1_gradient_norm=float(np.median([r['shared_aux_to_N1_gradient_norm'] for r in losses])),
        median_shared_N1_aux_cosine=float(np.median([r['shared_N1_aux_cosine'] for r in losses])))
    for name,rows in [('samples',samples),('stages',stages),('gradients',gradient_rows),('losses',losses)]:table(directory/(name+'.csv'),rows)
    write(directory/'FORMULA_CHECKS.json',cases);write(directory/'pooled.json',pooled)
    write(directory/'NO_UPDATE_PROOF.json',dict(state_before=before,state_after=after,all_grads_none=True,optimizer_steps=0,test_calls=0,
        scale_applied_once=True,four_parameter_gradient_components_measured=True,reference='completed Exp6D N1',**isolation))
    if replay:write(directory/'EXP6D_REPLAY_PASS.json',dict(checkpoint='A1S best used only for no-update replay',all_10_cohorts=True,values_match=True,all_4_modules_gradients_match=True))
    print('GEOMETRY_N1',pooled,flush=True)
    return pooled
