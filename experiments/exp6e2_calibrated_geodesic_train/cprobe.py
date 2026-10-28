"""Fixed train-only counterfactual gradients; never optimizer steps or Test."""
import math
from ccommon import *

PREFIXES=dict(shared_encoder='control.encoder.',classifier='control.classifier_',hyperbolic_projection='hyper.encoder.',auxiliary_decoder='hyper.decoder.')


def cpu_grad(loss,params,retain=True):
    grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    result=[None if g is None else g.detach().cpu() for g in grads]
    need(all(g is None or bool(torch.isfinite(g).all()) for g in result),'Nonfinite probe gradient')
    return result

def pair(a,b,indices):
    aa=bb=ab=combined=0.;ac=bc=0
    for i in indices:
        x=None if a[i] is None else a[i].double();y=None if b[i] is None else b[i].double()
        if x is not None:aa+=float(x.square().sum());ac+=1
        if y is not None:bb+=float(y.square().sum());bc+=1
        if x is not None and y is not None:ab+=float((x*y).sum());combined+=float((x+y).square().sum())
        elif x is not None:combined+=float(x.square().sum())
        elif y is not None:combined+=float(y.square().sum())
    return dict(a_norm=aa**.5,b_norm=bb**.5,cosine=ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,
        sum_norm=combined**.5,a_connected=ac,b_connected=bc)

def run_probe(root,tag,model,tokenizer,data,selection,gate=True):
    directory=root/'probes'/tag;directory.mkdir(parents=True,exist_ok=False)
    named=list(model.named_parameters());params=[p for _,p in named];before=state_hash(model.state_dict())
    groups={key:[i for i,(n,p) in enumerate(named) if n.startswith(prefix)] for key,prefix in PREFIXES.items()}
    need(sum(map(len,groups.values()))==len(params) and all(groups.values()),'Unexpected module parameter layout')
    need(all(p.grad is None for p in params),'Probe requires empty .grad')
    by_id={r['id']:r for r in data['train']};gradrows=[];lossrows=[];samples=[];stages=[];severe=[]
    with isolated_eval(model) as isolation:
        for cohort,records in selection['cohorts'].items():
            print('GEODESIC_NO_UPDATE_PROBE',tag,cohort,flush=True)
            batch=Collator(tokenizer)([by_id[r['id']] for r in records]);ids=batch.pop('ids');x={k:v.to('cuda') for k,v in batch.items()}
            with ScalingTrace(model,1.,ids,cohort) as trace:out=model(**x)
            c=out['components'];need(bool(torch.isfinite(out['loss'])),'Nonfinite probe loss')
            old_total=c['classification']+.1*c['coordinate_auxiliary']
            gb=cpu_grad(c['classification'],params);go=cpu_grad(.1*c['coordinate_auxiliary'],params)
            gn=cpu_grad(GAMMA*c['auxiliary'],params)
            gc=cpu_grad(GAMMA*c['positive_ce'].mean(),params);gd=cpu_grad(GAMMA*c['positive_distance'].mean(),params)
            gt=cpu_grad(out['loss'],params,retain=False)
            old2=new2=0.
            for group,indices in groups.items():
                old=pair(gb,go,indices);new=pair(gb,gn,indices);change=pair(go,gn,indices);total=pair(gt,gt,indices)
                pieces=pair(gc,gd,indices);bc=pair(gb,gc,indices);bd=pair(gb,gd,indices)
                need(abs(pieces['sum_norm']-new['b_norm'])<2e-5*max(1.,new['b_norm']),'Aux CE+distance gradients do not sum')
                if group=='classifier':need(old['a_norm']>0 and old['b_connected']==new['b_connected']==0,'Classifier gradient path changed')
                elif group in ['hyperbolic_projection','auxiliary_decoder']:need(old['a_connected']==new['a_connected']==0 and old['b_norm']>0 and new['b_norm']>0,'Auxiliary module gradient path wrong')
                else:need(old['a_norm']>0 and old['b_norm']>0 and new['b_norm']>0,'Shared encoder auxiliary gradient missing')
                need(abs(new['sum_norm']-total['a_norm'])<1e-5*max(1.,total['a_norm']),'Total gradient not BCE+weighted geodesic auxiliary')
                gradrows.append(dict(tag=tag,cohort=cohort,module=group,bce_norm=old['a_norm'],coordinate_aux_norm=old['b_norm'],
                    geodesic_aux_norm=new['b_norm'],coordinate_aux_to_bce=old['b_norm']/old['a_norm'] if old['a_norm'] else None,
                    geodesic_aux_to_bce=new['b_norm']/new['a_norm'] if new['a_norm'] else None,
                    geodesic_to_coordinate_aux_norm=new['b_norm']/old['b_norm'] if old['b_norm'] else None,
                    bce_coordinate_aux_cosine=old['cosine'],bce_geodesic_aux_cosine=new['cosine'],old_new_aux_cosine=change['cosine'],
                    coordinate_total_norm=old['sum_norm'],geodesic_total_norm=new['sum_norm'],
                    weighted_ce_norm=pieces['a_norm'],weighted_distance_norm=pieces['b_norm'],ce_distance_cosine=pieces['cosine'],
                    bce_ce_cosine=bc['cosine'],bce_distance_cosine=bd['cosine'],
                    bce_connected=old['a_connected'],coordinate_aux_connected=old['b_connected'],geodesic_aux_connected=new['b_connected']))
                old2+=old['sum_norm']**2;new2+=new['sum_norm']**2
            values=diagnostic_values(out);loss_ratio=GAMMA*abs(values['auxiliary'])/max(.1*abs(values['coordinate_auxiliary_same_forward']),1e-12)
            norm_ratio=math.sqrt(new2/old2);flag=loss_ratio>SEVERE_RATIO or norm_ratio>SEVERE_RATIO
            lossrows.append(dict(tag=tag,cohort=cohort,**values,coordinate_total=float(old_total.detach()),weighted_geodesic_to_coordinate_aux_absolute_scale=loss_ratio,
                old_total_gradient_norm=math.sqrt(old2),new_total_gradient_norm=math.sqrt(new2),total_gradient_norm_ratio=norm_ratio,severe_gate_triggered=flag))
            if flag:severe.append(dict(cohort=cohort,aux_scale_ratio=loss_ratio,total_gradient_norm_ratio=norm_ratio))
            first=next(e for e in trace.events if e['module']=='encoder' and e['operation']=='proj')
            exp=next(e for e in trace.events if e['module']=='encoder' and e['operation']=='expmap0')
            raw_norm=trace.raw_h.detach().norm(dim=-1).cpu().numpy();radii=out['z'].detach().norm(dim=-1).cpu().numpy()
            for i,rid in enumerate(ids):
                ex=next(r for r in trace.rows if r['id']==rid and r['stage']==exp['stage'])
                need(np.isclose(ex['norm_before'],raw_norm[i],rtol=2e-6,atol=2e-7),'Original A1 input scale1 changed')
                firstrow=next(r for r in trace.rows if r['id']==rid and r['stage']==first['stage'])
                later=[r for r in trace.rows if r['id']==rid and r['role']=='text' and r['operation']=='proj' and r['stage']!=first['stage']]
                samples.append(dict(cohort=cohort,id=rid,raw_h_norm=float(raw_norm[i]),first_clip=firstrow['actual_proj_clipped'],
                    later_clip=any(r['actual_proj_clipped'] for r in later),final_norm=float(radii[i]),positive_labels=int(x['labels'][i].sum()),
                    positive_distance=float(c['positive_distance'][i].detach())))
            np.savez_compressed(directory/(cohort+'.npz'),ids=np.array(ids),targets=x['labels'].detach().cpu().numpy(),
                **{k:out[k].detach().cpu().numpy() for k in ['logits','z','decoder_logits']})
            stages+=trace.rows
            del gb,go,gn,gc,gd,gt,out,c,x,batch,trace,old_total;release()
    after=state_hash(model.state_dict());need(before==after and all(p.grad is None for p in params) and model.emotion_embeddings.grad is None,'No-update probe changed state or gradients')
    table(directory/'MODULE_GRADIENTS.csv',gradrows);table(directory/'LOSS_SCALES.csv',lossrows);table(directory/'SAMPLES.csv',samples);table(directory/'STAGES.csv',stages)
    unique={}
    for r in samples:unique.setdefault(r['id'],r)
    u=list(unique.values())
    pooled=dict(tag=tag,unique_train_ids=len(u),positions=len(samples),first_clip_fraction=float(np.mean([r['first_clip'] for r in u])),
        later_clip_fraction=float(np.mean([r['later_clip'] for r in u])),final_fraction_above_099=float(np.mean([r['final_norm']>.99 for r in u])),
        mean_final_norm=float(np.mean([r['final_norm'] for r in u])),max_final_norm=float(max(r['final_norm'] for r in u)),
        max_weighted_aux_absolute_scale_ratio=max(r['weighted_geodesic_to_coordinate_aux_absolute_scale'] for r in lossrows),
        max_total_gradient_norm_ratio=max(r['total_gradient_norm_ratio'] for r in lossrows),severe_findings=severe)
    write(directory/'SUMMARY.json',pooled);write(directory/'NO_UPDATE_PROOF.json',dict(state_before=before,state_after=after,
        optimizer_steps=0,test_forward_calls=0,all_grads_none=True,**isolation))
    if gate and severe:
        write(root/'PRETRAIN_NUMERICAL_STOP.json',dict(severe=severe,gamma_G=GAMMA,optimizer_steps=0,no_automatic_change=True))
        raise RuntimeError('Calibrated pretraining weighted loss/total gradient >10x; stop without changing gamma')
    print('GEODESIC_PROBE_SUMMARY',pooled,flush=True);return pooled

def json_text(v):
    import json
    return json.dumps(v,indent=2,ensure_ascii=False)
