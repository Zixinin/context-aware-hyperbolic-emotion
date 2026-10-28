"""Validation-only comparison with frozen original A0/A1, no test-driven tuning."""
from ccommon import *

def report(root,manifest):
    allrows=[];aggregate={};historical=root/'historical_validation';historical.mkdir()
    for tag,path in [('A0',A0),('A1',A1),('A1-G-C',root)]:
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['label_names'].tolist()==manifest['names27'] and z['ids'].tolist()==manifest['splits']['validation']['retained_ids'],'Unpaired comparison arrays')
            m,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
            need(all(m[k]==v for k,v in read(path/'validation_metrics.json').items()),'Historical Validation metric mismatch')
        aggregate[tag]=m
        if tag!='A1-G-C':write(historical/(tag+'_metrics.json'),m);table(historical/(tag+'_per_label.csv'),rows)
        for row in rows:allrows.append(dict(model=tag,split='validation',seed=42,**row))
    table(root/'PER_LABEL_COMPARISON.csv',allrows)
    compact=[dict(model=tag,alignment={'A0':'none','A1':'negative coordinate inner product','A1-G-C':'positive unsquared geodesic distance'}[tag],
        split='validation',seed=42,auxiliary_coefficient={'A0':0.,'A1':.1,'A1-G-C':GAMMA}[tag],
        **{k:m[k] for k in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']}) for tag,m in aggregate.items()]
    table(root/'VALIDATION_COMPARISON.csv',compact)
    by={(r['model'],r['label']):r for r in allrows};deltas=[]
    for label in manifest['names27']:
        a,g=by['A1',label],by['A1-G-C',label]
        deltas.append(dict(label=label,support=g['support'],precision_delta_pp=100*(g['precision']-a['precision']),recall_delta_pp=100*(g['recall']-a['recall']),
            f1_delta_pp=100*(g['f1']-a['f1']),AP_delta_pp=100*(g['average_precision']-a['average_precision']),macro_contribution_pp=100*(g['f1']-a['f1'])/27))
    table(root/'PER_LABEL_GEODESIC_MINUS_COORDINATE.csv',deltas)
    rows=['# A1-G-C：校准系数后的测地对齐 seed42 Validation结果','',
        f'本轮使用测地距离对齐，并将整个辅助项系数固定为校准值gamma_G={GAMMA!r}；普通BCE、CE内部公式、固定27向量、s=1与其余训练设置保持。辅助CE的实际系数也改变，因此不是仅替换内积的单变量消融。新Test=0。','',
        '|模型|Alignment|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|','|---|---|---:|---:|---:|---:|']
    for r in compact:rows.append(f'|{r["model"]}|{r["alignment"]}|'+ '|'.join(f'{100*r[k]:.4f}' for k in ['micro_f1','macro_f1','rare_f1','macro_average_precision'])+'|')
    rows+=['','全部数值均为同一3834条Validation、27标签、seed42、阈值.30；没有把历史Test与本轮Validation混表。A0/A1读取封存数组重算，A1-G-C完整四轮后按Validation Macro-F1选轮并exact reload。','',
        '## A1-G-C相对A1的变化（pp）','',
        '|指标|变化|','|---|---:|']
    for k in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']:
        rows.append(f'|{k}|{100*(aggregate["A1-G-C"][k]-aggregate["A1"][k]):+.4f}|')
    delta=aggregate['A1-G-C']['macro_f1']-aggregate['A1']['macro_f1'];direction='提高' if delta>0 else '降低' if delta<0 else '持平'
    rows+=['',f'预定主指标Macro-F1相对原A1{direction}，变化{100*delta:+.4f}pp。全部27类P/R/F1/AP/TP/FP/FN及逐类贡献见CSV，包含负向结果。','',
        '## 逐标签变化','', '|情绪|support|F1变化pp|AP变化pp|','|---|---:|---:|---:|']
    for r in deltas:rows.append(f'|{r["label"]}|{r["support"]}|{r["f1_delta_pp"]:+.4f}|{r["AP_delta_pp"]:+.4f}|')
    rows+=['','## Loss尺度与数值稳定性','',
        'training_log.csv记录4轮BCE/CE/旧坐标项诊断/新距离项/aux及clip频率；batch_log.csv保存全部7648步。probes各目录包含同forward下旧/新aux与BCE的四模块梯度、cosine、边界截断及RNG/state不变证明。',
        f'gamma_G={GAMMA!r}来自fresh train-only预定log-median规则，训练前冻结、全程不调。它匹配的是初始共享aux梯度典型尺度，未匹配方向，也不保证未来训练尺度匹配。若有改善，只能解释为测地目标与校准权重的联合设置效果。',
        '## FYP Methodology建议','',
        '把原方法准确写为HNN/Poincare分支 + tangent-space decoder soft-target CE + coordinate inner-product alignment；原A1对齐不是geodesic distance。',
        f'A1-G-C单独列为探索性的测地目标与整个aux梯度尺度匹配实验，给出c=1、正标签均值/完整batch均值、gamma_G={GAMMA!r}与FP64局部距离计算。辅助CE系数也变，主分类仍是RoBERTa欧氏分类头。',
        '27类neutral-only过滤实验属于preliminary条件任务；计划中的28类任务及neutral mask要另述，不能当作已完成。',
        '说明固定发布标签先验及完整预训练日志不足的来源限制，披露近边界与clip；单seed、反复探索Validation、无新Test，不声称显著或最终泛化。',
        '测地距离式对齐优于坐标内积（若观察到）也不等于双曲几何优于matched Euclidean方法。四项审计与明确否定结果均应保留。',
        '本轮到此停止，没有新seed、权重搜索、欧氏训练、28类重跑或正式Test。']
    rows+=['','## 固定Rare7（全部Validation）','',
        '|标签|support|A0 F1 %|A1 F1 %|A1-G-C F1 %|A1-G-C AP %|','|---|---:|---:|---:|---:|---:|']
    for label in CONFIG['rare_labels']:
        a0,a1,c=by['A0',label],by['A1',label],by['A1-G-C',label]
        rows.append(f'|{label}|{c["support"]}|{100*a0["f1"]:.4f}|{100*a1["f1"]:.4f}|{100*c["f1"]:.4f}|{100*c["average_precision"]:.4f}|')
    rows+=['','## 真实训练投影截断与全局梯度裁剪','',
        '|epoch|首次text clip|后续text任一clip|最终半径>.99|平均半径|global clip1|最大未裁剪norm|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in csvrows(root/'training_log.csv'):
        rows.append(f'|{r["epoch"]}|{100*float(r["training_first_text_clip_fraction"]):.2f}%|{100*float(r["training_later_text_clip_fraction"]):.2f}%|{100*float(r["training_final_fraction_above099"]):.2f}%|{float(r["training_final_radius_mean"]):.6f}|{100*float(r["global_gradient_clip_fraction"]):.2f}%|{float(r["gradient_norm_max"]):.6f}|')
    rows+=['','上表来自各轮全部30587个真实训练positions，不是eval小探针估计。ProjectionMonitor只观察原proj输入/输出，不改Tensor/RNG；其train/eval forward逐元素一致性在初始化parity中核验。',
        's=1的首次截断问题保持并如实报告，不能因为系数降低就声称几何饱和已修复。固定train-only probe给出额外局部梯度/方向诊断，不用这些probe更改gamma。','',
        '## 固定train-only探针的共享梯度诊断','',
        '|时点|weighted aux/BCE中位数|BCE与aux cos中位数|weighted CE norm中位数|weighted distance norm中位数|CE-distance cos中位数|',
        '|---|---:|---:|---:|---:|---:|']
    for tag in ['fresh_initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        gg=[r for r in csvrows(root/'probes'/tag/'MODULE_GRADIENTS.csv') if r['module']=='shared_encoder']
        values=[np.median([float(r[k]) for r in gg]) for k in ['geodesic_aux_to_bce','bce_geodesic_aux_cosine','weighted_ce_norm','weighted_distance_norm','ce_distance_cosine']]
        rows.append('|'+tag+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    rows+=['','每时点使用同10cohorts/160位置/145独立train IDs、eval模式；方向匹配与范数匹配分别解释。CE/distance分量梯度本轮实际重新测量，不能从此前校准摘要推断这些分量各自匹配。']
    md(root/'GEODESIC_ALIGNMENT_SUMMARY_CN.md',rows)
