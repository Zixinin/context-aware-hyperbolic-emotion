"""Validation-only comparison with frozen original A0/A1, no test-driven tuning."""
from gcommon import *

def report(root,manifest):
    allrows=[];aggregate={};historical=root/'historical_validation';historical.mkdir()
    for tag,path in [('A0',A0),('A1',A1),('A1-G',root)]:
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['label_names'].tolist()==manifest['names27'] and z['ids'].tolist()==manifest['splits']['validation']['retained_ids'],'Unpaired comparison arrays')
            m,rows,_,pred=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
            need(all(m[k]==v for k,v in read(path/'validation_metrics.json').items()),'Historical Validation metric mismatch')
        aggregate[tag]=m
        if tag!='A1-G':write(historical/(tag+'_metrics.json'),m);table(historical/(tag+'_per_label.csv'),rows)
        for row in rows:allrows.append(dict(model=tag,split='validation',seed=42,**row))
    table(root/'PER_LABEL_COMPARISON.csv',allrows)
    compact=[dict(model=tag,alignment={'A0':'none','A1':'negative coordinate inner product','A1-G':'positive unsquared geodesic distance'}[tag],
        split='validation',seed=42,**{k:m[k] for k in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']}) for tag,m in aggregate.items()]
    table(root/'VALIDATION_COMPARISON.csv',compact)
    by={(r['model'],r['label']):r for r in allrows};deltas=[]
    for label in manifest['names27']:
        a,g=by['A1',label],by['A1-G',label]
        deltas.append(dict(label=label,support=g['support'],precision_delta_pp=100*(g['precision']-a['precision']),recall_delta_pp=100*(g['recall']-a['recall']),
            f1_delta_pp=100*(g['f1']-a['f1']),AP_delta_pp=100*(g['average_precision']-a['average_precision']),macro_contribution_pp=100*(g['f1']-a['f1'])/27))
    table(root/'PER_LABEL_GEODESIC_MINUS_COORDINATE.csv',deltas)
    rows=['# Geodesic Alignment Ablation：seed42 Validation结果','',
        '本轮只将原A1的负坐标内积对齐替换为正测地距离对齐；原scale1、普通BCE、相同soft-target CE、固定27向量、gamma.1和完整训练协议保持。新Test=0。','',
        '|模型|Alignment|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|','|---|---|---:|---:|---:|---:|']
    for r in compact:rows.append(f'|{r["model"]}|{r["alignment"]}|'+ '|'.join(f'{100*r[k]:.4f}' for k in ['micro_f1','macro_f1','rare_f1','macro_average_precision'])+'|')
    rows+=['','全部数值均为同一3834条Validation、27标签、seed42、阈值.30；没有把历史Test与本轮Validation混表。A0/A1读取封存数组重算，A1-G完整四轮后按Validation Macro-F1选轮并exact reload。','',
        '## A1-G相对A1的变化（pp）','',
        '|指标|变化|','|---|---:|']
    for k in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']:
        rows.append(f'|{k}|{100*(aggregate["A1-G"][k]-aggregate["A1"][k]):+.4f}|')
    delta=aggregate['A1-G']['macro_f1']-aggregate['A1']['macro_f1'];direction='提高' if delta>0 else '降低' if delta<0 else '持平'
    rows+=['',f'预定主指标Macro-F1相对原A1{direction}，变化{100*delta:+.4f}pp。全部27类P/R/F1/AP/TP/FP/FN及逐类贡献见CSV，包含负向结果。','',
        '## 逐标签变化','', '|情绪|support|F1变化pp|AP变化pp|','|---|---:|---:|---:|']
    for r in deltas:rows.append(f'|{r["label"]}|{r["support"]}|{r["f1_delta_pp"]:+.4f}|{r["AP_delta_pp"]:+.4f}|')
    rows+=['','## Loss尺度与数值稳定性','',
        'training_log.csv记录4轮BCE/CE/旧坐标项诊断/新距离项/aux及clip频率；batch_log.csv保存全部7648步。probes各目录包含同forward下旧/新aux与BCE的四模块梯度、cosine、边界截断及RNG/state不变证明。',
        '原权重不变不意味着监督尺度相同。若有改善，也不能排除距离梯度尺度变化的贡献；本轮没有调gamma。任何权重敏感性须另行定义，不与本轮混称严格单变量几何证明。',
        '## FYP Methodology建议','',
        '把原方法准确写为HNN/Poincare分支 + tangent-space decoder soft-target CE + coordinate inner-product alignment；原A1对齐不是geodesic distance。',
        'A1-G单独列为探索性对齐目标消融，给出正距离符号、c=1、正标签均值/完整batch均值、.1系数与FP64局部距离计算。主分类仍是RoBERTa欧氏分类头。',
        '27类neutral-only过滤实验属于preliminary条件任务；计划中的28类任务及neutral mask要另述，不能当作已完成。',
        '说明固定发布标签先验及完整预训练日志不足的来源限制，披露近边界与clip；单seed、反复探索Validation、无新Test，不声称显著或最终泛化。',
        '测地距离式对齐优于坐标内积（若观察到）也不等于双曲几何优于matched Euclidean方法。四项审计与明确否定结果均应保留。',
        '本轮到此停止，没有新seed、权重搜索、欧氏训练、28类重跑或正式Test。']
    md(root/'GEODESIC_ALIGNMENT_SUMMARY_CN.md',rows)
