"""Four paired seed42 Validation models; no test-driven tuning or hidden negatives."""
from mcommon import *


def report(root,manifest):
    models=[('A0',A0),('A1',A1),('A1-C-Gamma',root),('A1-G-C',G2)]
    metrics={};per={};allrows=[]
    for tag,path in models:
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
            need(z['ids'].tolist()==manifest['splits']['validation']['retained_ids'] and z['label_names'].tolist()==manifest['names27'],'Unpaired comparison data')
            m,rows,_,_=extended_score(z['targets'],probabilities(z['logits']),manifest['names27'])
            need(all(m[k]==v for k,v in read(path/'validation_metrics.json').items()),'Historical Validation metrics changed')
        metrics[tag]=m;per[tag]={r['label']:r for r in rows}
        allrows += [dict(model=tag,split='validation',seed=42,**r) for r in rows]
    fields=['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']
    table(root/'EXP6E3_A1_A1CG_A1GC_COMPARISON.csv',[dict(model=tag,split='validation',seed=42,labels=27,threshold=.3,
        alignment='none' if tag=='A0' else 'geodesic distance' if tag=='A1-G-C' else 'negative coordinate dot',
        gamma=0. if tag=='A0' else .1 if tag=='A1' else GAMMA,**{k:metrics[tag][k] for k in fields}) for tag,_ in models])
    table(root/'ALL_MODELS_PER_LABEL.csv',allrows)
    differences=[]
    questions=[('A_lower_gamma','A1-C-Gamma','A1'),('B_geodesic_vs_coordinate_same_gamma','A1-G-C','A1-C-Gamma'),
               ('C_coordinate_vs_A0','A1-C-Gamma','A0'),('C_geodesic_vs_A0','A1-G-C','A0')]
    for question,left,right in questions:
        for label in manifest['names27']:
            l,r=per[left][label],per[right][label]
            differences.append(dict(question=question,left=left,right=right,label=label,support=l['support'],
                f1_delta_pp=100*(l['f1']-r['f1']),AP_delta_pp=100*(l['average_precision']-r['average_precision']),
                precision_delta_pp=100*(l['precision']-r['precision']),recall_delta_pp=100*(l['recall']-r['recall']),
                macro_contribution_pp=100*(l['f1']-r['f1'])/27))
    table(root/'ALL_PAIRED_LABEL_DELTAS.csv',differences)
    table(root/'EXP6E3_RARE_LABEL_ANALYSIS.csv',[r for r in allrows if r['label'] in CONFIG['rare_labels']])
    history=csvrows(root/'training_log.csv');geometry=[]
    for r in history:
        geometry.append({k:r[k] for k in ['epoch','training_first_text_clip_fraction','training_later_text_clip_fraction',
            'training_final_fraction_above099','training_final_radius_mean','training_final_radius_min','training_final_radius_max',
            'global_gradient_clip_fraction','gradient_norm_mean','gradient_norm_max']})
    table(root/'EXP6E3_GEOMETRY_REPORT.csv',geometry)
    lines=['# Exp6E-3 A1-C-Gamma：匹配系数坐标对齐对照','',
        f'唯一训练改动相对原A1为整个aux系数 .1 → {GAMMA!r}。原BCE、soft-target CE、q与正标签坐标内积、HNN/decoder、s=1、固定27向量及训练预算保持。',
        '四轮7648步完成；seed42、同3834条Validation、27类、阈值.30；按Macro-F1选轮并完整重载。新Test=0。','',
        '|模型|Gamma|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|','|---|---:|---:|---:|---:|---:|']
    for tag,_ in models:
        gamma=0 if tag=='A0' else .1 if tag=='A1' else GAMMA
        lines.append('|'+tag+'|'+f'{gamma:.10g}'+'|'+'|'.join(f'{100*metrics[tag][k]:.4f}' for k in ['micro_f1','macro_f1','rare_f1','macro_average_precision'])+'|')
    for q,left,right in questions:
        rows=[r for r in differences if r['question']==q]
        delta=100*(metrics[left]['macro_f1']-metrics[right]['macro_f1'])
        lines+=['',f'## {q}: {left} vs {right}','',
            f'Macro-F1 {delta:+.4f}pp；Rare-F1 {100*(metrics[left]["rare_f1"]-metrics[right]["rare_f1"]):+.4f}pp；Micro-F1 {100*(metrics[left]["micro_f1"]-metrics[right]["micro_f1"]):+.4f}pp；Macro AP {100*(metrics[left]["macro_average_precision"]-metrics[right]["macro_average_precision"]):+.4f}pp。',
            f'27标签中{sum(r["f1_delta_pp"]>0 for r in rows)}类F1改善、{sum(r["f1_delta_pp"]<0 for r in rows)}类下降、{sum(r["f1_delta_pp"]==0 for r in rows)}类持平。',
            '这是完整逐类分解，不因结果删标签或改阈值。grief+relief对Macro贡献：'+f'{sum(r["macro_contribution_pp"] for r in rows if r["label"] in ["grief","relief"]):+.4f}pp；其他25类合计'+f'{sum(r["macro_contribution_pp"] for r in rows if r["label"] not in ["grief","relief"]):+.4f}pp。']
    lines+=['','## Rare7 F1/AP（同Validation）','',
        '|标签|support|A1 F1/AP %|A1-C-Gamma F1/AP %|A1-G-C F1/AP %|','|---|---:|---:|---:|---:|']
    for label in CONFIG['rare_labels']:
        vals=[per[tag][label] for tag in ['A1','A1-C-Gamma','A1-G-C']]
        lines.append('|'+label+'|'+str(vals[0]['support'])+'|'+'|'.join(f'{100*r["f1"]:.4f}/{100*r["average_precision"]:.4f}' for r in vals)+'|')
    lines+=['','## 27类完整方向','', '|标签|C-Gamma−A1 F1/AP pp|Geodesic−C-Gamma F1/AP pp|','|---|---:|---:|']
    for label in manifest['names27']:
        a=next(r for r in differences if r['question']=='A_lower_gamma' and r['label']==label)
        b=next(r for r in differences if r['question']=='B_geodesic_vs_coordinate_same_gamma' and r['label']==label)
        lines.append(f'|{label}|{a["f1_delta_pp"]:+.4f}/{a["AP_delta_pp"]:+.4f}|{b["f1_delta_pp"]:+.4f}/{b["AP_delta_pp"]:+.4f}|')
    lines+=['','## 真实训练几何/裁剪','', '|epoch|首投影clip|后续任一clip|最终>.99|平均半径|global clip1|','|---|---:|---:|---:|---:|---:|']
    for r in geometry:
        lines.append('|'+r['epoch']+'|'+'|'.join(f'{100*float(r[k]):.2f}%' for k in ['training_first_text_clip_fraction','training_later_text_clip_fraction','training_final_fraction_above099'])+'|'+f'{float(r["training_final_radius_mean"]):.6f}|{100*float(r["global_gradient_clip_fraction"]):.2f}%|')
    lines+=['','## 固定train-only共享梯度','', '|时点|weighted aux/BCE中位数|BCE–aux cos中位数|CE–negative dot cos中位数|','|---|---:|---:|---:|']
    for tag in ['fresh_initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        rows=[r for r in csvrows(root/'probes'/tag/'MODULE_GRADIENTS.csv') if r['module']=='shared_encoder']
        values=[np.median([float(r[k]) for r in rows]) for k in ['matched_aux_to_bce','bce_matched_aux_cosine','ce_negative_dot_cosine']]
        lines.append('|'+tag+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    b=100*(metrics['A1-G-C']['macro_f1']-metrics['A1-C-Gamma']['macro_f1'])
    judgement='测地对齐在本次同gamma的seed42 Validation中Macro-F1更高' if b>0 else '测地对齐在本次同gamma的seed42 Validation中Macro-F1更低' if b<0 else '两目标本次同gamma的Macro-F1持平'
    lines+=['','## 中文结论与限制','',
        '1. 本轮相对A1严格只修改整个aux系数；forward/未加权loss及梯度线性缩放在训练前验收。',
        '2. 系数降低是否改善Macro与Rare，按上方Question A实际正负值回答，不把预期当结果。',
        f'3. {judgement}，差{b:+.4f}pp；方向/范数不同仍是目标效应的一部分，同gamma并非同梯度强度。',
        '4. Rare是否普遍改善取决于完整Rare7及AP表，不能由平均Rare-F1或grief/relief两项自动得出。',
        '5. s=1保持；首/后续clip与最终半径分阶段记录。真实训练clip与小样本eval探针分别报告，不自动修几何或重估gamma。',
        '6. 单seed、探索过的Validation不构成统计显著/泛化或负曲率优于欧氏的证据；本轮同gamma对照可以缩小系数混杂，但不能证明几何独立价值。',
        '7. 是否做三seed须结合Question B效应大小、负向标签和稀有支持量另行决定。本轮没有自动多seed/新Test/搜索/Euclidean control。',
        'best_checkpoint/training_state.pt保存选中轮的optimizer/scheduler/CPU+CUDA/Python/NumPy RNG及DataLoader generator；未另存最终轮权重/状态，不提供自动续训。',
        '实验和备份分别验收：外接盘与Mac状态由独立migration记录及目标文件校验给出，科学完成标志不代表备份已完成。']
    md(root/'EXP6E3_TRAINING_SUMMARY_CN.md',lines)
