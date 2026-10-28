"""Run-bound comparisons. No formula/threshold/seed selection from Test."""
import numpy as np
from ncommon import *

METRICS=['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']

def write_reports(root):
    hist={tag:read(root/'historical_validation'/(tag+'_metrics.json')) for tag in ['A0','A1','A1S']}
    hist['N1']=read(root/'validation_metrics.json')
    comparison=[dict(metric=k,**{tag:m[k] for tag,m in hist.items()},delta_vs_A1S_pp=100*(hist['N1'][k]-hist['A1S'][k]),
        delta_vs_A0_pp=100*(hist['N1'][k]-hist['A0'][k])) for k in METRICS]
    table(root/'EXP6D1_VALIDATION_COMPARISON.csv',comparison)
    per={tag:csvrows(root/'historical_validation'/(tag+'_per_label.csv')) for tag in ['A0','A1','A1S']};per['N1']=csvrows(root/'validation_per_label.csv')
    keys=['tp','fp','fn','precision','recall','f1','average_precision','predicted_positive_count','mean_positive_probability']
    labels=[]
    for i,new in enumerate(per['N1']):
        aligned={tag:rows[i] for tag,rows in per.items()}
        need(all(r['label']==new['label'] and r['support']==new['support'] for r in aligned.values()),'Per-label comparison not paired')
        row=dict(label=new['label'],support=int(new['support']),rare=new['label'] in a0.RARE)
        for tag,r in aligned.items():
            for k in keys:row[tag+'_'+k]=float(r[k]) if r[k]!='' else None
        row.update(delta_F1_vs_A1S_pp=100*(row['N1_f1']-row['A1S_f1']),delta_AP_vs_A1S_pp=100*(row['N1_average_precision']-row['A1S_average_precision']),
            macro_F1_contribution_vs_A1S_pp=100*(row['N1_f1']-row['A1S_f1'])/27,
            rare_F1_contribution_vs_A1S_pp=100*(row['N1_f1']-row['A1S_f1'])/7 if row['rare'] else 0.)
        labels.append(row)
    table(root/'EXP6D1_PER_LABEL_COMPARISON.csv',labels)
    rare=[next(r for r in labels if r['label']==n) for n in a0.RARE]
    table(root/'EXP6D1_RARE_LABEL_COMPARISON.csv',rare)
    with np.load(root/'validation_predictions.npz',allow_pickle=False) as saved:
        _,numeric_rows,_,_=extended_score(saved['targets'],probabilities(saved['logits']),saved['label_names'].tolist())
    write(root/'rare_label_metrics.json',dict(definition='mean F1 over fixed seven labels; none excluded',labels=a0.RARE,
        rare_f1=hist['N1']['rare_f1'],per_label=[next(r for r in numeric_rows if r['label']==n) for n in a0.RARE],matched_comparison=rare))
    geometry=[];newgeo=[]
    for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        for model,path in [('A1S',CONTROL),('N1',root)]:
            p=read(path/'geometry'/tag/'pooled.json')
            geometry.append(dict(model=model,tag=tag,**{k:p[k] for k in ['unique_ids','cohort_positions','first_clip_fraction','later_clip_fraction',
                'final_fraction_above_099','final_mean_norm','final_min_norm','final_max_norm','final_norm_std','final_norm_IQR']},
                median_shared_aux_to_class_gradient_norm=p['median_shared_aux_to_N1_gradient_norm'] if model=='N1' else p['median_shared_aux_to_BCE_gradient_norm'],
                median_shared_class_aux_cosine=p['median_shared_N1_aux_cosine'] if model=='N1' else p['median_shared_gradient_cosine'],
                median_shared_N1_to_BCE_gradient_norm=p.get('median_shared_N1_to_BCE_gradient_norm')))
            if model=='N1':newgeo.append(p)
    table(root/'EXP6D1_GEOMETRY_REPORT.csv',geometry)
    write(root/'geometry_diagnostics.json',dict(pooled=newgeo,unique_train_ids=145,cohort_positions=160,
        mode='eval/FP32 forward, exact frozen scaling, autograd.grad; restore all RNG/modes; no optimizer',
        duplicate_policy='first occurrence in recorded cohort order for unique pooled geometry; retain cohort batches for gradients'))
    weights=csvrows(root/'EXP6D1_DISTANCE_WEIGHT_DIAGNOSTICS.csv');epochs=[r for r in weights if r['group']=='all']
    grads=csvrows(root/'geometry/initial/gradients.csv')
    ratios=[float(r['N1_to_BCE_norm']) for r in grads if r['module'] in ['shared_encoder','classification_head']]
    all_ratios=[]
    for tag in ['initial','epoch1','epoch2','epoch3','epoch4','best_reload']:
        all_ratios += [float(r['N1_to_BCE_norm']) for r in csvrows(root/'geometry'/tag/'gradients.csv') if r['module'] in ['shared_encoder','classification_head']]
    replays=read(root/'smoke/geometry/A1S_best_replay/EXP6D_REPLAY_PASS.json')
    md(root/'EXP6D1_N1_LOSS_VERIFICATION.md',['# Exp6D-1 N1 Loss验收','',
        '实现d_i为所有真阳性Poincaré geodesic均值，w_i=stopgrad(d_i/mean_current_batch d)，ell_i=mean_27 BCE，L=mean(w*ell)+.1aux。无alpha、eps、distance clip、temperature、rare weighting。',
        '距离以FP64计算后将归一化权重cast FP32，与已完成Exp6D工具audit_math.candidates逐例/逐模块核对。辅助项直接保留冻结A1-S forward返回的Tensor，保持归约与梯度图；先27维mean再sample weighted mean。',
        'smoke的single/multi-positive、B1和short11样本核对独立参考与sigmoid BCE解析导数。B1权重退化为1；最后训练batch11按自身均距归一化，在每epoch保存数组中复算。',
        f'Exp6D已训练A1-S checkpoint只用于只读replay：{replays}。正式训练重新从raw snapshot seed42初始化，没有从此checkpoint续训。',
        f'fresh initial probe分类N1/BCE参数梯度比范围：{min(ratios):.6f}–{max(ratios):.6f}。四个module分别计算ordinary BCE、N1 BCE、.1aux及combined梯度并检验相加关系。距离权重不反传至HNN/decoder；aux不连接分类头；固定向量无grad。',
        '原BCE(flat mean)与逐例mean再batch mean允许预声明FP32求和舍入差异atol1e-7/rtol1e-6；不把这类差异当新方法。权重和距离参考值、真实模块梯度replay及初始化证明详见smoke/、INITIALIZATION_PARITY.json和N1_INITIALIZATION_PARITY.json。'])
    selection=read(root/'checkpoint_selection.json');reload=read(root/'checkpoint_reload_verification.json');post=read(root/'FROZEN_POSTCHECK.json')
    md(root/'EXP6D1_VERIFICATION_REPORT.md',['# Exp6D-1 结果验收','',
        '最终SUCCESS仅在nrun.verify通过后生成；不能凭本Markdown存在认定成功。校验器逐项检查：源manifest、config、4轮7648步/765warmup、最后LR0、批日志与每轮样本权重、完整训练顺序、checkpoint SHA、Validation数组与所有27类metrics/AP、重载、无更新几何探针以及冻结保护证明。',
        '初始化来自相同raw snapshot；shared/head/HNN/decoder初值哈希与A1-S一致，安装N1前后train/eval logits、z、decoder输出及RNG消耗一致。',
        f'选择证据：{selection}',f'完整3834条Validation重载：{reload}',f'冻结文件复核：{post}',
        '新Test推理=0；无Test预测文件；没有rawF训练、额外seed或参数搜索。训练与smoke退出码由shell记录，完整SUCCESS封存后还会由CPU只读校验器复查。'])
    n=hist['N1'];c=hist['A1S'];base=hist['A0']
    if n['macro_f1']>c['macro_f1']:conclusion='归一化双曲距离加权在当前固定seed42设置下改善了A1-S的Validation Macro-F1。'
    elif n['macro_f1']<c['macro_f1']:conclusion='本次N1没有改善A1-S的Validation Macro-F1，需保留这一负结果，不自动尝试其他公式或seed。'
    else:conclusion='本次N1与A1-S的Validation Macro-F1相同，未观察到该主要指标收益。'
    over0='超过' if n['macro_f1']>base['macro_f1'] else '未超过'
    apdelta=100*(n['macro_average_precision']-c['macro_average_precision'])
    rows=['# Exp6D-1：N1 seed42 Validation-only pilot','',conclusion,'',
        '本轮唯一机制变化是A1-S的ordinary BCE改为N1样本加权BCE；s=.03、gamma=.1、辅助目标、固定27向量、head与全部训练协议保持。',
        f'从相同预训练权重独立训练4轮7648step；best epoch={selection["epoch"]}，按Validation Macro-F1 strict greater-than选轮，平局取早。完整3834条重载logits/IDs/gold/阈值预测/aggregate及per-label metrics核对。新Test=0。','',
        '## Validation匹配比较','',
        '|指标|A0|A1|A1-S|N1|对A1-S变化pp|对A0变化pp|','|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison:rows.append('|'+r['metric']+'|'+ '|'.join(f'{100*r[k]:.4f}%' for k in ['A0','A1','A1S','N1'])+f'|{r["delta_vs_A1S_pp"]:+.4f}|{r["delta_vs_A0_pp"]:+.4f}|')
    rows+=['','历史值从封存Validation数组重算，核对原metrics；没有按prompt四舍五入值填表，也没有使用Test选择方法。A1-S为主要对照，A0/A1为历史参考。','',
        '## 七个固定Rare标签','',
        '|标签|Val support|A1-S F1|N1 F1|变化pp|A1-S AP|N1 AP|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rare:rows.append(f'|{r["label"]}|{r["support"]}|{100*r["A1S_f1"]:.4f}%|{100*r["N1_f1"]:.4f}%|{r["delta_F1_vs_A1S_pp"]:+.4f}|{100*r["A1S_average_precision"]:.4f}%|{100*r["N1_average_precision"]:.4f}%|')
    gains=sorted(labels,key=lambda r:r['macro_F1_contribution_vs_A1S_pp'],reverse=True)
    rows+=['','grief、relief、pride分别保留，不因结果选择阈值或给其特殊权重。27类完整P/R/F1/AP/TP/FP/FN与Macro/Rare贡献在PER_LABEL_COMPARISON.csv；稀有支持量小，F1的阈值变化与AP的排序变化需分开解释。',
        'Macro改善贡献最大的三个标签（并不自动等价为总体改善）：'+', '.join(f'{r["label"]} {r["macro_F1_contribution_vs_A1S_pp"]:+.4f}pp' for r in gains[:3])+'。','',
        '## 实际训练距离/权重分布','',
        '|epoch|distance mean/std|weight mean/std|min/p95/max weight|weighted BCE/BCE|corr(weight,BCE)|',
        '|---|---|---|---|---|---|']
    for r in epochs:rows.append(f'|{r["epoch"]}|{float(r["distance_mean"]):.4f}/{float(r["distance_std"]):.4f}|{float(r["weight_mean"]):.6f}/{float(r["weight_std"]):.6f}|{float(r["weight_min"]):.4f}/{float(r["weight_p95"]):.4f}/{float(r["weight_max"]):.4f}|{float(r["weighted_bce_to_bce"]):.6f}|{r["weight_vs_bce_correlation"]}|')
    rows+=['','以上按每轮30587个真实train positions合并计算，single/multi-positive分别统计于DISTANCE_WEIGHT_DIAGNOSTICS.csv；逐样本距离、权重、BCE和正标签数保存于train_n1_epoch*.npz。权重非恒定只说明存在重加权，不能据此断言差异“足够有效”。','',
        '## 固定训练探针几何轨迹','',
        '|时点|首次clip|后续任一text clip|最终>0.99|半径均值|aux/N1共享梯度比中位数|N1/BCE共享梯度比中位数|N1-aux cos中位数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in newgeo:rows.append(f'|{r["tag"]}|{r["first_clip_fraction"]:.2%}|{r["later_clip_fraction"]:.2%}|{r["final_fraction_above_099"]:.2%}|{r["final_mean_norm"]:.6f}|{r["median_shared_aux_to_N1_gradient_norm"]:.4f}|{r["median_shared_N1_to_BCE_gradient_norm"]:.4f}|{r["median_shared_N1_aux_cosine"]:+.4f}|')
    final=newgeo[4];selected=newgeo[selection['epoch']]
    saturation=('仍有后续HNN投影截断' if any(r['later_clip_fraction']>0 for r in newgeo) else '固定探针中未观察到后续HNN投影截断')
    rows+=['','每时点相同10cohorts、160位置/145独立train IDs。探针eval状态且不更新权重，恢复所有RNG/mode；不能当整套train分布或真实训练clip频率。训练batch global gradient clipping另见training_log.csv。','',
        '## 中文十问','',
        '1. **N1是否正确实现？** 通过独立Exp6D公式、真实旧checkpoint数值/梯度replay、全部正标签、B1/B11与本轮保存权重重算验收后，才能生成SUCCESS。',
        f'2. **是否仍有十倍分类梯度放大？** fresh probe N1/BCE比范围{min(ratios):.4f}–{max(ratios):.4f}；initial/四轮/best reload全部probe的比值范围{min(all_ratios):.4f}–{max(all_ratios):.4f}。这些是固定train-only局部梯度，不等于已逐训练步分离测量所有梯度。异常>=10倍分类梯度的probe会停止本次运行。',
        f'3. **Macro/Micro/Rare变化？** 相对A1-S分别{100*(n["macro_f1"]-c["macro_f1"]):+.4f}/{100*(n["micro_f1"]-c["micro_f1"]):+.4f}/{100*(n["rare_f1"]-c["rare_f1"]):+.4f}pp。',
        f'4. **超过控制或A0？** {conclusion} 按Macro-F1，N1{over0}A0；其他指标分别见表，不用Micro替代预定主指标。',
        '5. **Rare恢复？** 上表完整报告七类F1/AP与support；不删除grief。单标签的小样本跳变不能直接写成普遍恢复。',
        f'6. **Macro AP改善？** 对A1-S变化{apdelta:+.4f}pp；AP与固定阈值F1分别解释。',
        '7. **权重差异是否足够？** 全train分布及single/multi差异已量化；是否有用只能结合识别结果，非零std本身不证明价值。',
        f'8. **边界饱和？** {saturation}；epoch4 later clip={final["later_clip_fraction"]:.2%}、最终>0.99={final["final_fraction_above_099"]:.2%}；选中轮later clip={selected["later_clip_fraction"]:.2%}。靠边界与实际截断分别记录，未自动改s。',
        '9. **能否说distance weighting有帮助？** '+conclusion+' 仅限当前seed42探索性Validation协议；尚无独立Test或多seed稳定性证据。',
        '10. **几何独立价值还缺什么？** 需要容量/辅助目标/标签先验/尺度/训练预算匹配的Euclidean control，再做经批准的重复实验；不能把额外监督或样本加权收益直接归因于Hyperbolic geometry。','',
        '本轮完成后停止。没有自动训练Raw F、更多seed、其他s/gamma/normalization，也没有新Test推理。即便结果改善，也不能宣称显著、泛化或双曲优越性。']
    md(root/'EXP6D1_TRAINING_SUMMARY.md',rows)
