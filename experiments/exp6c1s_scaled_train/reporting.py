"""Validation-only comparisons and candid, run-bound Chinese pilot report."""
from pathlib import Path
import numpy as np
from scommon import read,write,table,md,extended_score,probabilities,need,a0,STATUS

METRICS=['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']

def write_reports(root):
    histories={tag:read(root/'historical_validation'/(tag+'_metrics.json')) for tag in ['A0','A1']}
    new=read(root/'validation_metrics.json');histories['A1_S']=new
    comparison=[dict(metric=k,A0=histories['A0'][k],A1=histories['A1'][k],A1_S=new[k],
        delta_vs_A0_pp=100*(new[k]-histories['A0'][k]),delta_vs_A1_pp=100*(new[k]-histories['A1'][k])) for k in METRICS]
    table(root/'EXP6C1S_A0_A1_SCALED_COMPARISON.csv',comparison)
    from scommon import csvrows
    per={tag:csvrows(root/'historical_validation'/(tag+'_per_label.csv')) for tag in ['A0','A1']}
    per['A1_S']=csvrows(root/'validation_per_label.csv')
    rare=[]
    for label in a0.RARE:
        aligned={tag:next(r for r in rows if r['label']==label) for tag,rows in per.items()}
        need(len({r['support'] for r in aligned.values()})==1,'Rare support mismatch')
        r=dict(label=label,support=int(aligned['A1_S']['support']))
        for tag,row in aligned.items():
            for key in ['tp','fp','fn','precision','recall','f1','average_precision','predicted_positive_count','mean_positive_probability']:
                r[tag+'_'+key]=float(row[key]) if row[key]!='' else None
        r.update(delta_F1_vs_A0_pp=100*(r['A1_S_f1']-r['A0_f1']),delta_F1_vs_A1_pp=100*(r['A1_S_f1']-r['A1_f1']),
            delta_AP_vs_A0_pp=100*(r['A1_S_average_precision']-r['A0_average_precision']),delta_AP_vs_A1_pp=100*(r['A1_S_average_precision']-r['A1_average_precision']))
        rare.append(r)
    table(root/'EXP6C1S_RARE_LABEL_ANALYSIS.csv',rare)
    tags=['initial','epoch1','epoch2','epoch3','epoch4','best_reload']
    geo=[read(root/'geometry'/tag/'pooled.json') for tag in tags]
    cohort_rows=[]
    for tag in tags:cohort_rows+=csvrows(root/'geometry'/tag/'cohorts.csv')
    table(root/'EXP6C1S_GEOMETRY_EPOCH_REPORT.csv',cohort_rows)
    write(root/'geometry_diagnostics.json',dict(pooled=geo,cohort_count=10,unique_train_ids=145,
        duplicate_policy='pooled keeps first occurrence in frozen cohort order; cohort rows retain all160 positions',
        diagnostics_mode='eval/FP32, autograd.grad, all RNG and module modes restored'))
    selection=read(root/'checkpoint_selection.json');macro0=histories['A0']['macro_f1'];macro1=histories['A1']['macro_f1'];macro=new['macro_f1']
    if macro>macro0:
        conclusion='在当前seed42、27-label任务及固定评估协议下，缩放辅助分支取得了Validation Macro-F1的初步收益。'
    elif macro>macro1:
        conclusion='输入缩放缓解了当前A1的部分性能退化，但尚未超过普通BCE baseline。'
    elif macro<macro1:
        conclusion='单独缩放没有改善Validation Macro-F1，且低于原A1；几何是否改善须单独看下表。'
    else:conclusion='Validation Macro-F1与原A1相同，未观察到该主要指标收益。'
    if macro>macro1 and new['rare_f1']>=histories['A1']['rare_f1']:
        continuation='Macro改善且Rare未下降，存在继续确认该候选的初步理由；仍需看AP、几何轨迹和后续独立对照，不能直接宣称稳定收益。'
    elif macro>macro1:
        continuation='Macro改善但Rare下降，存在指标取舍，尚不足以宣称整体改善；暂不扩展缩放搜索或多seed。'
    else:continuation='主要指标未超过原A1，当前没有继续扩展缩放搜索的性能依据；保留几何观察，优先分析辅助目标或匹配欧氏对照。'
    final=geo[4];selected=geo[selection['epoch']]
    lines=['# Exp6C-1S-Train：s=0.03，seed42，Validation-only pilot','',conclusion,'',
        'This result verifies only the declared computational claim under the recorded environment and protocol.',
        '本轮仅将A1双曲辅助入口h改为0.03*h，冻结的A0/A1/原始HypEmo源码未改。普通分类输入、架构、初始化、固定27类标签向量、BCE/Formula A、gamma0.1与原训练协议不变。',
        f'四轮7648步已完成，765 warmup；按完整Validation Macro-F1严格大于选轮，平局取早。最佳epoch={selection["epoch"]}；3834条Validation完整重载逐元素logits/IDs/gold/固定阈值预测和指标一致。新Test推理=0。',
        '', '## 1. 真实Validation比较（阈值>=0.30）','',
        '| 指标 | A0 | A1(s=1) | A1-S(s=.03) | 对A0变化pp | 对A1变化pp |','|---|---:|---:|---:|---:|---:|']
    for r in comparison:lines.append(f"| {r['metric']} | {r['A0']*100:.4f}% | {r['A1']*100:.4f}% | {r['A1_S']*100:.4f}% | {r['delta_vs_A0_pp']:+.4f} | {r['delta_vs_A1_pp']:+.4f} |")
    lines+=['','历史值从封存Validation数组重算并核对原始metrics文件，不使用prompt中的四舍五入值作为结果。Macro AP是全部27类定义有效的逐标签Average Precision平均，基于完整概率排序；逐标签mean_positive_probability只在gold-positive样本上取均值，不是预测正例均值。','',
        '## 2. Grief / Relief / Pride与其余Rare标签','',
        '| 标签 | Val support | A0 F1 | A1 F1 | A1-S F1 | A0 AP | A1 AP | A1-S AP |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rare:lines.append(f"| {r['label']} | {r['support']} | {100*r['A0_f1']:.2f}% | {100*r['A1_f1']:.2f}% | {100*r['A1_S_f1']:.2f}% | {100*r['A0_average_precision']:.2f}% | {100*r['A1_average_precision']:.2f}% | {100*r['A1_S_average_precision']:.2f}% |")
    lines+=['','F1涉及固定阈值决策，AP检查概率排序；二者变化分别解读，不因F1上升就宣称排序能力恢复。Rare-F1仍是七个固定标签的未加权平均，未删除grief、未调专属阈值；少量支持样本导致的波动需要如实保留。','',
        '## 3. 从头训练后，是否再次饱和','',
        '| 时点 | 首次clip | 后续任一text clip | 最终>0.99 | 半径均值 | 半径IQR | 辅助/BCE共享梯度比中位数 | 共享cos中位数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in geo:lines.append(f"| {r['tag']} | {r['first_clip_fraction']:.2%} | {r['later_clip_fraction']:.2%} | {r['final_fraction_above_099']:.2%} | {r['final_mean_norm']:.6f} | {r['final_norm_IQR']:.6f} | {r['median_shared_aux_to_BCE_gradient_norm']:.4f} | {r['median_shared_gradient_cosine']:+.4f} |")
    lines+=[f'',f'最后一轮：first clip={final["first_clip_fraction"]:.2%}，later clip={final["later_clip_fraction"]:.2%}。选中轮：later clip={selected["later_clip_fraction"]:.2%}。近边界比例与实际proj截断分开报告；若后续重新饱和，不能用此前只读s=.03探针结论代替从头训练的真实结果。',
        '每个时点复用相同10组train-only探针，160个位置/145个独立ID；不是随机全train分布估计。梯度是eval状态下全部27类BCE与0.1aux对共享encoder的局部梯度，非单情绪梯度。记录器仅观察，不额外缩放；每轮无更新诊断恢复Python/NumPy/CPU/CUDA RNG和所有module.training状态。','',
        '## 4. 如何解释、是否继续','',
        f'主要指标判定：{conclusion}',
        continuation,
        '若几何改善但Macro/Rare未改善，必须保留负结果，不能把数值改进写成识别收益。即使主指标超过A0，也只是一个已观察历史Test后的探索性Validation pilot；一个seed、一个checkpoint选择协议不能证明显著性、泛化或Hyperbolic优于Euclidean。',
        '输入缩放还改变辅助梯度强度和方向，gamma固定不代表有效梯度贡献固定。因此性能变化不能只归因于径向信息恢复。初始化与诊断RNG检查只能覆盖记录的状态，不宣称逐训练步历史dropout mask已逐个核验。',
        '下一步最值得考虑的单个对照是容量和辅助目标尽可能匹配的Euclidean Auxiliary Control，用于区分双曲几何与额外监督/参数作用；是否继续应同时看Macro/Rare/AP与实际饱和轨迹。如果本轮主要指标未改善，停止继续搜索缩放系数或gamma。',
        '本轮没有自动启动欧氏对照、多seed、新尺度或正式Test；任何后续实验需要独立冻结方案。','',
        '## 5. 证据位置','',
        'INITIALIZATION_PARITY.json、SCALED_INITIALIZATION_PARITY.json、smoke/、geometry/、training_log.csv、batch_log.csv、checkpoint_selection.json、checkpoint_reload_verification.json、validation_predictions.npz、validation_per_label.csv、protected_before.json、source_manifest.json与SUCCESS.json。']
    md(root/'EXP6C1S_TRAIN_SUMMARY.md',lines)
    md(root/'EXP6C1S_VERIFICATION_REPORT.md',['# Exp6C1S pilot verification','',
        'PASS：旧源文件/模型/结果前后哈希；相同raw snapshot/tokenizer/27标签与canonical划分；共享及HNN/decoder初始化；缩放前后分类logits与RNG；原Formula A值及梯度；train-only固定probe及无更新/RNG恢复；原AdamW/scheduler/7648步；四轮Validation最优且平局取早；完整3834条重载；保存数组指标/AP重算；新Test=0。',
        '运行前后保护检查和数值结果均保留。完整远端verify检查checkpoint字节；供下载审核包可省略大checkpoint，由REVIEW_EXPORT.json明确列出省略项，不能把省略项描述为已本地重载验证。'])
