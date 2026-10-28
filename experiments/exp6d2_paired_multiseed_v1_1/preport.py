"""Three-seed descriptive paired analysis; no seed selection or p-value fishing."""
import numpy as np
from pcommon import *
METRICS=['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','weighted_f1','rare_f1','macro_average_precision']

def compute_statistics(records):
    need(len(records)==9 and {(r['seed'],r['variant']) for r in records}=={(s,k) for s in ALL_SEEDS for k in KINDS},'Need exactly nine unique preregistered cells')
    by={(r['seed'],r['variant']):r for r in records};summary=[];paired=[]
    for kind in KINDS:
        for metric in METRICS:
            v=np.array([by[s,kind][metric] for s in ALL_SEEDS],dtype=float)
            summary.append(dict(variant=kind,metric=metric,n_seeds=3,mean=float(v.mean()),sample_sd_ddof1=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max())))
    for comparator in ['A1S','A0']:
        for metric in METRICS:
            d=np.array([100*(by[s,'N1'][metric]-by[s,comparator][metric]) for s in ALL_SEEDS])
            paired.append(dict(comparison='N1-'+comparator,metric=metric,n_pairs=3,seed42_delta_pp=float(d[0]),seed123_delta_pp=float(d[1]),seed2026_delta_pp=float(d[2]),
                mean_delta_pp=float(d.mean()),sample_sd_delta_pp_ddof1=float(d.std(ddof=1)),minimum_delta_pp=float(d.min()),maximum_delta_pp=float(d.max()),
                positive_seeds=int((d>0).sum()),tied_seeds=int((d==0).sum()),negative_seeds=int((d<0).sum())))
    return summary,paired

def write_reports(suite):
    runs=[];per=[]
    for seed in ALL_SEEDS:
        for kind in KINDS:
            if seed==42:
                m=read(suite/'seed42_reused'/(kind+'_metrics.json'));rows=csvrows(suite/'seed42_reused'/(kind+'_per_label.csv'))
                origin=read(suite/'seed42_reused'/(kind+'_PROVENANCE.json'));epoch=origin['checkpoint_selection']['epoch'];path=origin['original_run']
            else:
                root=run_path(suite,seed,kind);m=read(root/'validation_metrics.json');rows=csvrows(root/'validation_per_label.csv');epoch=read(root/'checkpoint_selection.json')['epoch'];path=str(root)
            runs.append(dict(seed=seed,variant=kind,**{k:m[k] for k in METRICS},selected_epoch=epoch,reused=seed==42,source_run=path))
            for row in rows:
                per.append(dict(seed=seed,variant=kind,label=row['label'],support=int(row['support']),
                    **{k:float(row[k]) for k in ['precision','recall','f1','average_precision']},**{k:int(row[k]) for k in ['tp','fp','fn']}))
    summary,paired=compute_statistics(runs)
    table(suite/'EXP6D2_NINE_RUN_VALIDATION.csv',runs);table(suite/'EXP6D2_MODEL_MEAN_SD.csv',summary);table(suite/'EXP6D2_PAIRED_DELTAS.csv',paired)
    table(suite/'EXP6D2_PER_LABEL_ALL_SEEDS.csv',per);table(suite/'EXP6D2_RARE_ALL_SEEDS.csv',[r for r in per if r['label'] in a0.RARE])
    by={(r['seed'],r['variant'],r['label']):r for r in per};contributions=[];per_deltas=[]
    for seed in ALL_SEEDS:
        for label in read(suite/'dataset_manifest.json')['names27']:
            a=by[seed,'A1S',label];n=by[seed,'N1',label];f=100*(n['f1']-a['f1']);ap=100*(n['average_precision']-a['average_precision'])
            per_deltas.append(dict(seed=seed,label=label,support=n['support'],F1_delta_pp=f,AP_delta_pp=ap,macro_contribution_pp=f/27,rare_contribution_pp=f/7 if label in a0.RARE else 0.))
        records=[r for r in per_deltas if r['seed']==seed];grief=next(r for r in records if r['label']=='grief')
        contributions.append(dict(seed=seed,macro_delta_pp=sum(r['macro_contribution_pp'] for r in records),grief_macro_contribution_pp=grief['macro_contribution_pp'],
            other26_macro_contribution_pp=sum(r['macro_contribution_pp'] for r in records if r['label']!='grief'),
            rare_delta_pp=sum(r['rare_contribution_pp'] for r in records),grief_rare_contribution_pp=grief['rare_contribution_pp'],
            other6_rare_contribution_pp=sum(r['rare_contribution_pp'] for r in records if r['label']!='grief')))
    table(suite/'EXP6D2_LABEL_PAIRED_DELTAS.csv',per_deltas);table(suite/'EXP6D2_GRIEF_CONTRIBUTION.csv',contributions)
    rows=['# Exp6D-2：三 seed 配对验证','',
        '执行版本v1.1：原v1在prepare误读A0从未保存的完整顺序文件而停止，当时尚无smoke/正式训练。失败目录和原v1源码保留；只修正历史顺序证据提取，六次训练协议与损失公式未变。','',
        '新增六次训练：A0/A1-S/N1各seed123、2026；seed42读取冻结旧结果，不重训。每个新run从相同预训练snapshot独立开始，4epochs/7648step，Validation Macro-F1 earliest tie，固定阈值.30；新Test=0。','',
        '## 九个原始Validation结果','',
        '|Seed|模型|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|选中epoch|来源|','|---|---|---:|---:|---:|---:|---:|---|']
    for r in runs:rows.append(f'|{r["seed"]}|{r["variant"]}|{100*r["micro_f1"]:.4f}|{100*r["macro_f1"]:.4f}|{100*r["rare_f1"]:.4f}|{100*r["macro_average_precision"]:.4f}|{r["selected_epoch"]}|'+('冻结复用' if r['reused'] else '新增独立训练')+'|')
    rows+=['','## 三seed均值 ± 样本标准差','',
        '|模型|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|','|---|---|---|---|---|']
    for kind in KINDS:
        vals=[]
        for k in ['micro_f1','macro_f1','rare_f1','macro_average_precision']:
            r=next(r for r in summary if r['variant']==kind and r['metric']==k);vals.append(f'{100*r["mean"]:.4f} ± {100*r["sample_sd_ddof1"]:.4f}')
        rows.append('|'+kind+'|'+'|'.join(vals)+'|')
    rows+=['','SD采用ddof=1；统计重复单位是三个预定训练seed，不是3834条Validation文本、epoch或27个标签。相同seed为配对区组，不把九次运行当作九个独立seed。没有挑seed或按结果排除失败/负向seed。','',
        '## 配对变化（单位pp）','',
        '|比较|指标|seed42|seed123|seed2026|配对均值 ± SD|正/平/负seed数|','|---|---|---:|---:|---:|---|---|']
    for r in paired:
        if r['metric'] not in ['micro_f1','macro_f1','rare_f1','macro_average_precision']:continue
        rows.append(f'|{r["comparison"]}|{r["metric"]}|{r["seed42_delta_pp"]:+.4f}|{r["seed123_delta_pp"]:+.4f}|{r["seed2026_delta_pp"]:+.4f}|{r["mean_delta_pp"]:+.4f} ± {r["sample_sd_delta_pp_ddof1"]:.4f}|{r["positive_seeds"]}/{r["tied_seeds"]}/{r["negative_seeds"]}|')
    rows+=['','## 两个问题分别回答','']
    for comparator,question in [('A1S','N1相对缩放辅助分支的收益是否跨seed重复'),('A0','整套N1方案是否超过普通BCE')]:
        r=next(r for r in paired if r['comparison']=='N1-'+comparator and r['metric']=='macro_f1')
        consistency='三个预定seed的Macro差值方向均为正' if r['positive_seeds']==3 else '不同seed并非全部出现正向Macro差值'
        rows.append(f'- **{question}：** {consistency}；配对Macro均值差{r["mean_delta_pp"]:+.4f}pp，SD={r["sample_sd_delta_pp_ddof1"]:.4f}pp。结合Rare/AP及下列单标签贡献解释，不由均值单独判定普遍有效。')
    rows+=['','## grief与其他标签的贡献分解','',
        '|Seed|Macro总变化|grief贡献|其余26类贡献|Rare总变化|grief贡献|其余6个Rare贡献|','|---|---:|---:|---:|---:|---:|---:|']
    for r in contributions:rows.append('|'+str(r['seed'])+'|'+'|'.join(f'{r[k]:+.4f}' for k in ['macro_delta_pp','grief_macro_contribution_pp','other26_macro_contribution_pp','rare_delta_pp','grief_rare_contribution_pp','other6_rare_contribution_pp'])+'|')
    rows+=['','这是对完整27类/固定7个Rare指标的代数分解，没有删除grief或更换正式指标。逐标签F1、AP、TP/FP/FN及paired变化在CSV中全量报告，尤其保留grief/relief/pride的低支持波动。','',
        '## 解释边界与停止','',
        '配对seed控制初始化与shuffle种子，新增六个run核对初始分类logits及每轮完整训练顺序。A1-S/N1的HNN/decoder初值相同；A0没有辅助dropout，后续随机掩码轨迹不宣称逐步一致。seed42的A1-S/N1有已封存的四轮完整顺序可核对；旧A0只保存每轮首batch，四个首batch已核对，完整顺序仅通过冻结DataLoader重放重建，不声称它是保存过的全程历史记录。没有给旧A0补写文件。',
        'n=3只描述这三个seed下的方向与波动；同一Validation集合已被多轮探索，均值提升不能称显著、最终泛化或几何机制证明。本轮不计算探索性p值，不挑显著的标签或指标，不把mean±SD当置信区间。',
        '本轮只做稳定性验证，到此停止。Euclidean control仍是后续独立任务：需匹配参数量、标签先验、辅助目标、loss尺度和训练预算；没有自动实施。最终方法与协议确定后才考虑预先冻结的正式测试，本轮无新Test。']
    md(suite/'EXP6D2_SUMMARY_CN.md',rows)
