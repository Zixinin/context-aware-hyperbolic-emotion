"""Figures and evidence-labelled research summary; no model-development decisions."""
import numpy as np
from analysis import require


def figures(out, labels, branches, scores, true_loss, negatives, activity, gradient, siblings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out / 'figures'
    folder.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.tight_layout()
        path = folder / name
        require(not path.exists(), 'Figure would overwrite an existing file')
        fig.savefig(path, dpi=180)
        plt.close(fig)
    def barh(rows, key, title, xlabel, name, multiplier=1.):
        fig, ax = plt.subplots(figsize=(9, max(4, .28*len(rows)+1.4)))
        values = [multiplier*r[key] for r in rows]
        ax.barh([r.get('label', r.get('branch')) for r in rows], values,
                color=['#b45454' if v < 0 else '#37828b' for v in values])
        ax.axvline(0, color='black', lw=.6)
        ax.set(title=title, xlabel=xlabel); ax.invert_yaxis()
        save(fig, name)
    barh(sorted(labels, key=lambda r: r['delta_f1']), 'delta_f1', 'Fixed threshold 0.30: Exp3 minus Exp2',
         'Delta F1 (percentage points)', '01_per_label_delta_f1.png', 100)
    barh(branches, 'delta_mean_f1', 'Frozen hierarchy branches', 'Mean Delta F1 (pp)',
         '02_branch_delta_f1.png', 100)
    def scatter(x, y, xlabel, ylabel, title, name, names=None, logx=False):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(x, y, s=28, color='#37828b', alpha=.8)
        ax.axhline(0, color='gray', lw=.6)
        if logx:
            ax.set_xscale('log')
        if names:
            # All points are retained; annotate only large shifts and grief/relief.
            annotated = set(np.argsort(np.abs(y))[-5:].tolist())
            annotated.update(i for i, label in enumerate(names) if label in ('grief', 'relief'))
            groups = {}
            for i in sorted(annotated):
                groups.setdefault((float(x[i]), float(y[i])), []).append(names[i])
            midpoint = (min(x)+max(x))/2
            for (px, py), group in groups.items():
                right = px > midpoint
                ax.annotate('\n'.join(group), (px, py), xytext=(-5 if right else 5, 6),
                            textcoords='offset points', ha='right' if right else 'left', fontsize=8)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        save(fig, name)
    names = [r['label'] for r in labels]
    freq = [r['train_frequency'] for r in labels]
    scatter(freq, [100*r['delta_f1'] for r in labels], 'Train frequency (log scale)', 'Delta F1 (pp)',
            'All 28 labels; descriptive, n=28', '03_frequency_vs_delta_f1.png', names, True)
    scatter(freq, [r['delta_fp'] for r in labels], 'Train frequency (log scale)', 'Delta false positives',
            'All 28 labels; descriptive, n=28', '03b_frequency_vs_delta_fp.png', names, True)
    barh(true_loss, 'raw_loss_share', 'Best-checkpoint TRAIN snapshot: actual term losses',
         'Share of raw peer loss (%)', '04_peer_loss_by_true_label.png', 100)
    barh(negatives[:10], 'raw_loss_sum', 'Best-checkpoint TRAIN snapshot: top 10 negatives by LOSS',
         'Sum of actual associated term loss', '05_hard_negative_loss_contribution.png')
    fig, ax = plt.subplots(figsize=(9, 5))
    scopes = ['full_shared', 'BERT_encoder', 'GAT_path', 'shared_MLM_anchor_transform']
    # Plot all 10 observations; avoid violin/KDE implying a well-estimated distribution.
    gradient_rows = []
    import csv
    with (out / 'gradient_conflict_batches.csv').open() as stream:
        gradient_rows = list(csv.DictReader(stream))
    for i, scope in enumerate(scopes):
        points = [float(r['cosine']) for r in gradient_rows if r['scope'] == scope and r['cosine'] != '']
        jitter = np.linspace(-.13, .13, len(points))
        ax.scatter(i+jitter, points, s=28, label=scope)
        if points:
            ax.plot([i-.2, i+.2], [np.mean(points)]*2, color='black', lw=2)
    ax.axhline(0, color='gray', lw=.8); ax.axhspan(-.05, .05, color='gray', alpha=.12)
    ax.set_xticks(range(len(scopes)), ['Full shared', 'BERT encoder', 'GAT path', 'MLM anchor transform'])
    ax.set(ylabel='Gradient cosine', ylim=(-1.05, 1.05), title='Original LGSA vs peer: 10 fixed train batches at epoch 3')
    save(fig, '06_gradient_cosine_distribution.png')
    scatter([r['delta_gt_positive_score'] for r in scores], [r['delta_gt_negative_score'] for r in scores],
            'Delta mean score on GT positives', 'Delta mean score on GT negatives',
            'Saved test probabilities: Exp3 minus Exp2', '07_score_shift.png', [r['label'] for r in scores])
    part = [r for r in siblings if r['delta_sibling_margin'] is not None]
    scatter([r['delta_sibling_margin'] for r in part], [100*r['delta_f1'] for r in part],
            'Delta probability sibling margin', 'Delta label F1 (pp)', 'Label-wise relation; descriptive only',
            '08_sibling_margin_vs_f1.png', [r['label'] for r in part])
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ['0', '1', '2', '3+']; x = np.arange(4)
    for offset, tag, color in [(-.18, 'exp2', '#777777'), (.18, 'exp3', '#37828b')]:
        ax.bar(x+offset, [100*activity[tag]['count_distribution'][k]['fraction'] for k in categories],
               width=.36, label=tag.upper(), color=color)
    ax.set_xticks(x, categories)
    ax.set(xlabel='Predicted labels per test example', ylabel='Examples (%)', title='Fixed threshold 0.30')
    ax.legend(); save(fig, '09_prediction_label_count_distribution.png')
    require(len(list(folder.glob('*.png'))) == 10, 'Required figures incomplete')


def format_number(value, precision=4):
    return 'unavailable' if value is None else f'{value:.{precision}f}'


def markdown_table(rows, columns):
    lines = ['| ' + ' | '.join(title for key, title in columns) + ' |',
             '|' + '|'.join(['---']*len(columns)) + '|']
    for row in rows:
        cells = []
        for key, title in columns:
            value = row[key]
            cells.append(format_number(value) if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def write_report(out, summary, rows, parents, true_loss, negative_loss, val_rows):
    """A strategic stopping recommendation is not a proof that DPT is ineffective."""
    active, gains, gradients = summary['main_metrics'], summary['gain_loss'], summary['gradient_scopes']
    validation = summary['validation_epoch3_4']
    main = [{'metric': k, 'Exp2': 100*active['exp2'][k], 'Exp3': 100*active['exp3'][k],
             'delta_pp': 100*active['delta'][k]} for k in
            ('micro_precision', 'micro_recall', 'micro_f1', 'macro_f1', 'rare_mean_f1', 'nonrare_mean_f1')]
    lines = ['# Exp3B：实验三只读诊断（seed42）', '',
        '本次没有训练新模型、没有优化器更新，没有调 λ、阈值、层级或 sibling pairs。'
        '测试集只用于解释已经完成的结果；所有诊断写入新目录。', '',
        '## 证据范围', '',
        '- Exp2/Exp3：原有 seed42 测试预测，固定阈值 0.30；同一批 IDs 和 GT。',
        '- 实际 peer loss：best epoch3 固定权重，对全部 43,410 条现有训练样本做一次前向回放。'
        '原始训练模式的 dropout/MLM 保留，固定随机顺序和 batch seed，无梯度更新。'
        '**这是 checkpoint 快照，不是历史训练每一步的累计贡献。**',
        '- 梯度方向：epoch3 的预先固定 10 个训练 batch，两个损失来自同一 forward graph。'
        'original = MLM + classification + 饱和 ramp 的 HGA/GOF；只比较同时连接两损失的可训练参数。',
        '- 验证集：复用先前完成的 epoch3/4 预测及余弦矩阵，不再次推理。',
        f"- 完整性：{summary['protected_file_count']} 个既有结果/诊断/资产文件字节哈希保持不变，"
        '冻结 Exp3/LGSA 源码复核通过；模型全部参数及 buffers 精确不变。', '',
        '## 主指标', '']
    lines += markdown_table(main, [('metric', '指标'), ('Exp2', 'Exp2 (%)'), ('Exp3', 'Exp3 (%)'), ('delta_pp', '变化 (pp)')])
    lines += ['', '## Q1 哪些标签获益、哪些退步？', '',
        f"28 个标签：提升 {gains['all']['improved']}，下降 {gains['all']['decreased']}，持平 {gains['all']['tied']}。",
        '提升：' + ', '.join(gains['all']['improved_labels']),
        '下降：' + ', '.join(gains['all']['decreased_labels']),
        '持平：' + (', '.join(gains['all']['tied_labels']) or '无'), '', '前五提升 / 下降（F1 变化单位 pp）：', '']
    extremes = [{'label': r['label'], 'support': r['support'], 'delta_pp': 100*r['delta_f1']} for r in
                gains['top5_improved']+gains['top5_degraded']]
    lines += markdown_table(extremes, [('label', '标签'), ('support', '支持数'), ('delta_pp', '变化 (pp)')])
    lines += ['', 'Rare / nonrare / sibling-eligible / neutral 的全部标签名单与变化计数：'
              '`per_label_gain_loss_summary.json`；逐标签 TP/FP/FN 见完整 CSV。', '',
              '## Q2 退步是否集中在某个分支？', '']
    br = [{'branch': r['branch'], 'labels': r['fine_labels'], 'support': r['total_support'],
           'delta_f1_pp': 100*r['delta_mean_f1'], 'delta_p_pp': 100*r['delta_mean_precision'],
           'delta_r_pp': 100*r['delta_mean_recall'], 'delta_fp': r['delta_total_fp'], 'delta_fn': r['delta_total_fn']}
          for r in summary['branches']]
    lines += markdown_table(br, [('branch', '分支'), ('labels', '标签数'), ('support', 'GT 实例数'),
        ('delta_f1_pp', '平均 F1 变化 pp'), ('delta_p_pp', '平均 P 变化 pp'), ('delta_r_pp', '平均 R 变化 pp'),
        ('delta_fp', 'FP 变化'), ('delta_fn', 'FN 变化')])
    worst = min(br, key=lambda r: r['delta_f1_pp'])
    lines += ['', f"平均 F1 最低变化的分支是 {worst['branch']}（{worst['delta_f1_pp']:+.3f} pp）。"
              '这与 term 数量占比是不同测量，不能由频率直接推断它受到了损害。', '',
              '## Q3 高频标签是否主导实际 peer loss？', '',
              '有有效 peer terms 的标签中，训练频率与快照 loss 总贡献的 Spearman：' +
              format_number(summary['peer_loss_snapshot']['frequency_correlations']['raw_loss_sum']['rho']) +
              '；与平均 term loss 的 Spearman：' +
              format_number(summary['peer_loss_snapshot']['frequency_correlations']['raw_loss_mean']['rho']) + '。',
              f"按实际 loss 排名前五的 true labels 占原始 loss "
              f"{100*summary['peer_loss_snapshot']['top5_true_loss_share']:.2f}%。"
              '总贡献集中可能来自频率，也可能来自每项难度；必须同时读平均 loss，不等同于梯度/优化主导。', '']
    lines += markdown_table([{'branch': r['label'], 'count_pct': 100*r['term_count_share'],
        'raw_pct': 100*r['raw_loss_share'], 'weighted_pct': 100*r['weighted_batch_loss_share']} for r in parents],
        [('branch', '分支'), ('count_pct', 'term 数量 %'), ('raw_pct', '实际原始 loss %'), ('weighted_pct', '归一化加权 loss %')])
    lines += ['', '加权项 = 0.5 × term loss / 当 batch 有效 term 数；比例分母是全部加权 peer 项，'
              '不是包含 MLM/分类/GOF 的 total loss。历史总损失占比 5.373% 是另一项已有证据。', '',
              '## Q4 是否少数 hard negatives 主导实际 loss？', '',
              f"前五 / 前十 negative labels 占实际原始 loss "
              f"{100*summary['peer_loss_snapshot']['top5_negative_loss_share']:.2f}% / "
              f"{100*summary['peer_loss_snapshot']['top10_negative_loss_share']:.2f}%。以下按 LOSS 排序，不按次数排序。", '']
    lines += markdown_table([{'label': r['label'], 'count': r['eligible_terms'], 'mean': r['raw_loss_mean'],
        'sum': r['raw_loss_sum'], 'share_pct': 100*r['raw_loss_share']} for r in negative_loss[:10]],
        [('label', '负标签'), ('count', '次数'), ('mean', '平均项 loss'), ('sum', 'loss 总和'), ('share_pct', '占比 %')])
    lines += ['', '## Q5 原 LGSA 与 peer 是否梯度冲突？', '']
    lines += markdown_table([{'scope': scope, 'mean': r['mean'], 'sd': r['sd_sample'], 'min': r['min'], 'max': r['max'],
        'negative': r['fraction_negative'], 'near': r['fraction_near_zero_abs_le_0.05'], 'positive': r['fraction_positive']}
        for scope, r in gradients.items()], [('scope', '参数范围'), ('mean', '平均 cosine'), ('sd', '样本 SD'),
        ('min', '最小'), ('max', '最大'), ('negative', '<0 比例'), ('near', '|cos|≤0.05 比例'), ('positive', '>0 比例')])
    full = gradients['full_shared']
    conflict = full['mean'] < 0 and full['fraction_negative'] >= .6
    lines += ['', ('该快照有重复的全局负方向冲突迹象。' if conflict else
                  '该快照不能支持“普遍全局负方向冲突”的判断；也请检查模块差异和接近零的比例。'),
              '梯度结果只描述这 10 个 batch 和一个 checkpoint，不是致因证明。'
              '冻住的 raw HGS/prompt rows 不参与，MLM anchor transform 才是表中可训练的共享变换。'
              'near-zero 与严格 <0/>0 两列可重叠，不能相加为互斥类别。', '',
              '## Q6 是否只是预测更多标签？', '',
              f"每条样本预测标签均数：{active['exp2']['average_predicted_label_count']:.4f} → "
              f"{active['exp3']['average_predicted_label_count']:.4f}；预测正项总数："
              f"{active['exp2']['total_predicted_positives']} → {active['exp3']['total_predicted_positives']}。"
              f"TP {active['exp2']['tp']} → {active['exp3']['tp']}；FP {active['exp2']['fp']} → {active['exp3']['fp']}。",
              ('更多预测正项伴随更多 TP 和 FP，描述上符合 recall 上升 / precision 下降。' if
               active['delta']['total_predicted_positives'] > 0 and active['delta']['fp'] > 0 else
               '不能单凭预测活跃度解释全部 P/R 变化；需联合 TP/FP 与逐标签 score shift。'),
              '0/1/2/3+ 分布见 JSON 和图；单 GT / 多 GT 以及实际 peer-eligible 样本分组均已分别保存。', '',
              '## Q7 sibling 改善是否对应 label F1 改善？', '']
    for key, corr in summary['sibling_correlations'].items():
        lines.append(f"- {key} 与 ΔF1：Spearman={format_number(corr['rho'])}，n={corr['n']}。")
    lines += ['', '概率 margin 的相关系数若不高，只能说明该批标签的平均 margin 不是充分的性能代理。'
              '即使相关，也不能证明优化 margin 会改善 F1。尤其平均 margin 不等同于 0.30 阈值跨越。']
    for tag, r in summary['error_context'].items():
        lines.append(f"- {tag}: FP 中 sibling-linked 占 {100*r['sibling_linked_fp_fraction']:.2f}%；"
                     f"FN 中没有被预测为正的错误 sibling 的数量 {r['fn_without_predicted_fp_sibling']}/{r['total_fn']}。")
    lines += ['', '这些是错误的描述性共现分类，不能把对应比例直接当作 sibling confusion 的因果贡献。', '',
              '## Q8 grief 为什么提升、relief 为什么退步？', '']
    for r in summary['grief_relief']:
        lines.append(f"- {r['label']} (support={r['support']}): TP {r['exp2_tp']}→{r['exp3_tp']}，"
            f"FP {r['exp2_fp']}→{r['exp3_fp']}，FN {r['exp2_fn']}→{r['exp3_fn']}，"
            f"F1 {100*r['exp2_f1']:.2f}%→{100*r['exp3_f1']:.2f}%；GT 正项平均分数 "
            f"{r['exp2_gt_positive_score']:.4f}→{r['exp3_gt_positive_score']:.4f}，"
            f"对所有非 GT 标签最高分的平均 margin "
            f"{r['exp2_margin_to_top_wrong_all_labels']:.4f}→{r['exp3_margin_to_top_wrong_all_labels']:.4f}。"
            f"历史负标签次数：{r['historical_negative_counts']}。")
    lines += ['', 'TP/FP/FN 给出数值上的直接解释；分数与负标签只补充描述。'
              '6 / 11 个支持样本不足以判断普遍机制，具体信息保留在 grief_relief CSV。', '',
              '## Q9 epoch4 的 margin 更好，Macro-F1 为什么更差？', '',
              f"验证 Macro-F1：{100*validation['epoch3']['macro_f1']:.4f}% → "
              f"{100*validation['epoch4']['macro_f1']:.4f}%。"
              '平均 margin 提高不保证每个标签的阈值判定或排序反转改善。',
              '概率 sibling margin 提高但 F1 下降的标签：' +
              (', '.join(validation['labels_margin_improved_f1_decreased']) or '无') + '。',
              '余弦 margin 提高但 F1 下降的标签：' +
              (', '.join(validation['cosine_margin_improved_f1_decreased']) or '无') + '。', '']
    val_lost = sorted([r for r in val_rows if r['delta_f1'] < 0], key=lambda r: r['delta_f1'])[:5]
    lines += markdown_table([{'label': r['label'], 'f1_pp': 100*r['delta_f1'],
        'margin': r['delta_sibling_margin'], 'fp': r['delta_fp'], 'fn': r['delta_fn']} for r in val_lost],
        [('label', '验证退步前五'), ('f1_pp', 'F1变化 pp'), ('margin', 'sibling概率margin变化'), ('fp', 'FP变化'), ('fn', 'FN变化')])
    lines += ['', '逐标签、分支和反转率全表已保存。不能仅凭这个现象确定过拟合、λ 过大或 epoch 太少。', '',
              '## Q10 哪种解释最受支持？', '']
    comparison = {r['metric']: r for r in validation['existing_replay_comparison']}
    local_insufficiency = (comparison['mean_probability_margin']['delta_epoch4_minus_epoch3'] > 0 and
                           comparison['macro_f1']['delta_epoch4_minus_epoch3'] < 0)
    lines += [('C（局部 sibling separation 不是整体分类改进的充分条件）由该验证对比直接支持。'
               '这不等同于证明 sibling confusion 不是主要错误来源。' if local_insufficiency else
               'E：该验证对比没有复核出 margin 改善而 Macro-F1 下降的现象，不能强行指定 C。')
              + ('A 有快照梯度冲突的补充证据，但尚无反事实实验确认它造成 F1 退步。' if conflict else
                 'A 的全局冲突判断未得到该快照的一致支持，局部模块结果仍仅作描述。') +
              'B 可以从实际 loss 表中评估集中程度，但集中本身不能证明有害或不合理。'
              '因此不能把 A/B/C 强行归为唯一因果机制；若讨论混合解释，A/B 仍是待验证假设。', '',
              '## NEXT_STEP_RECOMMENDATION', '',
              ('STOP_DPT_BRANCH' if local_insufficiency else 'INSUFFICIENT_EVIDENCE'), '',
              ('当前诊断可以定位数值变化和局部方向，却没有验证一个能用单一改动可靠解决的原因。'
              '在已经完成的 seed42 pilot 与该快照范围内，建议停止继续优化 DPT 分支，'
              '把研究精力回到 emotion hierarchy 与 hyperbolic label representation。'
              '这是结合主线优先级的资源决策，不是证明 DPT 思想普遍无效。'
              '没有建议或自动执行 test-driven λ、阈值、pair 调整，也没有启动下一实验。' if local_insufficiency else
              '验证对比没有复核预期现象，应先核对这一证据差异，不得用当前报告启动新模型实验。'), '',
              '## 文件与可追溯性', '',
              '新增：独立 experiments/exp3b_diagnostics 源码与本输出目录。修改旧文件：0。'
              '复用路径、每个输入哈希、源码版本和运行前工作区状态见 provenance_before.json；'
              '所有新表/图/报告哈希见 SUCCESS.json。冻结实验结果和既有诊断完整性均通过。', '',
              'CSV 数值：P/R/F1/相关系数/占比为原始比例，图与本报告的 pp 列乘以 100。'
              '无预测或无 GT 的分类指标使用 zero_division=0；无法定义的 score/gradient/correlation 用空值，不伪造为零。', '']
    with (out / 'EXP3B_DIAGNOSTIC_SUMMARY.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    return 'STOP_DPT_BRANCH' if local_insufficiency else 'INSUFFICIENT_EVIDENCE'
