"""Descriptive paired n=3 analysis only. No tests of statistical significance."""
import numpy as np
from mslib import *

SETTINGS = ['fixed_0.30', 'validation_selected']
MAIN = ['micro_f1', 'macro_f1', 'rare_f1', 'nonrare_f1']


def stats(values):
    a = np.asarray(values, dtype=float)
    require(a.shape == (3,) and np.isfinite(a).all(), 'Three finite paired seeds required')
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)), 'min': float(a.min()), 'max': float(a.max())}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def generate(records, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    overall, curves, thresholds, perlabel = [], [], [], {}
    lookup = {}
    for r in records:
        model, seed, path = r['model'], r['seed'], Path(r['path'])
        for setting in SETTINGS:
            m = read(path / ('test_' + setting + '_metrics.json'))
            row = {'model': model.upper(), 'seed': seed, 'threshold_setting': setting,
                   'threshold': m['threshold'], **{k: m[k] for k in METRICS}}
            overall.append(row)
            lookup[setting, model, seed] = row
            perlabel[setting, model, seed] = csvread(path / ('per_label_' + setting + '_metrics.csv'))
        curve = csvread(path / 'validation_metrics_by_epoch.csv')
        curves += [{'model': model.upper(), 'seed': seed, 'epoch': int(row['epoch']),
                    'validation_micro_f1': float(row['validation_micro_f1']),
                    'validation_macro_f1': float(row['validation_macro_f1']),
                    'best_epoch': r['metadata']['best_epoch']} for row in curve]
        thresholds.append({'model': model.upper(), 'seed': seed,
                           'selected_threshold': r['metadata']['selected_threshold'],
                           'best_epoch': r['metadata']['best_epoch'],
                           'validation_macro_nondecreasing': all(float(b['validation_macro_f1']) >= float(a['validation_macro_f1']) for a, b in zip(curve, curve[1:]))})
    require(len(overall) == 12 and len(curves) == 24, 'Incomplete result matrix')
    csvwrite(out / 'overall_summary.csv', overall)
    csvwrite(out / 'validation_curves_all_seeds.csv', curves)
    csvwrite(out / 'threshold_stability.csv', thresholds)
    means, paired, pairs = [], [], {}
    for setting in SETTINGS:
        for model in ['bert', 'lgsa']:
            for metric in METRICS:
                means.append({'threshold_setting': setting, 'model': model.upper(), 'metric': metric,
                              **stats([lookup[setting, model, seed][metric] for seed in SEEDS])})
        for seed in SEEDS:
            row = {'threshold_setting': setting, 'seed': seed,
                   **{k: lookup[setting, 'lgsa', seed][k] - lookup[setting, 'bert', seed][k] for k in METRICS}}
            paired.append(row)
            pairs[setting, seed] = row
        paired.append({'threshold_setting': setting, 'seed': 'mean_delta',
                       **{k: float(np.mean([pairs[setting, s][k] for s in SEEDS])) for k in METRICS}})
    csvwrite(out / 'multiseed_mean_std.csv', means)
    csvwrite(out / 'paired_seed_deltas.csv', paired)
    delta_stats = [{'threshold_setting': setting, 'metric': k,
                    **stats([pairs[setting, s][k] for s in SEEDS]),
                    'lgsa_wins': sum(pairs[setting, s][k] > 0 for s in SEEDS),
                    'ties': sum(pairs[setting, s][k] == 0 for s in SEEDS)} for setting in SETTINGS for k in METRICS]
    csvwrite(out / 'paired_delta_stability.csv', delta_stats)
    label_rows, rare_rows, grief_relief, count_rows = [], [], [], []
    for setting in SETTINGS:
        for i, name in enumerate(NAMES):
            b = [float(perlabel[setting, 'bert', s][i]['f1']) for s in SEEDS]
            l = [float(perlabel[setting, 'lgsa', s][i]['f1']) for s in SEEDS]
            d = np.asarray(l) - b
            row = {'threshold_setting': setting, 'label_id': i, 'label': name,
                   'is_rare': i in RARE, 'support': int(perlabel[setting, 'bert', 42][i]['support']),
                   **{'bert_' + k: v for k, v in stats(b).items()},
                   **{'lgsa_' + k: v for k, v in stats(l).items()},
                   'mean_delta': float(d.mean()), 'delta_std': float(d.std(ddof=1)),
                   'lgsa_wins': int((d > 0).sum()), 'lgsa_losses': int((d < 0).sum()), 'ties': int((d == 0).sum()),
                   **{f'bert_seed{s}_f1': b[j] for j, s in enumerate(SEEDS)},
                   **{f'lgsa_seed{s}_f1': l[j] for j, s in enumerate(SEEDS)}}
            label_rows.append(row)
            if i in RARE:
                rare_rows.append(row)
            if name in ['grief', 'relief']:
                for model in ['bert', 'lgsa']:
                    for s in SEEDS:
                        values = perlabel[setting, model, s][i]
                        grief_relief.append({'threshold_setting': setting, 'label': name, 'model': model.upper(), 'seed': s,
                            **{k: values[k] for k in ['support', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'number_predicted_positive']}})
        for s in SEEDS:
            delta = np.array([float(perlabel[setting, 'lgsa', s][i]['f1']) - float(perlabel[setting, 'bert', s][i]['f1']) for i in range(28)])
            count_rows.append({'threshold_setting': setting, 'seed': s, 'improved': int((delta > 0).sum()),
                               'decreased': int((delta < 0).sum()), 'tied': int((delta == 0).sum())})
    csvwrite(out / 'per_label_multiseed_comparison.csv', sorted(label_rows, key=lambda r: (SETTINGS.index(r['threshold_setting']), -r['mean_delta'], r['label_id'])))
    csvwrite(out / 'per_label_multiseed_comparison_by_wins.csv', sorted(label_rows, key=lambda r: (SETTINGS.index(r['threshold_setting']), -r['lgsa_wins'], -r['mean_delta'], r['label_id'])))
    csvwrite(out / 'rare_label_seed_stability.csv', rare_rows)
    csvwrite(out / 'grief_relief_seed_analysis.csv', grief_relief)
    csvwrite(out / 'improved_label_counts_by_seed.csv', count_rows)
    distribution = [{'threshold_setting': setting,
                     **{f'labels_improved_in_{n}_of_3': sum(r['lgsa_wins'] == n for r in label_rows if r['threshold_setting'] == setting) for n in [3, 2, 1, 0]},
                     'labels_decreased_in_all_3': sum(r['lgsa_losses'] == 3 for r in label_rows if r['threshold_setting'] == setting)} for setting in SETTINGS]
    csvwrite(out / 'label_win_distribution.csv', distribution)
    tradeoff = [{'threshold_setting': setting, 'seed': s, 'delta_micro_precision': pairs[setting, s]['micro_precision'],
                'delta_micro_recall': pairs[setting, s]['micro_recall'],
                'recall_up_precision_down': pairs[setting, s]['micro_recall'] > 0 and pairs[setting, s]['micro_precision'] < 0}
                for setting in SETTINGS for s in SEEDS]
    csvwrite(out / 'precision_recall_tradeoff.csv', tradeoff)
    ranks = [{'seed': s, 'metric': k, 'fixed_delta': pairs['fixed_0.30', s][k],
              'validation_selected_delta': pairs['validation_selected', s][k],
              'ranking_changed': bool(np.sign(pairs['fixed_0.30', s][k]) != np.sign(pairs['validation_selected', s][k]))}
             for s in SEEDS for k in MAIN]
    csvwrite(out / 'threshold_ranking_comparison.csv', ranks)
    make_figures(out / 'figures', means, lookup, label_rows, curves)
    write_summary(out, records, means, pairs, delta_stats, rare_rows, grief_relief, label_rows,
                  count_rows, distribution, tradeoff, thresholds, ranks)
    return {'scientific_runs': 6, 'overall_rows': len(overall), 'validation_curve_rows': len(curves),
            'std_ddof': 1, 'threshold_settings': SETTINGS, 'statistical_tests': 'none',
            'plot_count': len(list((out / 'figures').glob('*.png')))}


def make_figures(out, means, lookup, labels, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir()
    colors = ['#4477AA', '#EE7733']
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + '.png'), dpi=200)
        fig.savefig(out / (name + '.pdf'))
        plt.close(fig)
    for setting in SETTINGS:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(4)
        for j, model in enumerate(['BERT', 'LGSA']):
            rows = [next(r for r in means if r['model'] == model and r['threshold_setting'] == setting and r['metric'] == k) for k in MAIN]
            ax.bar(x + (j-.5)*.34, [r['mean']*100 for r in rows], .34,
                   yerr=[r['std']*100 for r in rows], capsize=4, label=model, color=colors[j])
        ax.set_xticks(x, ['Micro-F1', 'Macro-F1', 'Rare F1', 'Non-rare F1'])
        ax.set_ylabel('F1 (%)'); ax.set_ylim(0, 100)
        ax.set_title(setting + ': mean ± sample SD (three seeds)'); ax.legend()
        save(fig, 'overall_mean_std_' + setting)
        for metric in ['macro_f1', 'rare_f1']:
            fig, ax = plt.subplots(figsize=(7, 5))
            for s in SEEDS:
                ax.plot([0, 1], [lookup[setting, m, s][metric]*100 for m in ['bert', 'lgsa']], marker='o', label='Seed ' + str(s))
            ax.set_xticks([0, 1], ['BERT', 'LGSA']); ax.set_xlim(-.2, 1.2)
            ax.set_ylabel(metric + ' (%)'); ax.set_title('Paired seeds: ' + setting); ax.legend()
            save(fig, metric + '_paired_' + setting)
        rare = [next(r for r in labels if r['label_id'] == i and r['threshold_setting'] == setting) for i in RARE]
        fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(7)
        for j, model in enumerate(['bert', 'lgsa']):
            ax.bar(x + (j-.5)*.34, [r[model+'_mean']*100 for r in rare], .34,
                   yerr=[r[model+'_std']*100 for r in rare], capsize=3, label=model.upper(), color=colors[j])
        ax.set_xticks(x, [r['label'] for r in rare], rotation=25, ha='right')
        ax.set_ylabel('F1 (%)'); ax.set_ylim(0, 105); ax.legend(); ax.set_title('Rare labels: ' + setting + ' (mean ± sample SD)')
        save(fig, 'rare_labels_mean_std_' + setting)
        rows = sorted([r for r in labels if r['threshold_setting'] == setting], key=lambda r: r['mean_delta'])
        fig, ax = plt.subplots(figsize=(9, 10))
        ax.barh([r['label'] for r in rows], [r['mean_delta']*100 for r in rows],
                color=[colors[1] if r['mean_delta'] > 0 else colors[0] for r in rows])
        ax.axvline(0, color='black', linewidth=.8); ax.set_xlabel('Mean paired Δ F1 (percentage points)')
        ax.set_title('LGSA − BERT: ' + setting)
        save(fig, 'all_labels_mean_delta_' + setting)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, ['BERT', 'LGSA']):
        for s in SEEDS:
            rows = [r for r in curves if r['model'] == model and r['seed'] == s]
            ax.plot([r['epoch'] for r in rows], [r['validation_macro_f1']*100 for r in rows], marker='o', label=str(s))
        ax.set_title(model); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('Epoch'); ax.legend(title='Seed')
    axes[0].set_ylabel('Validation Macro-F1 @ 0.30 (%)')
    save(fig, 'validation_macro_curves')


def write_summary(out, records, means, pairs, delta_stats, rare, grief, labels, counts, distribution, tradeoff, thresholds, ranks):
    parts = ['# Experiment 2.1 — Multi-Seed Stability Verification',
        '\n## 1. Experimental protocol\n',
        'Standard BERT 与 LGSA-GoEmotions sentiment-derived hierarchy v0；配对 seeds：42、123、2026。'
        'GoEmotions simplified 官方划分：43410 / 5426 / 5427。两者均独立从 raw bert-base-cased 初始化，固定同一模型 revision。'
        '每个模型保留各自原协议，训练 4 epochs；验证集 Macro-F1@0.30 选择 checkpoint，完全相同分数时保留更早 epoch。'
        '主比较固定阈值 0.30；辅助比较独立在各 seed 验证集上搜索 0.05–0.90（步长 0.05），并列选最小阈值。'
        '最终指标只计算 28 个 fine labels。coarse labels 是派生 supervision，不是原始标注。',
        '\nBERT：AdamW 5e-5、weight decay 0.01（排除 bias/LayerNorm）、warmup 1086。'
        'LGSA：Adam 1e-4、weight decay 0、warmup 1085、原 HGS/HGA/GAT/prompt/MLM/分类损失。'
        '均 batch16、FP32、text budget50。LGSA total length54。BERT 与 LGSA 沿用各自原依赖环境和确定性设置；'
        'LGSA CUDA scatter/GAT 可能存在非确定性，不声称逐位可复现。原始冻结仓库及原 seed42 产物均未被修改。',
        '\n' + table(['Model', 'Seed', '来源', 'Slurm job', 'Best epoch'],
                      [[r['model'].upper(), r['seed'], r['metadata']['seed42_decision'], r['metadata']['slurm_job_id'], r['metadata']['best_epoch']] for r in records]),
        '\n## 2. Mean ± standard deviation\n',
        '所有指标以下按百分比报告；标准差采用 sample SD（ddof=1），n=3。CSV 保存 0–1 原始精度。']
    for setting in SETTINGS:
        parts += ['\n**' + setting + '**\n', table(['Metric', 'BERT mean ± SD', 'LGSA mean ± SD'],
            [[k] + [f"{next(r for r in means if r['model']==m and r['threshold_setting']==setting and r['metric']==k)['mean']*100:.2f} ± "
                    f"{next(r for r in means if r['model']==m and r['threshold_setting']==setting and r['metric']==k)['std']*100:.2f}"
                    for m in ['BERT', 'LGSA']] for k in METRICS])]
    parts += ['\n## 3. Paired seed improvements\n', 'Δ = LGSA − BERT，单位：百分点（pp）。']
    for setting in SETTINGS:
        parts += ['\n**' + setting + '**\n', table(['Seed', *MAIN], [[s, *[f'{pairs[setting,s][k]*100:+.2f}' for k in MAIN]] for s in SEEDS]),
                  table(['Mean Δ', *MAIN], [['LGSA − BERT', *[f"{np.mean([pairs[setting,s][k] for s in SEEDS])*100:+.2f}" for k in MAIN]]])]
        for k in ['macro_f1', 'rare_f1', 'micro_f1']:
            wins = sum(pairs[setting, s][k] > 0 for s in SEEDS)
            parts.append(f'LGSA outperformed BERT on {k} in {wins} / 3 seeds ({setting}).')
    parts += ['\n## 4. Stability conclusion\n']
    for setting in SETTINGS:
        for metric in ['macro_f1', 'rare_f1']:
            deviations = {model: next(r['std'] for r in means if r['threshold_setting'] == setting and r['model'] == model and r['metric'] == metric)
                          for model in ['BERT', 'LGSA']}
            comparison = '大于' if deviations['LGSA'] > deviations['BERT'] else ('小于' if deviations['LGSA'] < deviations['BERT'] else '等于')
            parts.append(f"{setting} / {metric}：BERT 跨 seed SD={deviations['BERT']*100:.2f}，"
                         f"LGSA 跨 seed SD={deviations['LGSA']*100:.2f}；LGSA 的样本内波动{comparison}BERT。")
        for k in ['macro_f1', 'rare_f1']:
            row = next(r for r in delta_stats if r['threshold_setting'] == setting and r['metric'] == k)
            other = [pairs[setting, s][k] for s in [123, 2026]]
            seed42 = pairs[setting, 42][k]
            position = '高于另外两个 seed 的改善幅度' if seed42 > max(other) else ('低于另外两个 seed 的改善幅度' if seed42 < min(other) else '位于另外两个 seed 的改善幅度之间')
            direction = '三个受测 seed 的改善方向一致' if row['lgsa_wins'] == 3 else '未在全部三个受测 seed 上复现改善'
            parts.append(f"{setting} / {k}：{direction}。Δ 范围 {row['min']*100:+.2f} 至 {row['max']*100:+.2f} pp，"
                         f"Δ sample SD={row['std']*100:.2f} pp；seed42 {position}。这是样本内描述，不能据此认定统计异常或普遍优势。")
    parts += ['\n## 5. grief / relief analysis\n', '两类测试 support 极小（grief=6，relief=11）；必须同时查看 TP/FP/FN，不能只比较 F1。']
    for setting in SETTINGS:
        parts += ['\n**' + setting + '**\n', table(['Label', 'Model', 'Seed', 'Support', 'TP', 'FP', 'FN', 'Predicted +', 'P %', 'R %', 'F1 %'],
            [[r['label'], r['model'], r['seed'], r['support'], r['tp'], r['fp'], r['fn'], r['number_predicted_positive'],
              *[f"{float(r[k])*100:.2f}" for k in ['precision', 'recall', 'f1']]] for r in grief if r['threshold_setting'] == setting])]
        for name in ['grief', 'relief']:
            row = next(r for r in rare if r['label'] == name and r['threshold_setting'] == setting)
            selected = [r for r in grief if r['label'] == name and r['model'] == 'LGSA' and r['threshold_setting'] == setting]
            nz = sum(float(r['f1']) > 0 for r in selected)
            parts.append(f"{name}: LGSA 非零 F1 为 {nz}/3 seeds，优于 BERT 为 {row['lgsa_wins']}/3；"
                         f"LGSA F1 范围 {row['lgsa_min']*100:.2f}–{row['lgsa_max']*100:.2f}%，SD={row['lgsa_std']*100:.2f}。"
                         'TP/FP/FN 的跨 seed 变化见表；少数样本的变化就会明显改变 F1。')
    parts += ['\n## 6. Precision-recall behaviour\n', table(['Setting', 'Seed', 'Δ Micro-P pp', 'Δ Micro-R pp', 'Recall up / Precision down'],
        [[r['threshold_setting'], r['seed'], f"{r['delta_micro_precision']*100:+.2f}", f"{r['delta_micro_recall']*100:+.2f}", r['recall_up_precision_down']] for r in tradeoff])]
    for setting in SETTINGS:
        n = sum(r['recall_up_precision_down'] for r in tradeoff if r['threshold_setting'] == setting)
        parts.append(f'{setting}：提高 recall 且降低 precision 的模式出现在 {n}/3 seeds。' +
                     ('受测 seeds 均呈现该模式。' if n == 3 else '该行为未在全部受测 seeds 上复现；不能仅用 seed42 概括。'))
    parts += ['\n## 7. Label-level consistency\n', table(['Setting', 'Seed', 'Improved', 'Decreased', 'Tied'],
        [[r['threshold_setting'], r['seed'], r['improved'], r['decreased'], r['tied']] for r in counts])]
    for setting in SETTINGS:
        dist = next(r for r in distribution if r['threshold_setting'] == setting)
        rows = [r for r in labels if r['threshold_setting'] == setting]
        parts.append(f"{setting}：3/3、2/3、1/3、0/3 seeds 改善的标签数依次为 " +
                     ', '.join(str(dist[f'labels_improved_in_{n}_of_3']) for n in [3, 2, 1, 0]) + '。')
        parts.append('全部 seeds 改善的标签：' + (', '.join(r['label'] for r in rows if r['lgsa_wins'] == 3) or '无') + '。')
        parts.append('全部 seeds 下降的标签：' + (', '.join(r['label'] for r in rows if r['lgsa_losses'] == 3) or '无') + '。')
        two = sum(r['mean_delta'] for r in rows if r['label'] in ['grief', 'relief']) / 28
        rest = sum(r['mean_delta'] for r in rows if r['label'] not in ['grief', 'relief']) / 28
        parts.append(f'grief+relief 对整体平均 Macro-F1 Δ 的贡献为 {two*100:+.2f} pp；其余 26 类合计为 {rest*100:+.2f} pp。'
                     '这是等权 Macro-F1 的算术分解，不是机制归因。')
    parts += ['\n## 8. Scientific interpretation\n',
        '本实验只检验既定 BERT 和 LGSA 方案在三个配对随机种子上的稳定性。它不能单独隔离 HGS、HGA、loss 或 hierarchy 的因果贡献。'
        '没有运行显著性检验；n=3 的 mean/SD 与配对方向不能证明普遍优势。两种方法保留各自原 optimizer、loss 和训练协议，'
        '因此这里是完整既定方案的重复比较。', '\n**验证曲线与阈值**\n',
        table(['Model', 'Seed', 'Best epoch', 'Selected threshold', 'Validation Macro nondecreasing'],
              [[r['model'], r['seed'], r['best_epoch'], r['selected_threshold'], r['validation_macro_nondecreasing']] for r in thresholds])]
    for model in ['BERT', 'LGSA']:
        rr = [r for r in thresholds if r['model'] == model]
        ts = [r['selected_threshold'] for r in rr]
        parts.append(f"{model}: epoch4 最佳为 {sum(r['best_epoch']==4 for r in rr)}/3，validation Macro-F1 全程不下降为 "
                     f"{sum(r['validation_macro_nondecreasing'] for r in rr)}/3；所选阈值范围 {min(ts):.2f}–{max(ts):.2f}。"
                     '无论曲线是否仍在上升，均固定训练四个 epochs；没有平均阈值再复用。')
    parts.append('阈值选择改变配对模型排名的 seed/metric：' +
                 (', '.join(f"seed{r['seed']} {r['metric']}" for r in ranks if r['ranking_changed']) or '无') + '。')
    all_win = all(pairs['fixed_0.30', s][k] > 0 for s in SEEDS for k in ['macro_f1', 'rare_f1'])
    parts.append('固定阈值下，Macro-F1 与 Rare F1 均在全部受测 seeds 呈现正向改善；改善幅度及其种子敏感性应结合上述范围和 SD 解读。'
                 if all_win else '固定阈值下，Macro-F1 和 Rare F1 的改善未同时覆盖所有受测 seeds；seed42 的结果不能直接推广到其余 seeds。')
    parts += ['\n全部科研训练记录已核对 Slurm COMPLETED/0:0、checkpoint、产物哈希、28 类指标与配对样本。'
              'Smoke 指标未进入任何科学汇总。输出完成后停止；未运行 loss ablation 或 Hyperbolic 实验。']
    (out / 'MULTISEED_SUMMARY.md').write_text('\n\n'.join(parts) + '\n')
