"""Read saved predictions and loss traces only; no inference or training calls."""
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
from m4lib import *
from hierarchy_adapter import load_hierarchy
from preflight import canonical_records
from seed_extension import seed_audit

def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else None

def ranks(values):
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + 1 + j) / 2
        i = j
    return result

def correlation(x, y, rank=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.shape == y.shape and x.ndim == 1 and len(x) >= 2
        and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid correlation inputs')
    if rank:
        x, y = ranks(x), ranks(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def mean_std(values):
    a = np.asarray(values, dtype=float)
    require(a.shape == (3,) and np.isfinite(a).all(), 'Exactly three finite seed values needed')
    return float(a.mean()), float(a.std(ddof=1))

def harmonic(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0

def attribution(p0, r0, p1, r1):
    # Symmetric algebraic attribution, not a causal experiment or changed scores.
    p = ((harmonic(p1, r0) - harmonic(p0, r0)) +
         (harmonic(p1, r1) - harmonic(p0, r1))) / 2
    r = ((harmonic(p0, r1) - harmonic(p0, r0)) +
         (harmonic(p1, r1) - harmonic(p1, r0))) / 2
    return p, r

def paths(hierarchy):
    result = {}
    for i, name in enumerate(NAMES):
        chain, cur = [], i
        while cur != -1:
            chain.append(hierarchy['labels'][cur])
            cur = hierarchy['parents'][cur]
        chain = ['ROOT'] + list(reversed(chain))
        require(len(chain) - 1 == hierarchy['fine_depths'][i] + 1, 'Depth/path mismatch')
        result[name] = chain
    return result

def paired_labels(seed, y0, z0, ids0, y1, z1, ids1, support, hierarchy):
    require(y0.shape == y1.shape == z0.shape == z1.shape and y0.shape[1] == 28,
        'Only original28 output columns allowed')
    require(np.array_equal(ids0, ids1) and np.array_equal(y0, y1), 'Paired test IDs/GT/order differ')
    m0, a = metrics(y0, probabilities(z0), .30)
    m1, b = metrics(y1, probabilities(z1), .30)
    chains = paths(hierarchy)
    rows = []
    for i, name in enumerate(NAMES):
        r = {'seed': seed, 'label_id': i, 'label': name,
            'training_support': int(support[i]), 'training_frequency': float(support[i] / SIZES['train']),
            'test_support': b[i]['support'],
            'exp2_predicted_positive_count': a[i]['prediction_count'],
            'exp4b_predicted_positive_count': b[i]['prediction_count'],
            'prediction_count_ratio': ratio(b[i]['prediction_count'], a[i]['prediction_count']),
            'ratio_status': 'DEFINED' if a[i]['prediction_count'] else 'EXP2_ZERO_PREDICTED_POSITIVES',
            'prediction_count_change': b[i]['prediction_count'] - a[i]['prediction_count'],
            'exp2_predictions_to_test_support_ratio': ratio(a[i]['prediction_count'], b[i]['support']),
            'exp4b_predictions_to_test_support_ratio': ratio(b[i]['prediction_count'], b[i]['support']),
            'exp4b_count_below_exp2': b[i]['prediction_count'] < a[i]['prediction_count'],
            'exp4b_count_below_test_support': b[i]['prediction_count'] < b[i]['support'],
            'depth_in_exp4a_hierarchy': len(chains[name]) - 1,
            'root_to_leaf_path_length_edges': len(chains[name]) - 1,
            'root_to_leaf_path_node_count': len(chains[name]),
            'root_to_leaf_path': ' -> '.join(chains[name]),
            'rare_flag': i in RARE, 'group': 'rare' if i in RARE else 'non-rare'}
        for metric in ('precision', 'recall', 'f1'):
            r['exp2_' + metric], r['exp4b_' + metric] = a[i][metric], b[i][metric]
            r['delta_' + metric] = b[i][metric] - a[i][metric]
            r['delta_' + metric + '_pp'] = 100 * r['delta_' + metric]
        p, rec = attribution(a[i]['precision'], a[i]['recall'], b[i]['precision'], b[i]['recall'])
        require(abs(p + rec - r['delta_f1']) < 1e-12, 'F1 attribution identity failed')
        r.update(precision_f1_contribution_pp=100*p, recall_f1_contribution_pp=100*rec)
        rows.append(r)
    return m0, m1, rows

def loss_trace(path, cfg):
    path = Path(path)
    if not path.exists():
        return [], {'status': 'NOT_SAVED', 'reason': 'steps.jsonl absent; no recomputation or retraining'}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r['step'] for r in records] == list(range(1, 10857)), 'Saved loss steps are missing/duplicated/out of order')
    output = []
    available = all(all(k in r for k in ('classification', 'mlm', 'hga', 'alpha', 'loss')) for r in records)
    for r in records:
        if not available:
            output.append({'step': r['step'], 'epoch': r.get('epoch'), 'total_loss': r.get('loss'),
                'classification_loss': None, 'mlm_auxiliary_loss': None, 'hierarchy_hga_raw': None,
                'hierarchy_hga_weighted': None, 'auxiliary_combined_loss': None, 'alpha': r.get('alpha')})
            continue
        vals = [r[k] for k in ('classification', 'mlm', 'hga', 'alpha', 'loss')]
        require(np.isfinite(vals).all(), 'Nonfinite saved loss record')
        weighted = cfg['gof_loss_weight'] * r['alpha'] * r['hga']
        expected_alpha = min(1.0, (r['step'] - 1) / max(1, int(10856*cfg['hga_ramp_ratio'])))
        require(abs(r['alpha'] - expected_alpha) < 1e-12, 'Recorded ramp differs from frozen schedule')
        total = r['classification'] + r['mlm'] + weighted
        require(math.isclose(total, r['loss'], rel_tol=2e-5, abs_tol=2e-5), 'Saved total-loss decomposition mismatch')
        output.append({'step': r['step'], 'epoch': r['epoch'],
            'classification_loss': r['classification'], 'mlm_auxiliary_loss': r['mlm'],
            'hierarchy_hga_raw': r['hga'], 'hierarchy_hga_weighted': weighted,
            'auxiliary_combined_loss': r['mlm'] + weighted,
            'total_loss': r['loss'], 'alpha': r['alpha']})
    return output, {'status': 'SAVED_COMPONENTS_VERIFIED' if available else 'TOTAL_ONLY_COMPONENTS_NOT_SAVED',
        'steps': len(records), 'recomputed_training': False,
        'classification_scope': 'upstream multilabel categorical loss over56 hierarchy supervision nodes, not a separately saved fine28-only loss',
        'identity': 'total = classification + MLM + gof_loss_weight * alpha * HGA',
        'auxiliary_combined': 'MLM + weighted HGA; raw HGA is also reported separately'}

def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        y, z, ids = [saved[k].copy() for k in ('gold', 'logits', 'ids')]
    require(y.shape == z.shape == (5427, 28) and ids.shape == (5427,), 'Official test dimensions differ')
    require(len(set(ids.tolist())) == 5427 and np.isin(y, [0, 1]).all() and np.isfinite(z).all(), 'Bad test arrays')
    return y, z, ids

def group_rows(rows, scope):
    result = []
    for group in ('rare', 'non-rare'):
        subset = [r for r in rows if r['group'] == group]
        valid = [r['prediction_count_ratio'] for r in subset if r['prediction_count_ratio'] is not None]
        result.append({'scope': scope, 'group': group, 'labels': len(subset),
            'mean_prediction_count_change': float(np.mean([r['prediction_count_change'] for r in subset])),
            'sum_prediction_count_change': float(np.sum([r['prediction_count_change'] for r in subset])),
            'mean_prediction_count_ratio_defined_only': float(np.mean(valid)) if valid else None,
            'defined_ratio_labels': len(valid),
            'labels_with_fewer_predictions_than_exp2': sum(r['prediction_count_change'] < 0 for r in subset),
            'labels_with_fewer_predictions_than_test_support': sum(r['exp4b_predicted_positive_count'] < r['test_support'] for r in subset),
            'mean_delta_precision_pp': float(np.mean([r['delta_precision_pp'] for r in subset])),
            'mean_delta_recall_pp': float(np.mean([r['delta_recall_pp'] for r in subset])),
            'mean_delta_f1_pp': float(np.mean([r['delta_f1_pp'] for r in subset]))})
    return result

def correlation_rows(rows, scope):
    result = []
    for subset_name, subset in [('all28', rows), ('excluding_neutral27', [r for r in rows if r['label'] != 'neutral'])]:
        for xname, yname in [('training_support', 'delta_f1_pp'),
                             ('depth_in_exp4a_hierarchy', 'delta_f1_pp'),
                             ('training_support', 'delta_recall_pp'),
                             ('training_support', 'depth_in_exp4a_hierarchy')]:
            x, y = [r[xname] for r in subset], [r[yname] for r in subset]
            for method in ('pearson', 'spearman'):
                result.append({'scope': scope, 'subset': subset_name, 'n_labels': len(subset),
                    'x': xname, 'y': yname, 'method': method,
                    'coefficient': correlation(x, y, rank=method == 'spearman'),
                    'inference': 'descriptive association only; no independence or causality claim'})
            if xname == 'training_support':
                result.append({'scope': scope, 'subset': subset_name, 'n_labels': len(subset),
                    'x': 'log1p_training_support', 'y': yname, 'method': 'pearson',
                    'coefficient': correlation(np.log1p(x), y),
                    'inference': 'descriptive association only; no independence or causality claim'})
    return result

def run(work, out):
    validate_sources()
    require(not out.exists() and out.resolve().is_relative_to(Path(work).resolve()), 'New diagnostic directory only; never overwrite')
    candidates = candidate_paths(work)
    completed, metadata = {}, {}
    pilot_cfg = read(PILOT_RUN / 'config.json')
    for seed in SEEDS:
        completed[seed], metadata[seed] = verify_candidate(candidates[seed], seed)
        require(metadata[seed]['config'] == dict(pilot_cfg, seed=seed), 'Non-seed candidate config difference')
    root = REPO / 'experiments/exp2_lgsa/LGSA-GoEmotions'
    tree = REPO / 'outputs/exp4a_data_driven_hierarchy/hierarchy_primary_support70.json'
    audits = {seed: seed_audit(seed, REPO, root, tree, BASELINES / f'seed{seed}') for seed in SEEDS}
    hierarchy = load_hierarchy(tree)
    train_rows = canonical_records(REPO / '数据集/GoEmotions/processed/train.tsv')
    support = np.zeros(28, dtype=np.int64)
    for r in train_rows:
        support[r['labels']] += 1
    require(len(train_rows) == 43410 and set(sorted(range(28), key=lambda i: (support[i], i))[:7]) == set(RARE), 'Frozen training support/rare definition differ')
    inputs = {}
    for seed in SEEDS:
        for model, path in [('exp2', BASELINES / f'seed{seed}'), ('exp4b', candidates[seed])]:
            for name in ('SUCCESS.json', 'experiment_metadata.json', 'test_predictions.npz', 'steps.jsonl'):
                if (path / name).exists():
                    inputs[str(path / name)] = sha(path / name)
    inputs[str(tree)] = sha(tree)
    out.mkdir(parents=True, exist_ok=False)
    try:
        scores, labels, groups, correlations, all_losses, losses_info = [], [], [], [], [], []
        for seed in SEEDS:
            y0, z0, ids0 = load_npz(BASELINES / f'seed{seed}' / 'test_predictions.npz')
            y1, z1, ids1 = load_npz(candidates[seed] / 'test_predictions.npz')
            m0, m1, rows = paired_labels(seed, y0, z0, ids0, y1, z1, ids1, support, hierarchy)
            # Independent baseline recomputation already checked by the frozen provenance verifier.
            for k in FIELDS:
                require(abs(m0[k] - audits[seed][4]['recomputed_fixed_0_30_metrics'][k]) < 1e-12, 'Baseline recomputation changed')
            for model, values in [('Exp2', m0), ('Exp4B', m1), ('Paired_delta', {k: m1[k]-m0[k] for k in FIELDS})]:
                scores.append(dict(seed=seed, model=model, **{k: values[k] for k in FIELDS}))
            labels += rows
            groups += group_rows(rows, f'seed{seed}')
            correlations += correlation_rows(rows, f'seed{seed}')
            trace, info = loss_trace(candidates[seed] / 'steps.jsonl', metadata[seed]['config'])
            all_losses += [dict(seed=seed, **r) for r in trace]
            losses_info.append(dict(seed=seed, source=str(candidates[seed] / 'steps.jsonl'), **info))
        means = []
        count_columns = ('exp2_predicted_positive_count', 'exp4b_predicted_positive_count', 'prediction_count_change')
        measured = (*count_columns, 'exp2_precision', 'exp2_recall', 'exp2_f1', 'exp4b_precision', 'exp4b_recall', 'exp4b_f1',
            'delta_precision', 'delta_recall', 'delta_f1', 'delta_precision_pp', 'delta_recall_pp', 'delta_f1_pp',
            'precision_f1_contribution_pp', 'recall_f1_contribution_pp')
        for i, name in enumerate(NAMES):
            subset = [r for r in labels if r['label'] == name]
            mean = {k: subset[0][k] for k in ('label_id', 'label', 'training_support', 'training_frequency', 'test_support',
                'depth_in_exp4a_hierarchy', 'root_to_leaf_path_length_edges', 'root_to_leaf_path_node_count',
                'root_to_leaf_path', 'rare_flag', 'group')}
            mean['scope'] = 'three_seed_label_mean'
            for k in measured:
                mean[k], mean[k + '_std'] = mean_std([r[k] for r in subset])
            mean['prediction_count_ratio'] = ratio(mean['exp4b_predicted_positive_count'], mean['exp2_predicted_positive_count'])
            mean['ratio_status'] = 'DEFINED' if mean['prediction_count_ratio'] is not None else 'EXP2_ZERO_MEAN_PREDICTED_POSITIVES'
            mean['fewer_predictions_than_exp2_in_all_three_seeds'] = all(r['prediction_count_change'] < 0 for r in subset)
            means.append(mean)
        groups += group_rows(means, 'three_seed_label_mean')
        correlations += correlation_rows(means, 'three_seed_label_mean')
        summary = []
        for model in ('Exp2', 'Exp4B', 'Paired_delta'):
            subset = [r for r in scores if r['model'] == model]
            for k in FIELDS:
                mean, std = mean_std([r[k] for r in subset])
                summary.append({'model': model, 'metric': k, 'n_seeds': 3, 'mean': mean, 'std': std,
                    'mean_percent_or_pp': 100*mean, 'std_percent_or_pp': 100*std, 'ddof': 1})
        csvwrite(out / 'metrics_by_seed.csv', scores)
        csvwrite(out / 'metrics_mean_std.csv', summary)
        csvwrite(out / 'label_diagnostics_by_seed.csv', labels)
        csvwrite(out / 'label_diagnostics_mean_std.csv', means)
        csvwrite(out / 'focus_labels_by_seed.csv', [r for r in labels if r['label'] in FOCUS])
        csvwrite(out / 'focus_labels_mean_std.csv', [r for r in means if r['label'] in FOCUS])
        csvwrite(out / 'frequency_depth_correlations.csv', correlations)
        csvwrite(out / 'rare_nonrare_prediction_changes.csv', groups)
        if all_losses:
            csvwrite(out / 'loss_trace_by_step.csv', all_losses)
            epoch_losses = []
            for seed in SEEDS:
                for epoch in range(1, 5):
                    subset = [r for r in all_losses if r['seed'] == seed and r['epoch'] == epoch]
                    if subset:
                        row = {'seed': seed, 'epoch': epoch, 'logged_steps': len(subset)}
                        for k in ('classification_loss', 'mlm_auxiliary_loss', 'hierarchy_hga_raw', 'hierarchy_hga_weighted', 'auxiliary_combined_loss', 'total_loss'):
                            values = [r[k] for r in subset if r[k] is not None]
                            row[k + '_mean_per_update'] = float(np.mean(values)) if values else None
                        epoch_losses.append(row)
            if epoch_losses:
                csvwrite(out / 'loss_summary_by_epoch.csv', epoch_losses)
        write(out / 'loss_availability.json', losses_info)
        write(out / 'input_provenance.json', {'files_sha256': inputs,
            'seed42_retrained': False, 'training_changed_by_diagnostics': False,
            'extension_hashes': extension_hashes(), 'rare_ids': RARE, 'protocol': read(HERE / 'protocol.json')})
        report = render_summary(scores, summary, means, groups, correlations, losses_info)
        with (out / 'EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY.md').open('x', encoding='utf-8') as f:
            f.write(report)
        require(all(sha(Path(p)) == s for p, s in inputs.items()), 'Existing inputs changed during read-only diagnostics')
        for seed in SEEDS:
            require(seed_audit(seed, REPO, root, tree, BASELINES / f'seed{seed}')[-1] == audits[seed][-1],
                'Frozen source/data/baseline changed during diagnostics')
        validate_sources()
        record = {'status': 'EXPERIMENT_4B_MULTI_SEED_DIAGNOSTIC_COMPLETE', 'seeds': list(SEEDS),
            'all_seeds_verified': True, 'diagnostics_training_or_inference_performed': False,
            'hierarchy_changed': False, 'threshold': .30, 'std_ddof': 1,
            'completed_at': now(), 'artifacts_sha256': artifact_hashes(out)}
        write(out / 'SUCCESS.json', record)
        verify_run(out, record['status'])
        print(record['status'], out, flush=True)
    except BaseException as error:
        write(out / 'FAILURE.json', {'status': 'DIAGNOSTICS_FAILED', 'error': repr(error), 'failed_at': now()})
        raise

def render_summary(scores, summary, labels, groups, corr, losses):
    def format_value(value):
        return 'n.a. (zero variance)' if value is None else f'{value:+.4f}'
    def get_summary(model, metric):
        return next(r for r in summary if r['model'] == model and r['metric'] == metric)
    lines = ['# EXP4B_MULTI_SEED_DIAGNOSTIC_SUMMARY', '',
        '完整 Exp4A hierarchy 保持不变。seed42 复用已完成结果，新增训练仅 seed123/2026。',
        '正式评估固定 sigmoid≥0.30，28 labels，text50/total68，原始4 epochs、optimizer/loss及validation Macro-F1 checkpoint选择不变。',
        '原循环已有的 validation-selected threshold 辅助输出保留，但不用于本报告，不调threshold或树。', '',
        '## 三 seed 指标', '', '| Model | Seed | Micro-F1 | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall | Rare-F1 | Non-rare-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in scores:
        lines.append(f"| {r['model']} | {r['seed']} | " + ' | '.join(f'{100*r[k]:.3f}' for k in FIELDS) + ' |')
    lines += ['', '数值单位：模型指标为 %，paired delta 为 pp。std 为样本标准差（ddof=1，n=3）。', '',
        '| Metric | Exp2 mean ± std (%) | Exp4B mean ± std (%) | Paired delta mean ± std (pp) |', '|---|---:|---:|---:|']
    for k in FIELDS:
        values = [get_summary(model, k) for model in ('Exp2', 'Exp4B', 'Paired_delta')]
        lines.append('| ' + k + ' | ' + ' | '.join(f"{r['mean_percent_or_pp']:.3f} ± {r['std_percent_or_pp']:.3f}" for r in values) + ' |')
    delta = [r for r in scores if r['model'] == 'Paired_delta']
    lines += ['', '## 必须回答的五个问题', '', '### 1. Macro-F1 / Rare-F1 下降是否在三 seed 中一致？', '']
    for metric in ('macro_f1', 'rare_mean_f1'):
        falling = [r['seed'] for r in delta if r[metric] < 0]
        lines.append(f"{metric}：{len(falling)}/3 seeds 下降；下降 seeds={falling}。" + ('三 seed 方向一致。' if len(falling) == 3 else '不是三 seed 一致下降。'))
    lines += ['', '### 2. 下降主要来自 precision 还是 recall？', '']
    mp, mr = get_summary('Paired_delta', 'macro_precision')['mean_percent_or_pp'], get_summary('Paired_delta', 'macro_recall')['mean_percent_or_pp']
    pcon = float(np.mean([r['precision_f1_contribution_pp'] for r in labels]))
    rcon = float(np.mean([r['recall_f1_contribution_pp'] for r in labels]))
    component = 'recall' if rcon < pcon else 'precision' if pcon < rcon else '两者相同'
    lines += [f'Macro Precision 平均变化 {mp:+.3f} pp，Macro Recall {mr:+.3f} pp。',
        f'对逐标签 F1 的对称代数分解：precision贡献 {pcon:+.3f} pp，recall贡献 {rcon:+.3f} pp；更负的分量是 {component}。两分量之和等于平均 Macro-F1 delta。',
        '这是基于已保存P/R的代数归因，不是新训练、因果证明或修改预测。', '',
        '### 3. Rare labels 是否有系统性 under-prediction？', '']
    rare = [r for r in groups if r['group'] == 'rare' and r['scope'] != 'three_seed_label_mean']
    consistent = [r['label'] for r in labels if r['rare_flag'] and r['fewer_predictions_than_exp2_in_all_three_seeds']]
    if all(r['mean_prediction_count_change'] < 0 and r['mean_delta_recall_pp'] < 0 for r in rare):
        lines.append('三 seed 的rare组均出现平均预测正例数减少且平均recall下降，支持相对Exp2的群体性under-prediction现象；不代表每个rare label都如此。')
    else:
        lines.append('三 seed 未同时满足rare组平均预测数减少且recall下降，不能称为三 seed一致的群体性under-prediction。')
    lines.append(f'三 seed 均比同seed Exp2少预测的rare labels：{consistent}。')
    lines += ['', '| Scope | Group | Labels | Mean predicted-count change | Lower-count labels vs Exp2 | Delta P (pp) | Delta R (pp) | Delta F1 (pp) |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        lines.append(f"| {r['scope']} | {r['group']} | {r['labels']} | {r['mean_prediction_count_change']:+.3f} | {r['labels_with_fewer_predictions_than_exp2']} | {r['mean_delta_precision_pp']:+.3f} | {r['mean_delta_recall_pp']:+.3f} | {r['mean_delta_f1_pp']:+.3f} |")
    lines += ['', 'prediction count ratio = Exp4B/Exp2。Exp2计数为0时ratio为空/null，不替代为0或无穷。',
        '相对Exp2减少与低于test gold support是两个不同定义，两者均保存在诊断表。Rare集合沿用完整TRAIN频率最低7个labels的冻结定义。', '',
        '### 4. Label frequency 是否与 performance drop 有关系？', '',
        '下表报告training support与ΔF1、Δrecall的Pearson/Spearman描述相关，以及log1p support的Pearson。',
        '正相关表示高频label倾向于较大的delta（低频倾向于更负）；负相关方向相反。不能仅凭相关系数宣称显著性或因果。', '',
        '### 5. Hierarchy depth 是否与 performance drop 有关系？', '',
        '下表报告ROOT起算的edge depth与ΔF1的描述相关。负相关表示越深delta倾向更负；正相关相反。depth不等于独立实验变量。',
        '同时输出frequency与depth相关以提示混杂。每seed n=28；三seed均值按label聚合仍是28个观察，不把84行当独立样本。', '',
        '| Scope | Subset | X | Y | Method | Coefficient |', '|---|---|---|---|---|---:|']
    for r in corr:
        if r['scope'] == 'three_seed_label_mean' or (r['subset'] == 'all28' and r['method'] == 'spearman'):
            lines.append(f"| {r['scope']} | {r['subset']} | {r['x']} | {r['y']} | {r['method']} | {format_value(r['coefficient'])} |")
    # Explicit direction answers from the primary28-label three-seed mean, not a cherry-picked seed.
    for x, y in [('training_support', 'delta_f1_pp'), ('training_support', 'delta_recall_pp'), ('depth_in_exp4a_hierarchy', 'delta_f1_pp')]:
        value = next(r['coefficient'] for r in corr if r['scope'] == 'three_seed_label_mean'
            and r['subset'] == 'all28' and r['method'] == 'spearman' and r['x'] == x and r['y'] == y)
        description = '无法计算（变量零方差）' if value is None else '正向描述关联' if value > 0 else '负向描述关联' if value < 0 else '本样本秩相关为0'
        lines.append(f'{x} vs {y}：Spearman={format_value(value)}，{description}；不作显著性或因果结论。')
    lines += ['', '## 重点labels：三seed均值', '',
        '| Label | Train support | Test support | Depth | Exp2 predicted count | Exp4B predicted count | Count ratio | Delta P (pp) | Delta R (pp) | Delta F1 (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in FOCUS:
        r = next(r for r in labels if r['label'] == name)
        rt = 'n.a.' if r['prediction_count_ratio'] is None else f"{r['prediction_count_ratio']:.3f}"
        lines.append(f"| {name} | {r['training_support']} | {r['test_support']} | {r['depth_in_exp4a_hierarchy']} | {r['exp2_predicted_positive_count']:.2f} | {r['exp4b_predicted_positive_count']:.2f} | {rt} | {r['delta_precision_pp']:+.3f} | {r['delta_recall_pp']:+.3f} | {r['delta_f1_pp']:+.3f} |")
    lines += ['', 'Depth定义：ROOT=0；root-to-leaf path length按edge计数，节点数量=edge+1。完整path见逐标签表。', '',
        '## 已保存训练loss', '',
        '读取原steps.jsonl，不改任何训练逻辑。每步与每epoch汇总包含classification、MLM auxiliary、HGA raw、weighted HGA和total。',
        'total = classification + MLM + gof_loss_weight × alpha × HGA；auxiliary_combined = MLM + weighted HGA。',
        'classification是原循环对56个hierarchy targets的分类损失，不能声称已单独记录28 fine labels损失。',
        'epoch汇总是optimizer update的算术平均，不是按样本数量加权。缺失的分量写n.a./null，不从total反推。']
    lines += [f"seed{r['seed']}：{r['status']}；已保存steps={r.get('steps', 0)}。" for r in losses]
    lines += ['', '## 科学边界与停止条件', '',
        '相同full hierarchy保持56辅助节点、9深度、68总长度；只seed变化，未剪枝/压平/重连，未加入peer loss或hyperbolic。',
        'Exp2来自历史HPC，Exp4B来自Ubuntu4090D。三个配对seed比单seed更有信息，但n=3不是统计显著性的充分证明。',
        '相关分析受frequency/depth混杂、共享训练模型、标签依赖和低test support影响。没有根据test指标选择新的tree或threshold。',
        '所有数据、旧结果和训练代码在诊断前后核对hash，诊断不做forward/训练。',
        '三seed与只读诊断完成后停止，不自动启动其他实验。', '',
        '最终完成状态记录于SUCCESS.json，并在产物hash验证后输出完成标记。', '']
    return '\n'.join(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.work.resolve(), args.out.resolve())
