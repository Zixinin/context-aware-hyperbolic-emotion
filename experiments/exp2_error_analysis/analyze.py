"""Read-only paired analysis of certified Experiment 2.1 test predictions.

No training, inference, threshold selection or changes to original artifacts.
"""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
NAMES = 'admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral'.split()
SEEDS = (42, 123, 2026)
RARE = (16, 21, 23, 19, 12, 24, 14)
MODEL_DIR = {'bert': 'bert_baseline', 'lgsa': 'lgsa_goemotions'}
THRESHOLD = .30


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def objsha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def rate(n, d):
    return float(n / d) if d else None


def sigmoid(z, model):
    # Preserve the original model-specific arithmetic, including threshold ties.
    if model == 'bert':
        return np.exp(-np.logaddexp(0., -np.asarray(z, dtype=np.float64)))
    return 1. / (1. + np.exp(-np.clip(z, -80, 80)))


def hierarchy(path):
    h = read(path)
    labels = sorted(h['labels'], key=lambda r: r['fine_id'])
    require([r['fine_id'] for r in labels] == list(range(28)), 'Hierarchy must cover 28 unique fine IDs')
    require([r['fine_label'] for r in labels] == NAMES, 'Hierarchy label order differs')
    parents = np.array([r['coarse_label'] for r in labels])
    require(set(parents) == {'Positive', 'Negative', 'Ambiguous', 'Neutral'}, 'Unexpected hierarchy')
    return parents, Counter(parents)


def official_test(path):
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as f:
        for line in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL):
            require(len(line) == 3, 'Expected official text,labels,id TSV')
            text, annotations, sid = line
            labels = list(map(int, annotations.split(',')))
            require(labels and len(set(labels)) == len(labels) and all(0 <= i < 28 for i in labels), 'Invalid TSV labels')
            rows.append({'id': sid, 'text': text, 'labels': labels})
    require(len(rows) == 5427 and len({r['id'] for r in rows}) == 5427, 'Official test must have 5427 unique IDs')
    return rows


def signatures(rows):
    y = np.zeros((len(rows), 28), dtype=np.uint8)
    h = hashlib.sha256()
    for i, r in enumerate(rows):
        y[i, r['labels']] = 1
        h.update((json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode())
    return {'rows_sha256': h.hexdigest(), 'ids_sha256': objsha([r['id'] for r in rows]),
            'gold_sha256': hashlib.sha256(y.tobytes()).hexdigest(), 'size': len(rows)}, y


def classification(y, p):
    pred = p >= THRESHOLD
    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall, f1 = div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    m = {'micro_f1': float(div(2 * tp.sum(), (2 * tp + fp + fn).sum())),
         'macro_f1': float(f1.mean()), 'rare_f1': float(f1[list(RARE)].mean())}
    return m, [dict(support=int(y[:, i].sum()), tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
                    precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i])) for i in range(28)]


def load_runs(inputs, texts, parents, hierarchy_path):
    report = inputs / 'multiseed_comparison'
    certificate = read(report / 'SUCCESS.json')
    require(certificate['status'] == 'MULTISEED_COMPARISON_PASS' and str(certificate['training_job']) == '3006610', 'Require certified job3006610 outputs')
    integrity_path = report / 'INTEGRITY_PASS.json'
    require(certificate['artifacts_sha256'].get('INTEGRITY_PASS.json') == sha(integrity_path), 'Report integrity certificate hash mismatch')
    integrity = read(integrity_path)
    require(integrity['status'] == 'SIX_RUNS_INTEGRITY_PASS', 'Six-run integrity missing')
    for job, record in integrity['accounting'].items():
        lookup = {r[0]: r for r in record['rows']}
        require(str(job) in lookup and str(job) + '.batch' in lookup, 'Accounting main/batch evidence missing')
        require(all(r[1] == 'COMPLETED' and r[2] == '0:0' for r in lookup.values()), 'Accounting not COMPLETED/0:0')
    require('3006610' in integrity['accounting'], 'job3006610 accounting missing')
    text_lookup = {r['id']: r for r in texts}
    runs, proof, reference = {}, [], None
    for model, folder in MODEL_DIR.items():
        for seed in SEEDS:
            path = inputs / folder / f'seed{seed}'
            marker = read(path / 'SUCCESS.json')
            require(marker['status'] == 'NORMALIZED_RUN_PASS' and marker['model'] == model and marker['seed'] == seed, 'Invalid normalized identity')
            required = ['test_predictions.npz', 'experiment_metadata.json', 'checkpoint_info.json',
                        'test_fixed_0.30_metrics.json', 'per_label_fixed_0.30_metrics.csv']
            for name in required:
                require(marker['artifacts_sha256'].get(name) == sha(path / name), 'Input hash mismatch: ' + str(path / name))
            m = read(path / 'experiment_metadata.json')
            require(m['model'] == model and m['seed'] == seed and m['phase'] == 'full', 'Model/seed/phase mismatch')
            require(m['status'] == m['slurm_state'] == 'COMPLETED' and m['slurm_exit_code'] == '0:0', 'Unfinished input')
            require(m['label_names'] == NAMES and m['num_labels'] == 28 and m['fixed_threshold'] == THRESHOLD and m['rare_ids'] == list(RARE), 'Evaluation protocol mismatch')
            require(m['dataset_split_sizes'] == {'train': 43410, 'validation': 5426, 'test': 5427}, 'Dataset split sizes differ')
            require(m['checkpoint_reload_verified'] is True and m['checkpoint_reload_evidence'], 'Best checkpoint reload proof missing')
            require(m['orchestration_release_sha256'] == integrity['release_sha256'], 'Source release lineage mismatch')
            require(str(m['slurm_job_id']) in integrity['accounting'], 'Run accounting missing')
            if model == 'lgsa':
                require(m['hierarchy_sha256'] == sha(hierarchy_path), 'Hierarchy differs from training')
            checkpoint = read(path / 'checkpoint_info.json')
            require(checkpoint['selection'] == 'validation_macro_f1_at_fixed_0.30' and checkpoint['reload_evidence'], 'Wrong checkpoint selection')
            with np.load(path / 'test_predictions.npz', allow_pickle=False) as f:
                ids, y, z, p = f['ids'].astype(str), f['gold'], f['logits'], f['probabilities']
            require(y.shape == z.shape == p.shape == (5427, 28), 'Require 5427 x 28 predictions')
            require(len(set(ids)) == 5427 and set(ids) == set(text_lookup), 'Sample IDs differ from official test')
            require(np.isfinite(z).all() and np.isfinite(p).all() and np.isin(y, [0, 1]).all(), 'Invalid predictions')
            recomputed = sigmoid(z, model)
            score_difference = np.abs(p.astype(np.float64) - recomputed.astype(np.float64))
            score_spacing = np.abs(np.spacing(p)).astype(np.float64)
            require(np.all(score_difference <= 4.0 * score_spacing), 'Sigmoid recomputation exceeds 4 ULP')
            require(np.array_equal(p >= THRESHOLD, recomputed >= THRESHOLD), 'Sigmoid recomputation changes threshold decisions')
            score_ulps = np.divide(score_difference, score_spacing, out=np.zeros_like(score_difference), where=score_spacing > 0)
            score_check = dict(max_absolute_difference=float(score_difference.max()), max_ulp_difference=float(score_ulps.max()),
                               limit_ulps=4, threshold_decisions_identical=True, score_source='hash_verified_saved_probabilities')
            aligned = [text_lookup[sid] for sid in ids]
            signature, expected_y = signatures(aligned)
            require(np.array_equal(y, expected_y), 'GT differs from official annotations')
            require(signature == m['dataset_signatures']['test'], 'Official test text/ID/GT signature differs from frozen dataset')
            if reference is None:
                reference = (ids, y)
            require(np.array_equal(ids, reference[0]) and np.array_equal(y, reference[1]), 'Six runs must be paired in identical test order')
            y = y.astype(bool)
            metrics, perlabel = classification(y, p)
            stored = read(path / 'test_fixed_0.30_metrics.json')
            require(all(abs(metrics[k] - stored[k]) < 1e-10 for k in metrics), 'Recomputed test F1 differs')
            with (path / 'per_label_fixed_0.30_metrics.csv').open(newline='') as f:
                original_labels = list(csv.DictReader(f))
            require(len(original_labels) == 28, 'Per-label source incomplete')
            for i, original in enumerate(original_labels):
                require(original['label'] == NAMES[i], 'Per-label names differ')
                require(all(abs(float(original[k]) - v) < 1e-10 for k, v in perlabel[i].items()), 'Per-label counts/F1 differ')
            runs[model, seed] = dict(ids=ids, y=y, z=z, p=p, texts=[r['text'] for r in aligned], metrics=metrics, perlabel=perlabel)
            proof.append(dict(model=model, seed=seed, input_directory=str(path.resolve()),
                              test_predictions_sha256=sha(path / 'test_predictions.npz'),
                              metadata_sha256=sha(path / 'experiment_metadata.json'), success_sha256=sha(path / 'SUCCESS.json'),
                              checkpoint=checkpoint, dataset_signature=signature, recomputed_metrics=metrics, score_recomputation_check=score_check, test_text_parser='csv.QUOTE_MINIMAL',
                              checkpoint_hash_scope='Historical HPC certificate; checkpoint binary is not required or rehashed locally'))
    return runs, proof, integrity


def instances(run, model, seed, parents, sizes):
    y, z, p = run['y'], run['z'], run['p']
    require(y.shape == p.shape == z.shape and y.shape[1] == 28, 'Invalid analysis arrays')
    require(np.all(y.sum(1) > 0) and np.all(y.sum(1) < 28), 'Each sample requires GT and at least one negative')
    hard, fps = [], []
    for k, sid in enumerate(run['ids']):
        gt = np.flatnonzero(y[k])
        negative = np.flatnonzero(~y[k])
        # Rank raw logits (sigmoid is monotone); avoid artificial float saturation ties.
        best_z = z[k, negative].max()
        tied = negative[z[k, negative] == best_z]
        w = int(tied[0])  # exact ties choose the lowest official fine-label ID
        eligible_gt = any(sizes[parents[t]] >= 2 for t in gt)
        gt_labels = json.dumps([NAMES[t] for t in gt])
        for t in gt:
            siblings = negative[parents[negative] == parents[t]]
            sw = int(siblings[np.argmax(z[k, siblings])]) if len(siblings) else None
            hard.append(dict(model=model.upper(), seed=seed, sample_id=str(sid),
                             true_label=NAMES[t], true_parent=parents[t], top_wrong=NAMES[w], wrong_parent=parents[w],
                             same_parent=bool(parents[t] == parents[w]), sibling_eligible=bool(sizes[parents[t]] >= 2),
                             true_score=float(p[k, t]), wrong_score=float(p[k, w]), margin=float(p[k, t] - p[k, w]),
                             true_logit=float(z[k, t]), wrong_logit=float(z[k, w]), logit_margin=float(z[k, t] - z[k, w]),
                             wrong_is_false_positive=bool(p[k, w] >= THRESHOLD), true_is_false_negative=bool(p[k, t] < THRESHOLD),
                             negative_outranks_true=bool(z[k, w] > z[k, t]), top_wrong_tie_count=len(tied),
                             available_sibling_negatives=len(siblings), available_negatives=len(negative),
                             uniform_negative_same_parent_probability=len(siblings) / len(negative),
                             top_sibling_wrong=NAMES[sw] if sw is not None else '',
                             top_sibling_wrong_score=float(p[k, sw]) if sw is not None else None,
                             ground_truth_labels=gt_labels, text=run['texts'][k]))
        for w in np.flatnonzero((p[k] >= THRESHOLD) & ~y[k]):
            matching = [int(t) for t in gt if parents[t] == parents[w]]
            fps.append(dict(model=model.upper(), seed=seed, sample_id=str(sid), false_positive_label=NAMES[w],
                            false_positive_parent=parents[w], same_parent_any_gt=bool(matching),
                            sample_has_sibling_eligible_gt=bool(eligible_gt), score=float(p[k, w]),
                            matching_gt_labels=json.dumps([NAMES[t] for t in matching]), ground_truth_labels=gt_labels,
                            text=run['texts'][k]))
    return hard, fps


def aggregate_pairs(rows, model, seed):
    groups, support = defaultdict(list), Counter(r['true_label'] for r in rows)
    for r in rows:
        groups[r['true_label'], r['top_wrong']].append(r)
    result = []
    for (t, w), group in groups.items():
        r = group[0]
        result.append(dict(model=model, seed=seed, true_label=t, true_parent=r['true_parent'], top_wrong=w,
                           wrong_parent=r['wrong_parent'], same_parent=r['same_parent'], count=len(group),
                           true_label_instance_support=support[t], pair_rate_given_true_label=len(group) / support[t],
                           avg_true_score=mean(x['true_score'] for x in group), avg_wrong_score=mean(x['wrong_score'] for x in group),
                           avg_margin=mean(x['margin'] for x in group), avg_logit_margin=mean(x['logit_margin'] for x in group),
                           wrong_false_positive_count=sum(x['wrong_is_false_positive'] for x in group),
                           negative_outranks_true_count=sum(x['negative_outranks_true'] for x in group)))
    result.sort(key=lambda r: (-r['count'], NAMES.index(r['true_label']), NAMES.index(r['top_wrong'])))
    for rank, r in enumerate(result, 1):
        r['rank_by_count'] = rank
    confidence_order = sorted(result, key=lambda r: (-r['avg_wrong_score'], -r['count'], NAMES.index(r['true_label']), NAMES.index(r['top_wrong'])))
    for rank, r in enumerate(confidence_order, 1):
        r['rank_by_avg_wrong_score'] = rank
    return result


def summaries(hard, fps, model, seed):
    output = []
    for scope in ['all_labels', 'sibling_eligible_labels']:
        rows = [r for r in hard if scope == 'all_labels' or r['sibling_eligible']]
        same = sum(r['same_parent'] for r in rows)
        chance = mean(r['uniform_negative_same_parent_probability'] for r in rows)
        observed = rate(same, len(rows))
        output.append(dict(model=model, seed=seed, analysis='hard_negative_ranking', scope=scope,
                           counting_unit='sample x GT label x seed', total=len(rows), same_parent_count=same,
                           cross_parent_count=len(rows) - same, same_parent_rate=observed, cross_parent_rate=rate(len(rows) - same, len(rows)),
                           uniform_negative_expected_rate=chance,
                           observed_minus_expected=observed - chance if observed is not None else None,
                           observed_over_expected=observed / chance if chance else None,
                           mean_wrong_score=mean(r['wrong_score'] for r in rows), mean_margin=mean(r['margin'] for r in rows)))
        fr = [r for r in fps if scope == 'all_labels' or r['sample_has_sibling_eligible_gt']]
        same = sum(r['same_parent_any_gt'] for r in fr)
        output.append(dict(model=model, seed=seed, analysis='false_positive_at_0.30', scope=scope,
                           counting_unit='sample x FP label x seed; count once even if multiple matching GT labels',
                           total=len(fr), same_parent_count=same, cross_parent_count=len(fr) - same,
                           same_parent_rate=rate(same, len(fr)), cross_parent_rate=rate(len(fr) - same, len(fr)),
                           uniform_negative_expected_rate=None, observed_minus_expected=None, observed_over_expected=None,
                           mean_wrong_score=mean(r['score'] for r in fr), mean_margin=None))
    return output


def stability(pair_rows, top_k):
    out = []
    for model in ['BERT', 'LGSA']:
        lookup = {(r['seed'], r['true_label'], r['top_wrong']): r for r in pair_rows if r['model'] == model and r['seed'] != 'all_seeds'}
        pairs = sorted({(t, w) for _, t, w in lookup}, key=lambda p: (NAMES.index(p[0]), NAMES.index(p[1])))
        for t, w in pairs:
            values = [lookup.get((s, t, w)) for s in SEEDS]
            counts = [v['count'] if v else 0 for v in values]
            r = dict(model=model, true_label=t, top_wrong=w, same_parent=next(v['same_parent'] for v in values if v),
                     seeds_present=sum(c > 0 for c in counts), seeds_in_top_k=sum(v is not None and v['rank_by_count'] <= top_k for v in values),
                     top_k=top_k, stable_presence_3_of_3=all(c > 0 for c in counts),
                     stable_top_k_3_of_3=all(v is not None and v['rank_by_count'] <= top_k for v in values),
                     count_sum=sum(counts), count_mean=float(np.mean(counts)), count_sd=float(np.std(counts, ddof=1)))
            for s, v, c in zip(SEEDS, values, counts):
                r.update({f'seed{s}_count': c, f'seed{s}_rank': v['rank_by_count'] if v else None,
                          f'seed{s}_avg_wrong_score': v['avg_wrong_score'] if v else None, f'seed{s}_avg_margin': v['avg_margin'] if v else None})
            out.append(r)
    return sorted(out, key=lambda r: (r['model'], -r['seeds_in_top_k'], -r['count_sum'], NAMES.index(r['true_label']), NAMES.index(r['top_wrong'])))


def label_summary(run, rows, model, seed, parents):
    result = []
    for i, label in enumerate(NAMES):
        group = [r for r in rows if r['true_label'] == label]
        best = Counter(r['top_wrong'] for r in group)
        top = sorted(best, key=lambda n: (-best[n], NAMES.index(n)))
        result.append(dict(model=model.upper(), seed=seed, label=label, parent=parents[i], is_rare=i in RARE,
                           **run['perlabel'][i], hard_negative_count=len(group),
                           same_parent_count=sum(r['same_parent'] for r in group), same_parent_rate=mean(r['same_parent'] for r in group),
                           avg_true_score=mean(r['true_score'] for r in group), avg_wrong_score=mean(r['wrong_score'] for r in group),
                           avg_margin=mean(r['margin'] for r in group), avg_logit_margin=mean(r['logit_margin'] for r in group),
                           negative_outranks_true_count=sum(r['negative_outranks_true'] for r in group),
                           top_wrong=top[0] if top else '', top_wrong_count=best[top[0]] if top else 0,
                           avg_top_sibling_wrong_score=mean(r['top_sibling_wrong_score'] for r in group if r['top_sibling_wrong_score'] is not None)))
    return result


def paired_comparison(runs, hard, parents):
    lookup = {(r['model'], r['seed'], r['sample_id'], r['true_label']): r for r in hard}
    rows, fixed = [], []
    for seed in SEEDS:
        b, l = runs['bert', seed], runs['lgsa', seed]
        for i, label in enumerate(NAMES):
            idx = np.flatnonzero(b['y'][:, i])
            bs = [lookup['BERT', seed, b['ids'][k], label] for k in idx]
            ls = [lookup['LGSA', seed, l['ids'][k], label] for k in idx]
            r = dict(seed=seed, label=label, parent=parents[i], is_rare=i in RARE, support=len(idx))
            for model, run, instances_ in [('bert', b, bs), ('lgsa', l, ls)]:
                r.update({f'{model}_{k}': run['perlabel'][i][k] for k in ['tp', 'fp', 'fn', 'f1', 'precision', 'recall']})
                r.update({f'{model}_avg_true_score': mean(x['true_score'] for x in instances_),
                          f'{model}_avg_top_wrong_score': mean(x['wrong_score'] for x in instances_),
                          f'{model}_avg_margin': mean(x['margin'] for x in instances_),
                          f'{model}_same_parent_rate': mean(x['same_parent'] for x in instances_),
                          f'{model}_ranking_inversion_rate': mean(x['negative_outranks_true'] for x in instances_)})
            for metric in ['f1', 'avg_true_score', 'avg_top_wrong_score', 'avg_margin', 'ranking_inversion_rate']:
                r[f'delta_{metric}'] = r[f'lgsa_{metric}'] - r[f'bert_{metric}'] if len(idx) or metric == 'f1' else None
            r['macro_f1_contribution_pp'] = r['delta_f1'] * 100 / 28
            r['rare_f1_contribution_pp'] = r['delta_f1'] * 100 / 7 if i in RARE else None
            rows.append(r)
            # Compare the same (sample, true, wrong) under both models, even if top wrong switches.
            for w in range(28):
                if w == i:
                    continue
                valid = idx[~b['y'][idx, w]]
                if not len(valid):
                    continue
                bp, lp = b['p'][valid, w], l['p'][valid, w]
                bm, lm = b['p'][valid, i] - bp, l['p'][valid, i] - lp
                fixed.append(dict(seed=seed, true_label=label, wrong_label=NAMES[w], same_parent=bool(parents[i] == parents[w]),
                                  paired_samples=len(valid), bert_wrong_score=float(bp.mean()), lgsa_wrong_score=float(lp.mean()),
                                  delta_wrong_score=float((lp - bp).mean()), bert_margin=float(bm.mean()), lgsa_margin=float(lm.mean()),
                                  delta_margin=float((lm - bm).mean()), bert_wrong_fp_count=int((bp >= THRESHOLD).sum()),
                                  lgsa_wrong_fp_count=int((lp >= THRESHOLD).sum()),
                                  bert_inversion_count=int((b['z'][valid, w] > b['z'][valid, i]).sum()),
                                  lgsa_inversion_count=int((l['z'][valid, w] > l['z'][valid, i]).sum())))
    return rows, fixed


def export_predictions(out, runs):
    fields = ['model', 'seed', 'sample_id', 'text', 'ground_truth_labels', 'predicted_labels_at_0.30']
    fields += [f'logit_{n}' for n in NAMES] + [f'score_{n}' for n in NAMES]
    def rows():
        for (model, seed), r in runs.items():
            for i, sid in enumerate(r['ids']):
                yield {
                               **dict(model=model.upper(), seed=seed, sample_id=str(sid), text=r['texts'][i],
                                      ground_truth_labels=json.dumps([NAMES[t] for t in np.flatnonzero(r['y'][i])])),
                               'predicted_labels_at_0.30': json.dumps([NAMES[t] for t in np.flatnonzero(r['p'][i] >= THRESHOLD)]),
                               **{f'logit_{n}': float(r['z'][i, j]) for j, n in enumerate(NAMES)},
                               **{f'score_{n}': float(r['p'][i, j]) for j, n in enumerate(NAMES)}}
    write_csv(out / 'test_sample_predictions.csv', rows(), fields)


def figures(out, pairs, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, model in zip(axes, ['BERT', 'LGSA']):
        rows = [r for r in pairs if r['model'] == model and r['seed'] == 'all_seeds'][:15][::-1]
        colors = ['#198278' if r['same_parent'] else '#d88544' for r in rows]
        ax.barh([f"{r['true_label']} -> {r['top_wrong']}" for r in rows], [r['count'] for r in rows], color=colors)
        ax.set_title(model)
        ax.set_xlabel('GT-label instances summed across 3 seeds')
        for j, r in enumerate(rows):
            ax.text(r['count'], j, f"  {r['count']:,}", va='center', fontsize=9)
        ax.set_xlim(0, max([r['count'] for r in rows], default=1) * 1.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#198278', label='Same parent'), Patch(color='#d88544', label='Cross parent')], loc='lower center', ncol=2)
    fig.suptitle('Top 15 hard-negative pairs (exclude every ground-truth label)', fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(out / 'top15_hard_confusion_pairs.png', dpi=200)
    fig.savefig(out / 'top15_hard_confusion_pairs.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, analysis, title in zip(axes, ['hard_negative_ranking', 'false_positive_at_0.30'], ['Ranking: one top negative per GT label', 'Decision: FP labels at threshold 0.30']):
        rows = [r for r in summary if r['seed'] != 'all_seeds' and r['analysis'] == analysis and r['scope'] == 'all_labels']
        labels = [f"{r['model']}\n{s}" for r in rows for s in [r['seed']]]
        same = np.array([r['same_parent_rate'] if r['same_parent_rate'] is not None else np.nan for r in rows]) * 100
        cross = np.array([r['cross_parent_rate'] if r['cross_parent_rate'] is not None else np.nan for r in rows]) * 100
        ax.bar(labels, same, color='#198278', label='Same parent')
        ax.bar(labels, cross, bottom=same, color='#d88544', label='Cross parent')
        for j, n in enumerate(same):
            if np.isfinite(n):
                ax.text(j, n / 2, f'{n:.1f}%', ha='center', va='center', color='white')
            else:
                ax.text(j, 5, 'No FP', ha='center')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Share (%)')
        ax.set_title(title)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.15), ncol=2)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(out / 'same_parent_vs_cross_parent.png', dpi=200)
    fig.savefig(out / 'same_parent_vs_cross_parent.pdf')
    plt.close(fig)


def report(out, runs, summary, stable, comparisons):
    lines = ['# Experiment 2 多标签错误分析', '',
             '已验证 job3006610 的六组配对 test 预测；复用训练结束时加载 best checkpoint 后保存的 logits。阈值固定 0.30。', '',
             '## Same-parent 与 Cross-parent', '',
             '| 模型 | 分析 | 范围 | seed42 | seed123 | seed2026 | 均值 ± SD |',
             '|---|---|---|---:|---:|---:|---:|']
    for model in ['BERT', 'LGSA']:
        for analysis in ['hard_negative_ranking', 'false_positive_at_0.30']:
            for scope in ['all_labels', 'sibling_eligible_labels']:
                rs = [next(r for r in summary if r['model'] == model and r['seed'] == s and r['analysis'] == analysis and r['scope'] == scope) for s in SEEDS]
                vals = [r['same_parent_rate'] for r in rs]
                formatted = [f'{v*100:.2f}%' if v is not None else 'n/a (无事件)' for v in vals]
                stat = f'{np.mean(vals)*100:.2f} ± {np.std(vals, ddof=1)*100:.2f}%' if all(v is not None for v in vals) else 'n/a'
                lines.append('| ' + ' | '.join([model, analysis, scope, *formatted, stat]) + ' |')
    lines += ['', '表中的百分比为 same-parent；cross-parent = 100% − same-parent。SD 为三个 seed 的样本标准差。FP 为 0 时比例缺失，不写成 0%。', '',
              '## LGSA 稳定 Top 15 pairs', '', '| True | Wrong | 42 count / rank | 123 count / rank | 2026 count / rank |', '|---|---|---:|---:|---:|']
    selected = [r for r in stable if r['model'] == 'LGSA' and r['stable_top_k_3_of_3']]
    lines += ['| ' + ' | '.join([r['true_label'], r['top_wrong'], *[f"{r[f'seed{s}_count']} / {r[f'seed{s}_rank']}" for s in SEEDS]]) + ' |' for r in selected]
    if not selected:
        lines.append('没有 pair 在三个 seed 中都进入 Top 15。')
    lines += ['', '## Grief / Relief 与错误变化', '',
              '| Label | Seed | BERT F1 | LGSA F1 | Δ top-wrong score | Δ margin | Δ ranking inversion rate |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['label'] in ['grief', 'relief']:
            lines.append(f"| {r['label']} | {r['seed']} | {r['bert_f1']*100:.2f}% | {r['lgsa_f1']*100:.2f}% | {r['delta_avg_top_wrong_score']:+.4f} | {r['delta_avg_margin']:+.4f} | {r['delta_ranking_inversion_rate']*100:+.2f} pp |")
    lines += ['', '## 定义与解释边界', '',
              '- Hard-negative 单位为 sample × GT label；先排除该样本的全部 GT，再按 raw logits 取最大负标签。精确并列按官方 fine ID 最小者取一个，并记录 tie_count。',
              '- 每个真实标签都产生一行；同一个样本的不同 GT 可以共享同一个 top wrong。它是 ranking diagnostic，只有 wrong_score ≥ 0.30 才属于 FP。',
              '- Sibling-eligible ranking 范围为 parent 至少有 2 个 children 的真实标签；Neutral singleton 不进入。FP eligible 范围为样本至少含一个 sibling-eligible GT，FP 每个样本/标签只计一次。',
              '- uniform_negative_expected_rate 是每个 GT 实例从所有非 GT labels 均匀抽取时同父的机会率，考虑分支大小和 GT 共现。它是描述性参照，不是统计显著性检验。',
              '- 三 seed pooled counts 含重复 test 样本，不是三个独立 test sets；不把它们当独立观测来宣称显著性。3/3 出现与 3/3 Top 15 分开记录。',
              '- Top-wrong 身份可因模型改变；matched_pair_score_comparison.csv 比较同一 (sample,true,wrong) 条件下的两模型分数，全部排除 wrong 属于 GT 的样本。grief→sadness 和 sadness→grief 都包含在内。',
              '- 跨模型 sigmoid score 变化可能包含校准差异。需结合 TP/FP/FN、margin、ranking inversion 和固定 pair 的变化，不把 score 降低单独解释为语义辨别能力提高。',
              '- grief/relief 样本很少；hierarchy v0 是 sentiment-derived 实验结构，coarse labels 不是独立人工 GT。分析不证明 HGS/HGA 的因果机制或 DPT 会有效。',
              '- 机会率调整后若 sibling 集中仍在三个 seed 中稳定出现，才支持尝试 DPT-inspired sibling discrimination；不自动开始 Experiment 3。', '',
              '输入/脚本/输出哈希见 ANALYSIS_SUCCESS.json。原始 checkpoint 的哈希与 reload 证据沿用 HPC 证书，本机没有重新读取 checkpoint 二进制。', '']
    (out / 'ERROR_ANALYSIS_REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='Downloaded normalized outputs/job3006610 directory')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--test-tsv', type=Path, default=REPO / '数据集/GoEmotions/processed/test.tsv')
    p.add_argument('--hierarchy', type=Path, default=ROOT.parent / 'exp2_1_multiseed/frozen/hierarchy.json')
    args = p.parse_args()
    require(not args.out.exists(), 'Refuse to overwrite existing analysis; choose a new --out')
    parents, sizes = hierarchy(args.hierarchy)
    texts = official_test(args.test_tsv)
    runs, proof, integrity = load_runs(args.inputs, texts, parents, args.hierarchy)
    # No output is created before every real input passes all checks.
    args.out.mkdir(parents=True)
    hard, fps, pairs, summary, labels = [], [], [], [], []
    for (model, seed), run in runs.items():
        h, f = instances(run, model, seed, parents, sizes)
        hard += h
        fps += f
        pairs += aggregate_pairs(h, model.upper(), seed)
        summary += summaries(h, f, model.upper(), seed)
        labels += label_summary(run, h, model, seed, parents)
    for model in ['BERT', 'LGSA']:
        h = [r for r in hard if r['model'] == model]
        f = [r for r in fps if r['model'] == model]
        pairs += aggregate_pairs(h, model, 'all_seeds')
        summary += summaries(h, f, model, 'all_seeds')
    expected_hard = sum(int(r['y'].sum()) for r in runs.values())
    require(len(hard) == expected_hard, 'GT instance count reconciliation failed')
    require(len(fps) == sum(sum(x['fp'] for x in r['perlabel']) for r in runs.values()), 'FP event count reconciliation failed')
    require(sum(r['count'] for r in pairs if r['seed'] != 'all_seeds') == len(hard), 'Pair aggregation reconciliation failed')
    stable = stability(pairs, 15)
    comparisons, fixed = paired_comparison(runs, hard, parents)
    write_csv(args.out / 'hard_negative_instances.csv', hard)
    write_csv(args.out / 'hard_confusion_pairs.csv', pairs)
    write_csv(args.out / 'sibling_vs_cross_parent_summary.csv', summary)
    write_csv(args.out / 'per_label_confusion_summary.csv', labels)
    write_csv(args.out / 'seed_stability_confusion.csv', stable)
    fp_fields = ['model', 'seed', 'sample_id', 'false_positive_label', 'false_positive_parent', 'same_parent_any_gt',
                 'sample_has_sibling_eligible_gt', 'score', 'matching_gt_labels', 'ground_truth_labels', 'text']
    write_csv(args.out / 'false_positive_instances.csv', fps, fp_fields)
    write_csv(args.out / 'bert_vs_lgsa_per_label_comparison.csv', comparisons)
    write_csv(args.out / 'matched_pair_score_comparison.csv', fixed)
    parent_groups = Counter((r['model'], r['seed'], r['true_parent'], r['wrong_parent']) for r in hard)
    write_csv(args.out / 'parent_level_confusion.csv', [dict(model=m, seed=s, true_parent=t, wrong_parent=w, same_parent=t == w, count=n)
                                                     for (m, s, t, w), n in sorted(parent_groups.items())])
    export_predictions(args.out, runs)
    figures(args.out, pairs, summary)
    report(args.out, runs, summary, stable, comparisons)
    write_json(args.out / 'ANALYSIS_SUCCESS.json', dict(status='EXPERIMENT_2_ERROR_ANALYSIS_VERIFIED',
               threshold=THRESHOLD, seeds=SEEDS, models=list(MODEL_DIR), unique_test_examples=5427,
               prediction_rows=5427 * 6, hard_negative_rows=len(hard), fp_rows=len(fps),
               hierarchy_sha256=sha(args.hierarchy), test_tsv_sha256=sha(args.test_tsv), analyzer_sha256=sha(__file__),
               python=platform.python_version(), numpy=np.__version__, input_provenance=proof,
               original_hpc_integrity=integrity, tie_break='max raw logits; lowest official fine ID for exact ties',
               input_scope='Saved best-checkpoint predictions; no new training/inference or threshold tuning',
               artifacts_sha256={f.name: sha(f) for f in sorted(args.out.iterdir()) if f.is_file()}))
    print('EXPERIMENT_2_ERROR_ANALYSIS_VERIFIED', args.out.resolve())


if __name__ == '__main__':
    main()
