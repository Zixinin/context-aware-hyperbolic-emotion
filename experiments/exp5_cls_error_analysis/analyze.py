"""CPU-only diagnosis of sealed CLS predictions. No model load, inference or tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
          'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
          'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
          'sadness', 'surprise', 'neutral']
SEEDS = (42, 123, 2026)
RARE = (16, 21, 23, 19, 12, 24, 14)
RARE5 = tuple(i for i in RARE if i not in (16, 23))
PAIRS = (('anger', 'annoyance'), ('sadness', 'disappointment'), ('fear', 'nervousness'))
THRESHOLD = 0.30
SPLITS = {'train': ('train.tsv', 43410, '1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5'),
          'validation': ('dev.tsv', 5426, '575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62'),
          'test': ('test.tsv', 5427, '0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4')}
SOURCE_DIRS = ('exp5_label_aware_euclidean', 'exp5a_controls', 'exp5a_controls_multiseed')
ASL_URL = 'https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html'
LEAM_URL = 'https://aclanthology.org/P18-1216/'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def divided(a, b):
    return float(a / b) if b else None


def pct(value, digits=2):
    return 'n.a.' if value is None or pd.isna(value) else f'{100 * value:.{digits}f}'


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(np.isfinite(logits).all(), 'Non-finite saved logits')
    return np.exp(-np.logaddexp(0.0, -logits))


def validate_predictions(logits, targets, ids, dataset):
    require(logits.shape == targets.shape == dataset['gold'].shape, 'Prediction shape differs from canonical data')
    require(logits.ndim == 2 and logits.shape[1] == 28, 'Expected [N,28] predictions')
    require(np.array_equal(ids, dataset['ids']), 'Prediction IDs/order differ from canonical data')
    require(np.array_equal(targets, dataset['gold']), 'Prediction targets differ from canonical data')
    require(np.isin(targets, [0, 1]).all(), 'Nonbinary gold')
    return sigmoid(logits)


def load_data(repo):
    result, seen = {}, set()
    for split, (filename, size, digest) in SPLITS.items():
        path = repo / '数据集/GoEmotions/processed' / filename
        require(path.is_file(), f'Missing canonical data: {path}')
        require(sha(path) == digest, f'Official TSV checksum differs: {path}')
        with path.open(encoding='utf-8', newline='') as stream:
            records = list(csv.reader(stream, delimiter='\t'))
        require(len(records) == size, f'Wrong {split} size')
        ids, texts, gold = [], [], np.zeros((size, 28), dtype=np.uint8)
        for row, record in enumerate(records):
            require(len(record) == 3, f'Malformed {split} row {row}')
            labels = [int(i) for i in record[1].split(',')]
            require(labels and len(labels) == len(set(labels)) and all(0 <= i < 28 for i in labels), 'Invalid gold labels')
            gold[row, labels] = 1
            texts.append(record[0])
            ids.append(record[2])
        require(len(set(ids)) == size and not seen.intersection(ids), 'Duplicate/leaking sample IDs')
        seen.update(ids)
        result[split] = dict(ids=np.asarray(ids), gold=gold, texts=texts, path=path)
    counts = result['train']['gold'].sum(axis=0)
    require(tuple(sorted(range(28), key=lambda i: (int(counts[i]), i))[:7]) == RARE, 'Predefined training-frequency rare labels differ')
    return result


def manifest_check(directory):
    directory = Path(directory)
    expected = {}
    for line in (directory / 'PACKAGE_MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(Path(name).name == name and name not in expected, 'Unsafe/duplicate source manifest entry')
        expected[name] = digest
    actual = {p.name: sha(p) for p in directory.iterdir()
              if p.is_file() and p.suffix in ('.py', '.json', '.md', '.sh')}
    require(actual == expected, f'Source package changed: {directory}')
    return actual


def check_sources(repo):
    sources = {name: manifest_check(repo / 'experiments' / name) for name in SOURCE_DIRS}
    original, control, multi = [repo / 'experiments' / n for n in SOURCE_DIRS]
    a, b = read(control / 'frozen_dependency.json'), read(multi / 'frozen_dependency.json')
    require(a['source_sha256'] == sources[SOURCE_DIRS[0]], 'Control frozen source reference differs')
    require(a['manifest_sha256'] == sha(original / 'PACKAGE_MANIFEST.sha256'), 'Original manifest reference differs')
    require(b['source_sha256'] == dict(controls=sources[SOURCE_DIRS[1]], exp5a=sources[SOURCE_DIRS[0]]), 'Multi-seed frozen reference differs')
    require(b['control_manifest_sha256'] == sha(control / 'PACKAGE_MANIFEST.sha256'), 'Control manifest reference differs')
    return sources


def check_inventory(root):
    proof = read(root / 'SUCCESS.json')
    expected = proof['artifacts_sha256']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SUCCESS.json'}
    require(actual == set(expected), f'Frozen run inventory changed: {root}')
    for name, digest in expected.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact manifest path')
        require(sha(root / relative) == digest, f'Frozen artifact checksum differs: {root / relative}')
    return proof


def metric_rows(gold, probs, split, seed):
    pred, gold = probs >= THRESHOLD, gold.astype(bool)
    rows = []
    for i, label in enumerate(LABELS):
        tp = int((gold[:, i] & pred[:, i]).sum())
        fp = int((~gold[:, i] & pred[:, i]).sum())
        fn = int((gold[:, i] & ~pred[:, i]).sum())
        tn = int((~gold[:, i] & ~pred[:, i]).sum())
        support, positive = tp + fn, tp + fp
        rows.append(dict(split=split, seed=seed, label_id=i, label=label, rare7=i in RARE,
                         rare5=i in RARE5, support=support, tp=tp, fp=fp, fn=fn, tn=tn,
                         predicted_positive=positive, precision=divided(tp, positive) or 0.0,
                         recall=divided(tp, support) or 0.0, f1=divided(2*tp, 2*tp+fp+fn) or 0.0,
                         precision_defined=bool(positive), recall_defined=bool(support),
                         predicted_to_gold_ratio=divided(positive, support),
                         false_positive_rate=divided(fp, fp+tn)))
    return rows


def global_metrics(rows):
    table = pd.DataFrame(rows)
    tp, fp, fn = table[['tp', 'fp', 'fn']].sum()
    return dict(micro_precision=divided(tp, tp+fp) or 0.0,
                micro_recall=divided(tp, tp+fn) or 0.0,
                micro_f1=divided(2*tp, 2*tp+fp+fn) or 0.0,
                macro_precision=float(table.precision.mean()), macro_recall=float(table.recall.mean()),
                macro_f1=float(table.f1.mean()),
                weighted_precision=float(np.average(table.precision, weights=table.support)),
                weighted_recall=float(np.average(table.recall, weights=table.support)),
                weighted_f1=float(np.average(table.f1, weights=table.support)))


def verify_run(root, seed, data, source, expected_config=None):
    require(root.name == f'cls_seed{seed}', f'Not a formal CLS run: {root}')
    proof = check_inventory(root)
    cfg, env, selection = [read(root / name) for name in ('config.json', 'environment.json', 'checkpoint_selection.json')]
    require(cfg['seed'] == seed and cfg['mode'] == 'full' and cfg['representation'] == 'cls', 'Wrong seed/model/mode')
    protocol = dict(epochs=4, fixed_threshold=0.30, max_length=50, effective_batch_size=16, learning_rate=5e-5)
    require(all(cfg[k] == v for k, v in protocol.items()), 'Frozen CLS protocol differs')
    if expected_config is not None:
        require(cfg == expected_config, 'Full frozen configuration differs')
    expected_source = dict(controls=source[SOURCE_DIRS[1]], exp5a=source[SOURCE_DIRS[0]])
    expected_status = 'EXP5A_CONTROL_MODEL_VERIFIED'
    if seed != 42:
        expected_source = dict(multiseed=source[SOURCE_DIRS[2]], frozen=expected_source)
        expected_status = 'EXP5A_MULTISEED_MODEL_VERIFIED'
    require(proof['source_sha256'] == env['source_sha256'] == expected_source, 'Run/source provenance differs')
    require(proof['status'] == expected_status and proof['seed'] == seed, 'Invalid original SUCCESS identity')
    require(env['status'] == 'completed' and env['seed'] == seed and env['representation'] == 'cls', 'Incomplete/wrong run')
    require(env['first_batch_backward_verified'] and env['checkpoint_reload_verified'], 'Original gradient/reload proof absent')
    require(env['completed_optimizer_steps'] == env['total_optimizer_steps'] == 10856, 'Incomplete formal training')
    require((root / 'checkpoint_best.pt').stat().st_size > 0, 'Empty checkpoint')
    require(env['hostname'] == 'D3020-A3' and '4090' in env['gpu'], 'Unexpected training host/GPU')
    preflight_path = Path(env['preflight_path'])
    require(sha(preflight_path) == env['preflight_sha256'], 'Preflight proof changed')
    preflight = read(preflight_path)
    require(preflight['source_sha256'] == expected_source, 'Preflight source differs')
    require(preflight['packages'] == env['packages'], 'Recorded environment packages differ')
    require(env['parameters_runtime'] == preflight['parameters']['cls'], 'Runtime parameter proof differs')
    require(env['parameters_runtime']['trainable_parameters'] == 107741212 and
            env['parameters_runtime']['query_parameters'] == 0, 'Not the frozen CLS architecture')
    if seed != 42:
        require(env['seed42_reference'] == preflight['seed42_reference'], 'Frozen seed42 reference differs')
    require((root / 'pip-freeze.txt').read_text() == preflight['pip_freeze'], 'Recorded training environment differs from preflight')
    counts = data['train']['gold'].sum(axis=0).tolist()
    statistics = read(root / 'dataset_statistics.json')
    require(statistics['train_label_counts'] == counts and statistics['rare_ids'] == list(RARE), 'Recorded training counts/rare labels differ')
    require(statistics['label_order'] == {str(i): label for i, label in enumerate(LABELS)}, 'Label order differs')
    require(statistics['splits'] == {k: len(v['ids']) for k, v in data.items()}, 'Recorded split sizes differ')
    sample_ids = read(root / 'sample_ids.json')
    require(sample_ids == {k: data[k]['ids'].tolist() for k in ('train', 'validation')}, 'Training/validation membership differs')
    history = pd.read_csv(root / 'training_log.csv')
    require(history.epoch.tolist() == [1, 2, 3, 4] and history.global_step.tolist() == [2714, 5428, 8142, 10856], 'Epoch/step history differs')
    require(np.isfinite(history.to_numpy()).all(), 'Non-finite history')
    best = history.iloc[int(history.validation_macro_f1.to_numpy().argmax())]
    require(selection['epoch'] == int(best.epoch) and selection['global_step'] == int(best.global_step) and selection['threshold'] == THRESHOLD, 'Wrong validation-selected checkpoint')
    predictions = {}
    for split in ('validation', 'test'):
        path = root / f'{split}_predictions.npz'
        require(path.is_file(), f'Missing saved {split} logits/targets/IDs: {path}. No inference recovery is authorized.')
        with np.load(path, allow_pickle=False) as npz:
            require({'logits', 'targets', 'ids'}.issubset(npz.files), f'Missing saved prediction fields: {path}')
            probs = validate_predictions(npz['logits'], npz['targets'], npz['ids'], data[split])
        predictions[split] = probs
        rows = metric_rows(data[split]['gold'], probs, split, seed)
        measured, saved = global_metrics(rows), read(root / f'{split}_fixed_0.30_metrics.json')
        for name, value in measured.items():
            require(abs(value - saved[name]) < 1e-12, f'{split} metric mismatch: {name}')
        require(saved == read(root / f'{split}_metrics.json')['fixed_0.30'], 'Fixed-threshold metric copies differ')
        per_label = pd.read_csv(root / f'{split}_fixed_0.30_per_label.csv').sort_values('label_id')
        require(per_label.label.tolist() == LABELS and per_label.support.tolist() == [r['support'] for r in rows], 'Per-label support/order differs')
        np.testing.assert_allclose(per_label[['precision', 'recall', 'f1']],
                                   [[r[k] for k in ('precision', 'recall', 'f1')] for r in rows], rtol=0, atol=1e-12)
        if split == 'validation':
            require(abs(selection['validation_macro_f1'] - measured['macro_f1']) < 1e-12 and
                    abs(best.validation_macro_f1 - measured['macro_f1']) < 1e-12, 'Checkpoint validation metric differs')
    evidence = dict(seed=seed, run_dir=str(root), checkpoint_sha256=sha(root / 'checkpoint_best.pt'),
                    success_sha256=sha(root / 'SUCCESS.json'), best_epoch=selection['epoch'],
                    checkpoint_verification='sealed hash + original numerical reload proof; NOT loaded again',
                    validation_macro_f1=selection['validation_macro_f1'], training_gpu=env['gpu'],
                    steps=env['completed_optimizer_steps'], preflight_path=str(preflight_path),
                    preflight_sha256=sha(preflight_path))
    return predictions, evidence


def diagnosis(precision, recall):
    if recall < .50 and precision - recall >= .10:
        return 'recall_dominant'
    if precision < .50 and recall - precision >= .10:
        return 'precision_dominant'
    if precision < .50 and recall < .50:
        return 'both_low'
    return 'no_dominant_flag'


def aggregate_labels(metrics, predictions, data):
    rows = []
    for split in ('validation', 'test'):
        for i, label in enumerate(LABELS):
            local = metrics[(metrics.split == split) & (metrics.label_id == i)].sort_values('seed')
            row = dict(split=split, label_id=i, label=label, rare7=i in RARE, rare5=i in RARE5,
                       train_positive=int(data['train']['gold'][:, i].sum()),
                       validation_positive=int(data['validation']['gold'][:, i].sum()),
                       test_positive=int(data['test']['gold'][:, i].sum()))
            for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'predicted_positive'):
                values = local[key].to_numpy(dtype=float)
                row.update({f'{key}_mean': float(values.mean()), f'{key}_sd': float(values.std(ddof=1)),
                            f'{key}_min': float(values.min()), f'{key}_max': float(values.max())})
            for seed in SEEDS:
                r = local[local.seed == seed].iloc[0]
                row.update({f'{key}_seed{seed}': float(r[key]) for key in ('precision', 'recall', 'f1', 'predicted_positive')})
            row['diagnosis'] = diagnosis(row['precision_mean'], row['recall_mean'])
            row['low_precision'] = row['precision_mean'] < .50
            row['low_recall'] = row['recall_mean'] < .50
            row['seed_sensitive'] = row['f1_sd'] >= .05 or row['f1_max'] - row['f1_min'] >= .10
            row['zero_prediction_seeds'] = int((local.predicted_positive == 0).sum())
            stack = np.stack([predictions[s][split][:, i] >= THRESHOLD for s in SEEDS])
            disagreement = stack.any(axis=0) & ~stack.all(axis=0)
            gold = data[split]['gold'][:, i].astype(bool)
            row['any_seed_disagreement_n'] = int(disagreement.sum())
            row['any_seed_disagreement_fraction'] = float(disagreement.mean())
            row['gold_positive_seed_disagreement_n'] = int((gold & disagreement).sum())
            row['gold_positive_seed_disagreement_fraction'] = divided((gold & disagreement).sum(), gold.sum())
            row['missed_by_all_three_seeds_n'] = int((gold & ~stack.any(axis=0)).sum())
            row['recalled_by_all_three_seeds_n'] = int((gold & stack.all(axis=0)).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def group_metrics(metrics):
    groups = dict(all28=list(range(28)), rare7=list(RARE), rare5_excluding_grief_relief=list(RARE5),
                  grief_relief_only=[16, 23], nonrare21=[i for i in range(28) if i not in RARE])
    rows = []
    for split in ('validation', 'test'):
        for seed in SEEDS:
            for name, ids in groups.items():
                local = metrics[(metrics.split == split) & (metrics.seed == seed) & metrics.label_id.isin(ids)]
                tp, fp, fn = [int(x) for x in local[['tp', 'fp', 'fn']].sum()]
                rows.append(dict(split=split, seed=seed, group=name, n_labels=len(ids),
                                 support=int(local.support.sum()), predicted_positive=int(local.predicted_positive.sum()),
                                 tp=tp, fp=fp, fn=fn, macro_precision=float(local.precision.mean()),
                                 macro_recall=float(local.recall.mean()), macro_f1=float(local.f1.mean()),
                                 micro_precision=divided(tp, tp+fp), micro_recall=divided(tp, tp+fn),
                                 micro_f1=divided(2*tp, 2*tp+fp+fn)))
    return pd.DataFrame(rows)


def pair_counts(gold, probs, a, b):
    ga, gb = gold[:, a].astype(bool), gold[:, b].astype(bool)
    pa, pb = probs[:, a] >= THRESHOLD, probs[:, b] >= THRESHOLD
    aonly, bonly, both, neither = ga & ~gb, gb & ~ga, ga & gb, ~ga & ~gb
    out = dict(gold_a=int(ga.sum()), gold_b=int(gb.sum()), gold_both=int(both.sum()),
               gold_a_only=int(aonly.sum()), gold_b_only=int(bonly.sum()), gold_neither=int(neither.sum()),
               predicted_a=int(pa.sum()), predicted_b=int(pb.sum()), predicted_both=int((pa & pb).sum()),
               gold_jaccard=divided(both.sum(), (ga | gb).sum()),
               predicted_jaccard=divided((pa & pb).sum(), (pa | pb).sum()),
               both_gold_both_recalled=int((both & pa & pb).sum()),
               both_gold_a_only_recalled=int((both & pa & ~pb).sum()),
               both_gold_b_only_recalled=int((both & ~pa & pb).sum()),
               both_gold_both_missed=int((both & ~pa & ~pb).sum()),
               a_only_a_missed=int((aonly & ~pa).sum()), b_only_b_missed=int((bonly & ~pb).sum()),
               a_only_a_FN_b_FP=int((aonly & ~pa & pb).sum()),
               b_only_b_FN_a_FP=int((bonly & ~pb & pa).sum()),
               neither_both_FP=int((neither & pa & pb).sum()))
    out['a_only_a_FN_b_FP_rate'] = divided(out['a_only_a_FN_b_FP'], out['gold_a_only'])
    out['b_only_b_FN_a_FP_rate'] = divided(out['b_only_b_FN_a_FP'], out['gold_b_only'])
    return out


def pair_tables(predictions, data):
    counts, patterns, examples = [], [], []
    chosen = {(LABELS.index(a), LABELS.index(b)) for a, b in PAIRS}
    chosen |= {(b, a) for a, b in chosen}
    for split in ('validation', 'test'):
        gold = data[split]['gold'].astype(bool)
        for seed in SEEDS:
            probs, pred = predictions[seed][split], predictions[seed][split] >= THRESHOLD
            for a, b in combinations(range(28), 2):
                row = dict(split=split, seed=seed, label_a=LABELS[a], label_b=LABELS[b],
                           predefined_pair=(a, b) in chosen)
                counts.append(dict(row, **pair_counts(gold, probs, a, b)))
                if (a, b) not in chosen:
                    continue
                for ga, gb in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    g = (gold[:, a] == ga) & (gold[:, b] == gb)
                    for pa, pb in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        n = int((g & (pred[:, a] == pa) & (pred[:, b] == pb)).sum())
                        patterns.append(dict(row, gold_a=ga, gold_b=gb, predicted_a=pa, predicted_b=pb,
                                             stratum_n=int(g.sum()), count=n, rate_within_gold_stratum=divided(n, g.sum())))
                if split == 'validation':
                    for left, right in ((a, b), (b, a)):
                        mask = gold[:, left] & ~gold[:, right] & ~pred[:, left] & pred[:, right]
                        # Deterministic source-order examples; no test-text inspection or example optimization.
                        for idx in np.flatnonzero(mask)[:3]:
                            examples.append(dict(seed=seed, split=split, id=str(data[split]['ids'][idx]),
                                                 gold_present_label=LABELS[left], gold_absent_label=LABELS[right],
                                                 p_present_label=float(probs[idx, left]), p_absent_label=float(probs[idx, right]),
                                                 all_gold_labels=','.join(LABELS[i] for i in np.flatnonzero(gold[idx])),
                                                 text=data[split]['texts'][idx]))
    return pd.DataFrame(counts), pd.DataFrame(patterns), pd.DataFrame(examples, columns=[
        'seed', 'split', 'id', 'gold_present_label', 'gold_absent_label', 'p_present_label',
        'p_absent_label', 'all_gold_labels', 'text'])


def probability_tables(predictions, data):
    summaries, histograms, boundaries, loss_proxy = [], [], [], []
    edges = np.array([0, .05, .10, .20, .25, .30, .35, .40, .50, .70, .90, 1.0])
    gold = data['validation']['gold'].astype(bool)
    for seed in SEEDS:
        probs = predictions[seed]['validation']
        for i, label in enumerate(LABELS):
            g, p = gold[:, i], probs[:, i]
            pred = p >= THRESHOLD
            categories = dict(gold_positive=g, gold_negative=~g, tp=g & pred, fn=g & ~pred,
                              fp=~g & pred, tn=~g & ~pred)
            for category, mask in categories.items():
                values = p[mask]
                qs = np.quantile(values, [.10, .25, .50, .75, .90]) if len(values) else [None]*5
                summaries.append(dict(split='validation', seed=seed, label=label, population=category,
                                      n=int(mask.sum()), mean=float(values.mean()) if len(values) else None,
                                      **dict(zip(('q10', 'q25', 'median', 'q75', 'q90'), qs))))
                if category.startswith('gold_'):
                    hist, _ = np.histogram(values, edges)
                    for j, n in enumerate(hist):
                        histograms.append(dict(split='validation', seed=seed, label=label, population=category,
                                               bin_left=float(edges[j]), bin_right=float(edges[j+1]),
                                               right_closed=bool(j == len(hist)-1), count=int(n), population_n=len(values)))
            fn, fp = g & ~pred, ~g & pred
            near_fn = int((fn & (p >= .25)).sum())
            near_fp = int((fp & (p < .35)).sum())
            boundaries.append(dict(split='validation', seed=seed, label=label, support=int(g.sum()),
                                   fn=int(fn.sum()), fp=int(fp.sum()), fn_p_lt_025=int((fn & (p < .25)).sum()),
                                   fn_p_025_to_030=near_fn, fp_p_030_to_035=near_fp,
                                   fp_p_ge_035=int((fp & (p >= .35)).sum()),
                                   near_fn_fraction=divided(near_fn, fn.sum()), near_fp_fraction=divided(near_fp, fp.sum()),
                                   errors_near_fraction=divided(near_fn+near_fp, fn.sum()+fp.sum())))
            eps = np.finfo(np.float64).tiny
            pos_loss = -np.log(np.maximum(p[g], eps)).sum()
            neg_losses = -np.log1p(-np.minimum(p[~g], 1-np.finfo(np.float64).eps))
            easy = p[~g] < .05
            total = float(pos_loss + neg_losses.sum())
            loss_proxy.append(dict(split='validation', seed=seed, label=label, positive_n=int(g.sum()),
                                   negative_n=int((~g).sum()), negative_to_positive_ratio=divided((~g).sum(), g.sum()),
                                   easy_negative_p_lt_005_n=int(easy.sum()), positive_BCE_sum=float(pos_loss),
                                   negative_BCE_sum=float(neg_losses.sum()), easy_negative_BCE_sum=float(neg_losses[easy].sum()),
                                   negative_BCE_share=divided(neg_losses.sum(), total),
                                   easy_negative_BCE_share=divided(neg_losses[easy].sum(), total)))
    return [pd.DataFrame(x) for x in (summaries, histograms, boundaries, loss_proxy)]


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def report(data, summary, groups, pairs, boundary, evidence):
    test = summary[summary.split == 'test'].copy()
    val = summary[summary.split == 'validation'].copy()
    lines = ['# Exp5 CLS GoEmotions 错误分析', '',
             '## 1. 范围与核验', '',
             '只诊断正式 CLS seeds 42/123/2026，固定 sigmoid≥0.30。checkpoint 只核对封存哈希和原训练的数值重载证明，不反序列化权重，不新增推理、训练、阈值搜索或测试集调参。',
             '官方 TSV SHA256、跨 split ID、28-label 顺序、三种子样本/真实标签与原指标均已核对。下面是已选 checkpoint 的描述性结果；test 不参与候选方法、标签对或诊断阈值的自动选择。', '',
             md_table(['Seed', 'Best epoch', 'Steps', 'Validation Macro-F1 (%)', 'Checkpoint SHA256'],
                      [[e['seed'], e['best_epoch'], e['steps'], pct(e['validation_macro_f1']), e['checkpoint_sha256']] for e in evidence]), '',
             '## 2. 28-label 主表', '',
             'Train/Val/Test 为正例数。P/R/F1 为 test 三种子的逐种子指标均值，F1 ± 样本 SD。TP/FP/FN/Pred 为每个模型的计数均值，不是把同一测试集当成三份独立样本。完整整数计数与 validation 指标见 CSV。', '',
             md_table(['Label', 'Train/Val/Test', 'P %', 'R %', 'F1 % ±SD', 'TP/FP/FN/Pred 均值', '诊断'],
                      [[r.label + (' [rare]' if r.rare7 else ''), f'{r.train_positive}/{r.validation_positive}/{r.test_positive}',
                        pct(r.precision_mean), pct(r.recall_mean), f'{pct(r.f1_mean)} ±{pct(r.f1_sd)}',
                        '/'.join(f'{getattr(r,k+"_mean"):.1f}' for k in ('tp','fp','fn','predicted_positive')),
                        r.diagnosis + ('; seed_sensitive' if r.seed_sensitive else '')] for r in test.itertuples()]), '',
             '诊断是透明的描述规则，不是统计检验：recall_dominant 为 R<50% 且 P−R≥10 pp；precision_dominant 对称；both_low 为 P、R 均<50% 且未满足前两类。seed_sensitive 为 F1 样本 SD≥5 pp 或种子极差≥10 pp。未标记不等于模型没有问题。',
             'P 在零预测正例时按原协议记 0，CSV 的 precision_defined/zero_prediction_seeds 标明不可解释性；不能把它当有很多 FP。', '',
             '## 3. Validation 主导的问题定位', '',
             '后续假设优先依据 validation；test 仅描述既有错误。以下每项同时给出三种子计数，避免只凭总体均值下结论。']
    for name, flag in [('Recall 主导', 'recall_dominant'), ('Precision 主导', 'precision_dominant'), ('P/R 同时偏低', 'both_low')]:
        local = val[val.diagnosis == flag].sort_values('f1_mean')
        if local.empty:
            lines += ['', f'**{name}：** 没有标签满足预设描述规则。']
        else:
            parts = [f'{r.label}：P/R={pct(r.precision_mean)}/{pct(r.recall_mean)}%，FN/FP 均值={r.fn_mean:.1f}/{r.fp_mean:.1f}，'
                     f'三种子 Pred={r.predicted_positive_seed42:g}/{r.predicted_positive_seed123:g}/{r.predicted_positive_seed2026:g}' for r in local.itertuples()]
            lines += ['', f'**{name}：** ' + '；'.join(parts) + '。']
    lines += ['', '### 随机种子敏感性', '',
              md_table(['Label', 'Split', 'F1 seed42/123/2026 (%)', 'F1 SD pp', '正例中种子预测不一致', '三种子共同 FN'],
                       [[r.label, split, '/'.join(pct(getattr(r,f'f1_seed{s}')) for s in SEEDS), pct(r.f1_sd),
                         f'{r.gold_positive_seed_disagreement_n}/{getattr(r,"validation_positive" if split=="validation" else "test_positive")}',
                         r.missed_by_all_three_seeds_n]
                        for split, local in [('validation', val), ('test', test)]
                        for r in local.sort_values('f1_sd', ascending=False).head(5).itertuples()]),
              '', '只列各 split 波动最高的 5 个标签，完整标记见 label_summary.csv。三个种子不是三个独立测试集；极低支持数会放大 F1 波动，不能把 SD 当置信区间。', '',
              '## 4. Rare-7 与排除 grief/relief 的 Rare-5', '',
              'Rare-7 固定按 training 最低频的七类：grief、pride、relief、nervousness、embarrassment、remorse、fear。Rare-5 只排除 grief、relief，不重新选择标签。', '',
              md_table(['Split', 'Group', 'Macro P % ±SD', 'Macro R % ±SD', 'Macro F1 % ±SD', 'F1 seed42/123/2026 %'],
                       [[split, group] + [f'{pct(g[key].mean())} ±{pct(g[key].std(ddof=1))}' for key in ('macro_precision','macro_recall','macro_f1')] +
                        ['/'.join(pct(g[g.seed==s].iloc[0].macro_f1) for s in SEEDS)]
                        for split in ('validation','test') for group in ('rare7','rare5_excluding_grief_relief','grief_relief_only')
                        for g in [groups[(groups.split==split)&(groups.group==group)]]]), '',
              '七个稀有类的逐类 P/R、整数 TP/FP/FN 和 Pred 均在 label_metrics_by_seed.csv。组内 macro 不按 support 加权；额外提供 group 内 micro，不能混用。grief/relief 只作为独立敏感性诊断，不据此主张普遍长尾改善。', '',
              '## 5. 多标签相似情绪分析', '',
              '先按真实标签区分 A-only、B-only、both、neither，再分别统计四种预测模式。both 中同时预测两个标签是合法命中，不算混淆。A-only 中漏 A 且多报 B 只记为“相对现有标注的交叉错误”，不能证明语义误解、标注错误或 B 在文本中绝不成立。', '',
              md_table(['Pair', 'Split', 'Gold A-only/B-only/both', 'Pred both 均值', 'A-only 漏A多报B 均值/占比', 'B-only 漏B多报A 均值/占比', 'Gold both 全漏均值'],
                       [[f'{a}/{b}', split, f'{int(g.iloc[0].gold_a_only)}/{int(g.iloc[0].gold_b_only)}/{int(g.iloc[0].gold_both)}',
                         f'{g.predicted_both.mean():.1f}',
                         f'{g.a_only_a_FN_b_FP.mean():.1f}/{pct(g.a_only_a_FN_b_FP_rate.mean())}%',
                         f'{g.b_only_b_FN_a_FP.mean():.1f}/{pct(g.b_only_b_FN_a_FP_rate.mean())}%',
                         f'{g.both_gold_both_missed.mean():.1f}']
                        for a0,b0 in PAIRS for a,b in [sorted((a0,b0),key=LABELS.index)]
                        for split in ('validation','test')
                        for g in [pairs[(pairs.split==split)&(pairs.label_a==a)&(pairs.label_b==b)]]]), '',
              '占比分母为对应 gold-only 层样本数；0 分母显示 n.a.。pair_patterns.csv 给出每层全部计数，pair_overlap.csv 给出所有 378 对的描述统计，但不按 test 排名选择下一轮标签对。validation_pair_examples.csv 仅保存预定义三对的源顺序前 3 个交叉错误实例/方向/seed，无自动语义判定。', '',
              '## 6. Validation 概率与固定阈值附近错误', '',
              '只计算 validation。预定义 near-FN 为 0.25≤p<0.30，near-FP 为 0.30≤p<0.35；没有扫描、推荐或替换阈值。正负例、TP/FP/FN/TN 概率的均值和分位数，以及正负例直方图均保存在 CSV。', '',
              md_table(['Label', 'FN 均值', '其中 near-FN 均值', 'FP 均值', '其中 near-FP 均值', '近阈值错误占全部 FP+FN'],
                       [[label, f'{g.fn.mean():.1f}', f'{g.fn_p_025_to_030.mean():.1f}', f'{g.fp.mean():.1f}', f'{g.fp_p_030_to_035.mean():.1f}',
                         pct(divided(g.fn_p_025_to_030.sum()+g.fp_p_030_to_035.sum(),g.fn.sum()+g.fp.sum())) + '%']
                        for label in LABELS for g in [boundary[boundary.label==label]]]), '',
              '近阈值占比高提示阈值/概率尺度敏感，但不证明校准失败；远低于 0.25 的 FN 不能描述成“只差一点阈值”。validation_BCE_proxy.csv 只给已有 validation 概率上的 BCE 分解，不是训练梯度证据。', '',
              '## 7. 下一轮 Euclidean 假设，不实施新实验', '']
    recall_flags = val[val.diagnosis == 'recall_dominant']
    support = data['train']['gold'].sum(axis=0)
    lines += [f'**ASL：** training 每标签负/正比例范围为 {(len(data["train"]["ids"])-support).min()/support.max():.2f}–{((len(data["train"]["ids"])-support)/np.maximum(support,1)).max():.2f}；'
              f'validation 有 {len(recall_flags)} 个 recall_dominant 标签（' + ', '.join(recall_flags.label.tolist()) + '）。'
              '这些观测可支持检验“正负项不平衡是否影响召回”的假设，但不证明当前训练被 easy negatives 的梯度主导。'
              f'[ASL 原论文]({ASL_URL}) 通过非对称调制和负例概率平移降低易负例影响；原证据主要来自视觉多标签任务，不能直接保证 GoEmotions Macro/Rare-F1 提升。Precision 主导的错误也不会自动被 ASL 修复。', '']
    selected = pairs[(pairs.split=='validation') & pairs.predefined_pair]
    cross = int(selected.a_only_a_FN_b_FP.sum()+selected.b_only_b_FN_a_FP.sum())
    denominator = int(selected.gold_a_only.sum()+selected.gold_b_only.sum())
    lines += [f'**Semantic Label Embedding：** 预定义三对在 validation 的单边标注层中，三种子累计有 {cross}/{denominator} 次“漏真实类且多报另一类”'
              f'（{pct(divided(cross,denominator))}%，种子×标签对层重复观察，不是独立样本数）。'
              + ('这支持将这些层作为标签语义区分的待检验诊断目标。' if cross else '没有观察到此类交叉错误，不能据此声称需要修复这些标签对的混淆。') +
              f'[词与标签联合嵌入工作]({LEAM_URL}) 提供文本/标签共享表示的机制参考，但不是本项目的效果证明。'
              '语义相似也可能使相关标签更难区分，需同时监控合法共现与单边 FP/FN；不能把所有相似标签强制互斥。仅把自由分类器权重改名为 label embeddings 不构成新的语义机制。', '',
              '两条路线应先分别与冻结 CLS 基线做受控比较，不能由这次描述性 test 分析挑损失超参、阈值或最好种子。当前分析不启动 ASL、语义嵌入或其他实验。', '',
              '## 8. 缺失数据与科学边界', '',
              '- 本诊断不读取或生成 training logits。标准产物没有保存它们，training 只报告标签频次，不推测训练集 P/R、概率、梯度或表征。',
              '- checkpoint 本次不加载。哈希、配置、checkpoint 选择记录与原训练重载证明提供关联证据，不能替代新的独立 checkpoint→logits 前向复验。',
              '- 现有 logits 不能单独解释截断、讽刺、语境或标注噪声的因果作用；无此证据就不写成已发现原因。',
              '- 三个种子共享同一数据划分；均值与样本 SD 是有限复现性描述，不作显著性、普适性或因果主张。',
              '- 验证/测试指标均对应各自种子按 validation Macro-F1 选出的 checkpoint，不强制取最后 epoch。',
              '- 所有输出写入新的独立诊断目录。受保护源文件、三个原 CLS 运行目录及核验输入的文件清单和 SHA256 在前后逐一比较。', '']
    return '\n'.join(lines)


def analyze_predictions(data, predictions, evidence, out):
    counts = []
    for i, label in enumerate(LABELS):
        counts.append(dict(label_id=i, label=label, rare7=i in RARE, rare5=i in RARE5,
                           **{f'{split}_positive': int(d['gold'][:,i].sum()) for split,d in data.items()},
                           train_negative=int((1-data['train']['gold'][:,i]).sum()),
                           train_negative_to_positive_ratio=divided((1-data['train']['gold'][:,i]).sum(),data['train']['gold'][:,i].sum())))
    metrics = pd.DataFrame([r for seed in SEEDS for split in ('validation','test')
                            for r in metric_rows(data[split]['gold'],predictions[seed][split],split,seed)])
    summary, groups = aggregate_labels(metrics,predictions,data), group_metrics(metrics)
    pairs, patterns, examples = pair_tables(predictions,data)
    probabilities, histograms, boundaries, loss_proxy = probability_tables(predictions,data)
    tables = {'label_counts.csv': pd.DataFrame(counts), 'label_metrics_by_seed.csv': metrics,
              'label_summary.csv': summary, 'rare_group_metrics.csv': groups, 'pair_overlap.csv': pairs,
              'pair_patterns.csv': patterns, 'validation_pair_examples.csv': examples,
              'validation_probability_summary.csv': probabilities, 'validation_probability_bins.csv': histograms,
              'validation_threshold_neighborhood.csv': boundaries, 'validation_BCE_proxy.csv': loss_proxy}
    for name, table in tables.items():
        table.to_csv(out/name,index=False,float_format='%.15g')
    (out/'EXP5_CLS_ERROR_ANALYSIS.md').write_text(report(data,summary,groups,pairs,boundaries,evidence),encoding='utf-8')
    return tables


def snapshot(trees, extra):
    paths = set(extra)
    for tree in trees:
        paths.update(p for p in tree.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    require(out.parent == repo/'results/exp5_cls_error_analysis' and out.name.startswith('run_'), 'Output must be a NEW independent diagnostic run')
    require(not out.exists(), 'Refusing output overwrite; choose a fresh diagnostic directory')
    out.mkdir(parents=True,exist_ok=False)
    try:
        own_source = manifest_check(Path(__file__).resolve().parent)
        source = check_sources(repo)
        runs = {42: repo/'results/exp5a_controls/run_20261013T103802Z/cls_seed42',
                **{s: repo/f'results/exp5a_controls_multiseed/run_20261013T113904Z/cls_seed{s}' for s in (123,2026)}}
        for seed, root in runs.items():
            for name in ('SUCCESS.json','checkpoint_best.pt','validation_predictions.npz','test_predictions.npz'):
                require((root/name).is_file(), f'Missing input seed{seed}: {root/name}; no model inference/retraining authorized')
        extras = [repo/'数据集/GoEmotions/processed'/x[0] for x in SPLITS.values()]
        for root in runs.values():
            env = read(root/'environment.json')
            extras.append(Path(env['preflight_path']))
        extras.append(repo/'experiments/exp2_lgsa/LGSA-GoEmotions/assets/goemotions-v0/manifest.json')
        trees = [*runs.values(),*[repo/'experiments'/n for n in SOURCE_DIRS]]
        before = snapshot(trees,extras)
        data = load_data(repo)
        predictions, evidence = {}, []
        for seed in SEEDS:
            cfg = dict(read(repo/'experiments/exp5_label_aware_euclidean/config.json'),
                       experiment='Exp5A-Control', representation='cls', mode='full',
                       dropout_layout='independent_per_label_matching_frozen_exp5a',
                       head_initialization='skip_query_draw_then_normal_0.02_matching_exp5a')
            if seed != 42:
                cfg.update(experiment='Exp5A-Control-MultiSeed', seed=seed,
                           authorized_seeds=[123,2026], aggregation_seeds=list(SEEDS))
            env = read(runs[seed]/'environment.json')
            require(env['dataset_manifest_sha256'] == sha(extras[-1]), 'Recorded dataset manifest differs')
            if seed != 42:
                require(env['seed42_reference']['cls_success_sha256'] == sha(runs[42]/'SUCCESS.json'), 'Original seed42 CLS identity differs')
            predictions[seed], proof = verify_run(runs[seed],seed,data,source,cfg)
            evidence.append(proof)
            print(f'CLS_INPUT_VERIFIED seed={seed} best_epoch={proof["best_epoch"]} checkpoint_sha256={proof["checkpoint_sha256"]}',flush=True)
        manifest = read(extras[-1])
        require(manifest['status'] == 'GOEMOTIONS_ASSETS_READY' and manifest['rare_ids'] == list(RARE) and
                manifest['split_sizes'] == {k:len(v['ids']) for k,v in data.items()}, 'Asset manifest differs')
        tables = analyze_predictions(data,predictions,evidence,out)
        after = snapshot(trees,extras)
        require(before == after, 'Protected input files/inventory changed during diagnosis')
        write_json(out/'PROTECTED_INPUTS.json',dict(before_sha256=before,after_sha256=after,unchanged=True))
        write_json(out/'INPUT_AUDIT.json',dict(status='CLS_INPUTS_VERIFIED',seeds=evidence,source_sha256=source,
                   diagnosis_source_sha256=own_source,threshold=THRESHOLD,model_loaded=False,forward_calls=0,
                   training_calls=0,threshold_search=False,test_threshold_tuning=False,
                   numpy=np.__version__,pandas=pd.__version__,python=sys.version,host=platform.node(),
                   row_counts={name:len(table) for name,table in tables.items()},
                   missing=['training logits/gradients are not required and not generated']))
        artifacts = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write_json(out/'ANALYSIS_SUCCESS.json',dict(status='EXP5_CLS_ERROR_ANALYSIS_COMPLETE_VERIFIED',
                   completed_at=datetime.now(timezone.utc).isoformat(),artifact_sha256=artifacts,protected_inputs_unchanged=True))
        for name,digest in artifacts.items():
            require(sha(out/name)==digest,f'Output verification failed: {name}')
        print(f'EXP5_CLS_ERROR_ANALYSIS_COMPLETE_VERIFIED\nEXP5_CLS_ANALYSIS_DIR={out}',flush=True)
    except Exception as error:
        write_json(out/'ANALYSIS_FAILED.json',dict(status='ANALYSIS_NOT_COMPLETE',error_type=type(error).__name__,message=str(error)))
        (out/'MISSING_OR_INVALID_INPUTS.md').write_text('# 分析未完成\n\n'+str(error)+'\n\n未进行新训练、模型前向、阈值搜索或自动修复输入。保留目录供排查。\n',encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
