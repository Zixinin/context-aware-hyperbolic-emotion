"""Pure read-only diagnostic math. No model construction, optimizer or data I/O."""
from __future__ import annotations
import hashlib
import math
import numpy as np

RARE = ['grief', 'pride', 'relief', 'nervousness', 'embarrassment', 'remorse', 'fear']

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def label_comparison(gold, p0, p1, names, train_support):
    from sklearn.metrics import average_precision_score
    gold = np.asarray(gold)
    need(gold.shape == p0.shape == p1.shape and gold.shape[1] == 27, 'Validation shape differs')
    need(len(names) == 27 and len(set(names)) == 27 and set(RARE) <= set(names), 'Label identity differs')
    need(np.isin(gold, [0, 1]).all(), 'Nonbinary targets')
    need(all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in [p0, p1]), 'Invalid probabilities')
    preds = [p >= .30 for p in [p0, p1]]
    rows = []
    for k, name in enumerate(names):
        pos, neg = gold[:, k] == 1, gold[:, k] == 0
        a, b = preds[0][:, k], preds[1][:, k]
        row = dict(label=name, train_support=int(train_support[k]), validation_support=int(pos.sum()),
                   rare=name in RARE,
                   TP_to_FN=int((pos & a & ~b).sum()), FN_to_TP=int((pos & ~a & b).sum()),
                   FP_to_TN=int((neg & a & ~b).sum()), TN_to_FP=int((neg & ~a & b).sum()),
                   changed_predictions=int((a != b).sum()),
                   mean_probability_shift=float((p1[:, k] - p0[:, k]).mean()),
                   fraction_scores_shifted_down=float((p1[:, k] < p0[:, k]).mean()))
        for tag, p, pred in zip(['A0', 'A1'], [p0, p1], [a, b]):
            tp, fp, fn = int((pos & pred).sum()), int((neg & pred).sum()), int((pos & ~pred).sum())
            row.update({tag+'_support': int(pos.sum()), tag+'_tp': tp, tag+'_fp': fp, tag+'_fn': fn,
                        tag+'_predicted_positive_count': int(pred.sum()),
                        tag+'_precision': tp/(tp+fp) if tp+fp else 0.,
                        tag+'_recall': tp/(tp+fn) if tp+fn else 0.,
                        tag+'_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                        tag+'_mean_probability_all': float(p[:, k].mean()),
                        tag+'_mean_positive_probability': float(p[pos, k].mean()) if pos.any() else None,
                        tag+'_mean_negative_probability': float(p[neg, k].mean()) if neg.any() else None,
                        tag+'_average_precision': float(average_precision_score(pos, p[:, k])) if pos.any() else None})
        for key in ['support','tp','fp','fn','predicted_positive_count','precision','recall','f1',
                    'mean_probability_all','mean_positive_probability','mean_negative_probability','average_precision']:
            row['delta_'+key] = None if row['A0_'+key] is None else row['A1_'+key] - row['A0_'+key]
        row['macro_f1_contribution_pp'] = 100*row['delta_f1']/27
        row['rare_f1_contribution_pp'] = 100*row['delta_f1']/7 if name in RARE else 0.
        rows.append(row)
    summary = dict(num_examples=len(gold), threshold=.30,
        changed_example_count=int((preds[0] != preds[1]).any(1).sum()),
        changed_example_fraction=float((preds[0] != preds[1]).any(1).mean()),
        changed_label_decisions=int((preds[0] != preds[1]).sum()),
        A0_total_predicted_positives=int(preds[0].sum()), A1_total_predicted_positives=int(preds[1].sum()),
        TP_to_FN=sum(r['TP_to_FN'] for r in rows), FN_to_TP=sum(r['FN_to_TP'] for r in rows),
        macro_delta_pp=sum(r['macro_f1_contribution_pp'] for r in rows),
        rare_delta_pp=sum(r['rare_f1_contribution_pp'] for r in rows),
        labels_with_recall_decline=sum(r['delta_recall'] < 0 for r in rows),
        rare_labels_with_recall_decline=sum(r['rare'] and r['delta_recall'] < 0 for r in rows),
        zero_support_labels=[r['label'] for r in rows if not r['validation_support']])
    return rows, summary

def poincare_distances(e):
    e = np.asarray(e, dtype=np.float64)
    need(e.ndim == 2 and np.isfinite(e).all(), 'Nonfinite embeddings')
    norms2 = (e*e).sum(1)
    need((norms2 < 1).all(), 'Label vectors outside ball')
    diff2 = ((e[:, None, :] - e[None, :, :])**2).sum(-1)
    d = np.arccosh(1 + 2*diff2/((1-norms2[:, None])*(1-norms2[None, :])))
    need(np.isfinite(d).all() and np.array_equal(d, d.T), 'Invalid distances')
    return d

def graph_audit(edges):
    edges = set(edges)
    need(edges and all(a != b for a,b in edges), 'Invalid hierarchy edges')
    nodes = sorted({x for edge in edges for x in edge})
    descendants = {n: set() for n in nodes}
    for a,b in edges:
        descendants[a].add(b)
    for _ in range(len(nodes)):
        old = {n: set(v) for n,v in descendants.items()}
        for n in nodes:
            for child in old[n]:
                descendants[n].update(old[child])
        if old == descendants:
            break
    need(not any(n in descendants[n] for n in nodes), 'Hierarchy cycle')
    covers = {(a,b) for a,b in edges if not any(c != b and b in descendants[c] for c in descendants[a])}
    parents = {n: sorted(a for a,b in covers if b == n) for n in nodes}
    depths = {}
    def depth(n):
        if n not in depths:
            depths[n] = 0 if not parents[n] else 1+max(depth(a) for a in parents[n])
        return depths[n]
    for n in nodes:
        depth(n)
    closure = {(a,b) for a in nodes for b in descendants[a]}
    return dict(nodes=nodes, roots=sorted(n for n in nodes if not parents[n]), parents=parents,
                depths=depths, edges=sorted(edges), covers=sorted(covers),
                redundant_edges=sorted(edges-covers), missing_closure=sorted(closure-edges))

def choose_cohorts(records, names, support, size=16):
    # All choices derive from TRAIN labels/IDs only. Never consult validation/test errors.
    order = sorted(records, key=lambda r: (hashlib.sha256(('Exp6C1D-v1:'+r['id']).encode()).hexdigest(), r['id']))
    frequent = set(sorted(range(27), key=lambda k: (-support[k], names[k]))[:7])
    predicates = {'single_positive': lambda r: len(r['labels']) == 1,
                  'multi_positive': lambda r: len(r['labels']) > 1,
                  'common_top7_train': lambda r: bool(set(r['labels']) & frequent)}
    for label in RARE:
        k = names.index(label)
        predicates['rare_'+label] = lambda r, k=k: k in r['labels']
    result = {name: [r for r in order if keep(r)][:size] for name,keep in predicates.items()}
    need(all(result[n] for n in result), 'Required train cohort empty')
    return result

def gradient_statistics(named, gb, ga, grouping):
    """None means structurally disconnected; accumulate float64 one tensor at a time."""
    import torch
    rows = []
    for group, include in grouping.items():
        n0=n1=0
        b2=a2=dot=0.
        count=0
        for (name,p), b,a in zip(named, gb, ga):
            if not include(name):
                continue
            count += p.numel()
            if b is not None:
                need(bool(torch.isfinite(b).all()), 'Nonfinite BCE gradient')
                v=b.detach().double(); b2 += float(v.square().sum()); n0+=1
            if a is not None:
                need(bool(torch.isfinite(a).all()), 'Nonfinite auxiliary gradient')
                w=a.detach().double(); a2 += float(w.square().sum()); n1+=1
            if b is not None and a is not None:
                dot += float((v*w).sum())
        bnorm, anorm = math.sqrt(b2), math.sqrt(a2)
        rows.append(dict(module=group, num_parameters=count, bce_connected_tensors=n0, aux_connected_tensors=n1,
            bce_connection='connected' if n0 else 'not connected by design',
            aux_connection='connected' if n1 else 'not connected by design',
            bce_norm=bnorm, weighted_aux_norm=anorm,
            weighted_aux_to_bce_norm=anorm/bnorm if bnorm else None,
            cosine=max(-1.,min(1.,dot/(bnorm*anorm))) if bnorm*anorm else None,
            dot_product=dot,
            total_norm=math.sqrt(max(0.,b2+a2+2*dot))))
    return rows
