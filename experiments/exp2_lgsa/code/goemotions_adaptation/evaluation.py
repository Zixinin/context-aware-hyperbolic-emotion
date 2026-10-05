"""Only the 28 original fine labels contribute to any reported task metric."""
import csv
import numpy as np
from common import NAMES, require, write


def metrics(gold, scores, threshold, rare):
    require(gold.shape == scores.shape and gold.ndim == 2 and gold.shape[1] == 28, 'Metrics require 28 fine labels')
    require(np.isfinite(scores).all() and np.isin(gold, [0, 1]).all(), 'Invalid metric inputs')
    pred = scores >= threshold
    y = gold.astype(bool)
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    p, r, f = divide(tp, tp+fp), divide(tp, tp+fn), divide(2*tp, 2*tp+fp+fn)
    nonrare = [i for i in range(28) if i not in rare]
    result = {'threshold': float(threshold), 'examples': len(y), 'num_labels': 28,
        'micro_precision': float(divide(tp.sum(), (tp+fp).sum())),
        'micro_recall': float(divide(tp.sum(), (tp+fn).sum())),
        'micro_f1': float(divide(2*tp.sum(), (2*tp+fp+fn).sum())),
        'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
        'rare_mean_f1': float(f[rare].mean()), 'nonrare_mean_f1': float(f[nonrare].mean())}
    labels = [dict(label_id=i, label=NAMES[i], precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                   support=int((tp+fn)[i]), prediction_count=int((tp+fp)[i]),
                   tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]), is_rare=i in rare) for i in range(28)]
    return result, labels


def select_threshold(gold, scores, candidates, rare):
    trials = [metrics(gold, scores, t, rare)[0] for t in sorted(candidates)]
    best = max(trials, key=lambda r: (r['macro_f1'], -r['threshold']))
    return {'selected_threshold': best['threshold'], 'selection_split': 'validation',
            'objective': 'macro_f1', 'tie_break': 'smallest threshold', 'trials': trials}


def save_report(directory, name, gold, logits, rare, threshold):
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    summary, labels = metrics(gold, scores, threshold, rare)
    write(directory / (name + '_metrics.json'), summary)
    with (directory / (name + '_per_label.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    return summary


def plots(out, hierarchy, learned):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    initial = torch.load(out / 'initial_geometry.pt', map_location='cpu', weights_only=False)
    names = [hierarchy['labels'][i] for i in range(32)]
    for stage, value in {**{k: initial[k] for k in ('initial', 'hgs')}, 'learned_gat': learned}.items():
        x = value.detach().cpu().double().numpy()
        normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        fig, ax = plt.subplots(figsize=(11, 9))
        artist = ax.imshow(normalized @ normalized.T, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(32), names, rotation=90, fontsize=7)
        ax.set_yticks(range(32), names, fontsize=7)
        ax.set_title(stage + ': label cosine similarity')
        fig.colorbar(artist, ax=ax)
        fig.tight_layout(); fig.savefig(out / (stage + '_cosine.png'), dpi=180); plt.close(fig)
        coords = PCA(n_components=2).fit_transform(x)
        fig, ax = plt.subplots(figsize=(10, 8))
        for parent, children in hierarchy['slots'].items():
            ids = children + [parent]
            ax.scatter(coords[ids, 0], coords[ids, 1], label=names[parent])
            for i in ids:
                ax.annotate(names[i], coords[i], fontsize=7)
        ax.legend(); ax.set_title(stage + ': PCA (descriptive, not an orthogonality test)')
        fig.tight_layout(); fig.savefig(out / (stage + '_pca.png'), dpi=180); plt.close(fig)
