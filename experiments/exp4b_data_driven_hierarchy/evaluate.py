"""Only the original 28 labels leave the public task classifier/evaluation."""
import csv
from pathlib import Path
import numpy as np
from b4lib import NAMES, RARE, require, write

class EmotionClassifier:
    """Non-parameter facade over the unchanged LGSA auxiliary MLM model.

    Upstream Prompt.num_labels means ALL hierarchy supervision nodes (old32,
    new56), not the number of emotion task classes. This public task classifier
    always returns Bx28 logits. It introduces no new head or learned parameters.
    """
    output_dimension = 28
    label_names = tuple(NAMES)

    def __init__(self, lgsa, hierarchy):
        self.lgsa, self.hierarchy = lgsa, hierarchy

    def forward(self, input_ids):
        import torch
        with torch.no_grad():
            _, auxiliary = self.lgsa.generate(input_ids, depth2label=self.hierarchy['depths'])
            return self.select(auxiliary, self.hierarchy)

    @staticmethod
    def select(auxiliary, hierarchy):
        import torch
        require(tuple(auxiliary.shape[1:]) == (hierarchy['num_depths'], hierarchy['num_nodes']), 'Auxiliary tensor dimensions changed')
        indices = torch.arange(28, device=auxiliary.device)
        depths = torch.tensor(hierarchy['fine_depths'], device=auxiliary.device)
        result = auxiliary[:, depths, indices]
        require(result.shape == (auxiliary.shape[0], 28) and bool(torch.isfinite(result).all()), 'Task classifier must output finite Bx28')
        return result

def predict(model, data, hierarchy, cfg, loader):
    import torch
    model.eval()
    classifier = EmotionClassifier(model, hierarchy)
    gold, logits = [], []
    with torch.no_grad():
        for batch in loader(data, cfg['eval_batch_size']):
            z = classifier.forward(batch['input_ids'].cuda())
            gold.append(batch['fine'].numpy())
            logits.append(z.cpu().numpy())
    return np.concatenate(gold), np.concatenate(logits)

def metrics(gold, scores, threshold, rare=RARE):
    require(gold.shape == scores.shape and gold.ndim == 2 and gold.shape[1] == 28, 'Only N x 28 task arrays accepted')
    require(np.isin(gold, [0, 1]).all() and np.isfinite(scores).all(), 'Invalid metric arrays')
    y, p = gold.astype(bool), scores >= threshold
    tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall, f1 = div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    support = tp + fn
    other = [i for i in range(28) if i not in rare]
    result = {'threshold': float(threshold), 'examples': len(y), 'num_labels': 28,
        'micro_precision': float(div(tp.sum(), (tp + fp).sum())),
        'micro_recall': float(div(tp.sum(), (tp + fn).sum())),
        'micro_f1': float(div(2 * tp.sum(), (2 * tp + fp + fn).sum())),
        'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()), 'weighted_f1': float(div((f1 * support).sum(), support.sum())),
        'rare_mean_f1': float(f1[rare].mean()), 'nonrare_mean_f1': float(f1[other].mean())}
    labels = [dict(label_id=i, label=NAMES[i], support=int(support[i]), precision=float(precision[i]),
        recall=float(recall[i]), f1=float(f1[i]), tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]),
        prediction_count=int(tp[i] + fp[i])) for i in range(28)]
    return result, labels

def probabilities(logits):
    # Exactly Exp2 LGSA's existing dtype-preserving clipped sigmoid, NOT float64 recasting.
    require(np.isfinite(logits).all(), 'Nonfinite logits')
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

def csvwrite(path, rows):
    require(bool(rows), 'Empty CSV report')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def save_report(directory, name, gold, logits, rare, threshold):
    summary, labels = metrics(gold, probabilities(logits), threshold, rare)
    write(directory / (name + '_metrics.json'), summary)
    csvwrite(directory / (name + '_per_label.csv'), labels)
    return summary
