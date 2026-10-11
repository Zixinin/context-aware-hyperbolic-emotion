"""DPT-derived sibling hard-negative loss; no new parameters or random sampling."""
from collections import Counter
import math
import torch
from upstream.dpt_sampling import negative_sampling
from upstream.dpt_positive import positive_label_contrastive


class PeerHardNegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, parent_of, temperature=0.1, top_k=1):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        self.parent_of = tuple(parent_of)
        if not self.parent_of or any(type(p) is not int for p in self.parent_of):
            raise ValueError('parent_of must contain one integer parent per fine ID')
        self.temperature = temperature
        self.top_k = top_k
        self.families = {p: [i for i, value in enumerate(self.parent_of) if value == p]
                         for p in sorted(set(self.parent_of))}

    def siblings(self, label):
        return [i for i in self.families[self.parent_of[label]] if i != label]

    def select(self, logits, gold):
        if logits.shape != gold.shape or logits.ndim != 2 or logits.shape[1] != len(self.parent_of):
            raise ValueError('Expected matching [B,num_fine_labels] logits and gold')
        if not bool(torch.isfinite(logits).all()) or not bool(((gold == 0) | (gold == 1)).all()):
            raise ValueError('Nonfinite scores or non-binary targets')
        terms = []
        for i in range(len(gold)):
            gt = torch.where(gold[i] == 1)[0].tolist()
            for y in gt:
                family = self.families[self.parent_of[y]]
                # Execute the verbatim OFFICIAL DPT hard-verb sampling function.
                negatives = negative_sampling({0: family}, [gt], [self.top_k],
                    mode='hard-verb', output_logits=logits[i].detach())[0]
                if negatives:
                    if any(n in gt or self.parent_of[n] != self.parent_of[y] for n in negatives):
                        raise RuntimeError('Invalid sibling negative')
                    terms.append((i, y, negatives))
        return terms

    def forward(self, anchors, label_vectors, logits, gold):
        if anchors.ndim != 2 or label_vectors.ndim != 2 or anchors.shape[1] != label_vectors.shape[1]:
            raise ValueError('Anchors [B,H] and keys [C,H] must share H')
        if len(anchors) != len(logits) or len(label_vectors) != len(self.parent_of):
            raise ValueError('Batch/label shape mismatch')
        if not bool(torch.isfinite(anchors).all() and torch.isfinite(label_vectors).all()):
            raise ValueError('Nonfinite representations')
        terms = self.select(logits, gold)
        losses, pos, neg = [], [], []
        pairs = Counter()
        for i, y, negatives in terms:
            loss, p, n = positive_label_contrastive(anchors[i], label_vectors[y:y+1],
                                                   label_vectors[negatives], self.temperature)
            losses.append(loss)
            pos.append(p[0].detach())
            neg.append(n.max().detach())
            pairs.update((y, label) for label in negatives)
        if not terms:
            zero = anchors.sum() * 0.0 + label_vectors.sum() * 0.0
            return zero, {'number_of_peer_terms': 0, 'number_of_samples_with_peer_terms': 0,
                          'mean_positive_similarity': 0.0, 'mean_hard_negative_similarity': 0.0,
                          'mean_peer_margin': 0.0, 'selected_pairs': {}}
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite peer loss')
        p, n = torch.stack(pos), torch.stack(neg)
        return loss, {'number_of_peer_terms': len(terms),
            'number_of_samples_with_peer_terms': len({i for i, _, _ in terms}),
            'mean_positive_similarity': float(p.mean()), 'mean_hard_negative_similarity': float(n.mean()),
            'mean_peer_margin': float((p - n).mean()),
            'selected_pairs': {f'{y}->{n}': count for (y, n), count in sorted(pairs.items())}}
