"""Positive HierPCL kernel extracted/adapted from official DPT.

HTCDualPromptModel.calculate_contrastive_loss, lines281..290 at commit
43e2daf6578ed6df885329cf229c3f5b749e19e7. Inputs substitute existing LGSA
representations and exclude the negative-prompt key. One GT per term.
See SOURCE.md for the original expression and every change.
"""
import torch
import torch.nn as nn


def positive_label_contrastive(query, positives, negatives, temperature):
    # These five normalization/cosine operations come from the DPT kernel.
    query_vector = nn.functional.normalize(query, dim=0)
    pos_key_matrix = nn.functional.normalize(positives, dim=1)
    neg_key_matrix = nn.functional.normalize(negatives, dim=1)
    pos_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), pos_key_matrix)
    neg_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), neg_key_matrix)
    # DPT: numerator=sum(exp(pos_sim/tau)); denominator=numerator+sum(exp(neg_sim/tau));
    # loss=-log(numerator/denominator). Evaluate exactly that ratio in log space.
    positive = pos_sim / temperature
    negative = neg_sim / temperature
    loss = torch.logsumexp(torch.cat([positive, negative]), dim=0) - torch.logsumexp(positive, dim=0)
    return loss, pos_sim, neg_sim
