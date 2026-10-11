import random 
import torch 


def sort_pred_labels_by_logits(depth2label, prediction_scores):
    """ Sort the output labels of each sample by probability.
    
    Args:
        prediction_scores: torch.Size(label_deep, label_num)
    """
    predict_labels = []

    end_pos = 0
    for layer, score in enumerate(prediction_scores):   
        predict_labels.append([])
        start_pos = end_pos
        end_pos = start_pos + len(depth2label[layer])
        layer_score = score[start_pos:end_pos]     
        sort_score, sort_idx = torch.sort(layer_score, descending=True)
        predict_labels[-1].extend([depth2label[layer][idx] for idx in sort_idx])    

    return predict_labels


def sort_pred_labels_by_verb_logits(depth2label, prediction_scores):
    """ Sort the output labels of each sample by probability.
    
    Args:
        prediction_scores: torch.Size(label_deep, label_num)
    """
    predict_labels = []
    for layer, nodes in depth2label.items():   
        layer_score = prediction_scores[nodes]  
        sort_score, sort_idx = torch.sort(layer_score, descending=True)
        predict_labels.append([depth2label[layer][idx] for idx in sort_idx])    

    return predict_labels


def negative_sampling(depth2label, pos_label_ids, sample_num_list, mode='random', output_logits=None):
    """ Sampling negative samples for each sample.

    Args:
        depth2label: 
        pos_label_ids: List[List[int]]，Positive labels at each label level.
        sample_num_list: List[int], Number of negative samples to be sampled each layer.
        mode: 
            'random': Completely random without maintaining hierarchical structure.
            'hard-hpt': Negative labels with the highest model score for `HPT` decoder.
            'hard-verb': Negative labels with the highest model score for `SoftVerbalizer` decoder.
        output_logits: torch. Probability matrix of model output. Required only when mode in {'hard-htp', 'hard-verb'}.
             if mode == 'hard-hpt': torch.Size([layer_depth, layer_num]).
             if mode == 'hard-verb': torch.Size([layer_num]).

    Returns:
        negatives: List[int], Sampled negative label IDs.
            
    """
    assert mode in {'random', 'hard-hpt', 'hard-verb'}, f"'{mode}' mode is not supported"
    negatives = []
    if mode == 'random':
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in label_list if c not in pos_label_ids[layer]]
            negatives.append(random.sample(candidates, k=min(len(candidates), sample_num_list[layer])))

    elif mode == 'hard-hpt':
        b_preds = sort_pred_labels_by_logits(depth2label, output_logits) 
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in b_preds[layer] if c not in pos_label_ids[layer]]
            negatives.append(candidates[:sample_num_list[layer]])

    elif mode == 'hard-verb':
        b_preds = sort_pred_labels_by_verb_logits(depth2label, output_logits) 
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in b_preds[layer] if c not in pos_label_ids[layer]]
            negatives.append(candidates[:sample_num_list[layer]])

    return negatives
