"""Sentiment-derived hierarchy; coarse supervision is derived, never ground truth."""
from common import NAMES, require, read


def load_hierarchy(path):
    obj = read(path)
    require(obj['version'] == 'sentiment-derived-v0', 'Unapproved hierarchy version')
    require(obj['status'] == 'ADOPTED_FOR_EXPERIMENT_2', 'Hierarchy has not been adopted')
    require(obj['root_is_prediction_target'] is False, 'ROOT cannot be a target')
    require(obj['num_fine_labels'] == 28 and obj['num_all_labels'] == 32, 'Wrong class counts')
    labels = sorted(obj['labels'], key=lambda x: x['fine_id'])
    require([x['fine_id'] for x in labels] == list(range(28)), 'Missing/duplicate fine labels')
    require([x['fine_label'] for x in labels] == NAMES, 'Original label IDs changed')
    coarse = {x['id']: x['name'] for x in obj['coarse_labels']}
    require(coarse == {28: 'Positive', 29: 'Negative', 30: 'Ambiguous', 31: 'Neutral'}, 'Invalid coarse classes')
    parents = {}
    slots = {p: [] for p in coarse}
    for label in labels:
        c, p = label['fine_id'], label['coarse_id']
        require(p in coarse and coarse[p] == label['coarse_label'], 'Invalid parent')
        parents[c] = p
        slots[p].append(c)
    require(all(slots.values()) and slots[31] == [27], 'Incorrect Neutral branch')
    parents.update({p: -1 for p in coarse})
    # Disjoint fine/coarse ID sets and all coarse parents=-1 guarantee acyclicity.
    depths = {0: list(coarse), 1: list(range(28))}
    edges = [(parents[c], c) for c in range(28)]
    edges += [(32 + depth, node) for depth, nodes in depths.items() for node in nodes]
    return {'labels': dict(enumerate(NAMES)) | coarse, 'parents': parents,
            'slots': slots, 'depths': depths, 'edges': edges, 'source': obj}


def targets(fine, hierarchy):
    require(bool(fine) and len(set(fine)) == len(fine), 'Empty or duplicate fine labels')
    require(all(type(i) is int and 0 <= i < 28 for i in fine), 'Invalid fine ID')
    active = set(fine) | {hierarchy['parents'][i] for i in fine}
    layers = [[-100] * 32 for _ in range(2)]
    for depth, nodes in hierarchy['depths'].items():
        for node in nodes:
            layers[depth][node] = int(node in active)
    return layers[0] + layers[1], [int(i in fine) for i in range(28)]
