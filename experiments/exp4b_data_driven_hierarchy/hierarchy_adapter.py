"""Exact full-tree adapter; fine IDs 0..27, auxiliary IDs 28..55, ROOT=-1."""
import re
from collections import defaultdict
from b4lib import NAMES, TREE_SHA, read, require, sha

PAIRS = [('fear', 'nervousness'), ('anger', 'annoyance'),
    ('disappointment', 'sadness'), ('confusion', 'curiosity'),
    ('excitement', 'joy'), ('desire', 'optimism')]

def adapt(obj):
    require(obj.get('source') == 'TRAIN_ONLY_RAW_RATER_AVERAGED', 'Wrong hierarchy source')
    require(obj.get('neutral_excluded_from_clustering') is True, 'Neutral clustering changed')
    root = obj.get('root')
    require(isinstance(root, dict) and root.get('node_id') == 'ROOT', 'Missing ROOT')
    nodes, parents_by_name, depths_by_name = {}, {}, {}
    stack = [(root, None, 0)]
    while stack:
        node, parent, depth = stack.pop()
        require(isinstance(node, dict), 'Malformed tree node')
        name = node.get('node_id')
        require(isinstance(name, str) and name not in nodes, 'Duplicate node/multiple parent/cycle')
        children = node.get('children')
        require(isinstance(children, list), 'Children must be an explicit list')
        require(depth < 100, 'Tree depth guard exceeded')
        nodes[name], parents_by_name[name], depths_by_name[name] = node, parent, depth
        for child in reversed(children):
            stack.append((child, name, depth + 1))
    leaves = {name: n for name, n in nodes.items() if not n['children']}
    require(set(leaves) == set(NAMES), 'Missing/extra/duplicate fine labels')
    for i, name in enumerate(NAMES):
        require(type(leaves[name].get('label_id')) is int
            and leaves[name]['label_id'] == i, 'Fine label IDs changed: ' + name)
    internal = [name for name, n in nodes.items() if n['children'] and name != 'ROOT']
    require(len(internal) == 28 and len(nodes) == 57, 'Full tree node count changed')
    require(set(internal) == {'EMOTION', 'NEUTRAL'} |
        {f'raw_cluster_{i:03d}' for i in range(1, 27)}, 'Internal IDs changed')
    require([n['node_id'] for n in root['children']] == ['EMOTION', 'NEUTRAL'], 'Root branches changed')
    require([n['node_id'] for n in nodes['EMOTION']['children']] == ['raw_cluster_026'], 'EMOTION root changed')
    require([n['node_id'] for n in nodes['NEUTRAL']['children']] == ['neutral'], 'Neutral branch changed')
    for name in internal:
        node = nodes[name]
        require('label_id' not in node, 'Internal node became a fine prediction label')
        if name.startswith('raw_cluster_'):
            require(len(node['children']) == 2, 'Binary cluster was compressed')
            support = node.get('bootstrap_support')
            require(isinstance(support, (float, int)) and 0 <= support <= 1, 'Invalid bootstrap support')
            require(type(node.get('stable_70')) is bool and node['stable_70'] == (support >= .70), 'Support annotation changed')
    def descendants(name):
        n = nodes[name]
        labels = [name] if not n['children'] else sum(
            (descendants(c['node_id']) for c in n['children']), [])
        require(len(labels) == len(set(labels)), 'Duplicate descendant leaf')
        if 'leaf_count' in n:
            require(n['leaf_count'] == len(labels), 'leaf_count changed')
        if 'leaf_labels' in n:
            require(sorted(n['leaf_labels']) == sorted(labels), 'leaf_labels changed')
        return labels
    require(set(descendants('ROOT')) == set(NAMES), 'Unreachable leaf')
    pair_checks = []
    for a, b in PAIRS:
        parent = parents_by_name[a]
        require(parent == parents_by_name[b] and parent.startswith('raw_cluster_'), 'Expected sibling pair missing')
        require(set(nodes[parent]['leaf_labels']) == {a, b}, 'Expected two-leaf clade changed')
        pair_checks.append({'labels': [a, b], 'parent': parent,
            'bootstrap_support': nodes[parent]['bootstrap_support']})
    # Preserve source names and original IDs. Numeric auxiliary ordering is deterministic.
    ordered = ['EMOTION', 'NEUTRAL'] + sorted(
        (n for n in internal if n.startswith('raw_cluster_')), key=lambda n: int(re.search(r'\d+$', n).group()))
    by_name = dict(zip(NAMES + ordered, range(56))) | {'ROOT': -1}
    labels = {i: name for name, i in by_name.items() if i >= 0}
    parents = {i: by_name[parents_by_name[name]] for name, i in by_name.items() if i >= 0}
    slots, depths = defaultdict(list), defaultdict(list)
    for i in range(56):
        if parents[i] != -1:
            slots[parents[i]].append(i)
        depths[depths_by_name[labels[i]] - 1].append(i)
    require(sorted(depths) == list(range(9)), 'Hierarchy must have nine modeled depths')
    # Old GAT convention: parent->child edges, then depth-prompt->node edges.
    edges = [(parents[c], c) for c in range(56) if parents[c] != -1]
    edges += [(56 + d, c) for d in sorted(depths) for c in depths[d]]
    fine_depths = [depths_by_name[n] - 1 for n in NAMES]
    summary = {'status': 'EXP4B_HIERARCHY_VALIDATION_PASS',
        'leaves': 28, 'final_classifier_output_dimension': 28,
        'internal_nodes_including_root': 29, 'internal_nodes_excluding_root': 28,
        'auxiliary_hierarchy_supervision_nodes': 56, 'modeled_depths': 9,
        'minimum_leaf_depth_from_ROOT': min(fine_depths) + 1,
        'maximum_leaf_depth_from_ROOT': max(fine_depths) + 1,
        'average_leaf_depth_from_ROOT': sum(d + 1 for d in fine_depths) / 28,
        'neutral_path': ['ROOT', 'NEUTRAL', 'neutral'],
        'retained_unstable_clusters': sum(not nodes[n]['stable_70'] for n in ordered if n.startswith('raw_cluster_')),
        'sanity_check_pairs': pair_checks, 'topology_modified': False,
        'auxiliary_nodes_are_final_prediction_labels': False}
    return {'labels': labels, 'parents': parents, 'slots': dict(slots), 'depths': dict(depths),
        'edges': edges, 'fine_depths': fine_depths, 'num_nodes': 56, 'num_depths': 9,
        'summary': summary, 'source': obj}

def load_hierarchy(path, enforce_pin=True):
    if enforce_pin:
        require(sha(path) == TREE_SHA, 'The approved Exp4A tree bytes changed')
    return adapt(read(path))

def targets(fine, hierarchy):
    require(fine and len(set(fine)) == len(fine)
        and all(type(i) is int and 0 <= i < 28 for i in fine), 'Invalid fine IDs')
    active = set(fine)
    for leaf in fine:
        current = hierarchy['parents'][leaf]
        while current != -1:
            active.add(current)
            current = hierarchy['parents'][current]
    result = []
    for d in range(hierarchy['num_depths']):
        layer = [-100] * hierarchy['num_nodes']
        for node in hierarchy['depths'][d]:
            layer[node] = int(node in active)
        result.extend(layer)
    return result, [int(i in fine) for i in range(28)]
