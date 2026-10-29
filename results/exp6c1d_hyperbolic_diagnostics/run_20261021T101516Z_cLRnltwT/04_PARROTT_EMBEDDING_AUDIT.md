# 04 Parrott embedding audit

以实际GoEmotions label_tree.tsv为对象，不把Parrott概念层数直接当作本实现层数。TSV为child/ancestor；原trainer使用Relations(reverse=True)，本诊断同样先反转为parent→child再作约简/闭包。
Roots=['root']；最长有向路径边数=3，含根最多4层。
原始边75，cover边33，可由其他路径推出的冗余边42，闭包中未显式列出的边0。
原trainer先从闭包恢复tree_relations供可视化，再移除root相关边；实际PoincareModel(train_data=transitive_relations)使用去root的传递关系，而非仅直接父子边。约简描述结构，不把闭包边误认为全部直接父子边。
27个向量与原版Exp6B矩阵按canonical names精确对齐；缺失树节点=[]。
nervousment在树中=True；nervousness在树中=True；nervousness在发布向量映射中=True。
树同时保留nervousment额外节点及正确nervousness。按原trainer，额外节点进入去root的训练关系；最终仅按config中正式label2idx键导出27类，nervousment不导出。可确认其进入训练关系，无法从发布向量恢复其实际数值影响；不把两节点合并或重新训练。

距离使用 c=1 Poincare公式 d(x,y)=acosh(1+2||x-y||²/((1-||x||²)(1-||y||²)))，float64，无裁剪/重新投影/距离归一化。
按直接父节点交集定义同父；未知父节点的对不进入同父/异父比较，完整CSV保留它们。
同父分布：{"n": 64, "min": 5.063452675599793, "median": 9.804033801090974, "mean": 9.51845775440734, "p95": 10.520036440976474, "max": 10.57018226802474}
异父分布：{"n": 287, "min": 9.340401552006158, "median": 12.097048862450029, "mean": 11.999904810754677, "p95": 14.006700347688984, "max": 14.578605118391403}
标签范数：{"n": 27, "min": 0.9927296809644236, "median": 0.9966615733955332, "mean": 0.9960402693376649, "p95": 0.9987070209503655, "max": 0.9989144614981014}
距离/层级的关联仅说明先验结构，不证明下游识别收益；各标签固定，不能声称在A1训练中被挤走。

| Rare | rank | 最近邻 | 双曲距离 |
|---|---:|---|---:|
| grief | 1 | anger | 9.532109 |
| grief | 2 | disgust | 9.609241 |
| grief | 3 | sadness | 9.745134 |
| grief | 4 | remorse | 10.019650 |
| grief | 5 | disapproval | 10.131465 |
| pride | 1 | optimism | 9.334984 |
| pride | 2 | admiration | 9.454421 |
| pride | 3 | relief | 9.771663 |
| pride | 4 | love | 9.843159 |
| pride | 5 | excitement | 9.955429 |
| relief | 1 | caring | 9.433581 |
| relief | 2 | admiration | 9.663522 |
| relief | 3 | pride | 9.771663 |
| relief | 4 | desire | 9.796633 |
| relief | 5 | approval | 9.822846 |
| nervousness | 1 | disgust | 9.340402 |
| nervousness | 2 | disapproval | 9.862021 |
| nervousness | 3 | annoyance | 9.893207 |
| nervousness | 4 | embarrassment | 10.082428 |
| nervousness | 5 | disappointment | 10.142421 |
| embarrassment | 1 | disappointment | 9.882266 |
| embarrassment | 2 | nervousness | 10.082428 |
| embarrassment | 3 | annoyance | 10.147417 |
| embarrassment | 4 | fear | 10.224888 |
| embarrassment | 5 | anger | 10.314974 |
| remorse | 1 | fear | 9.526731 |
| remorse | 2 | anger | 9.811434 |
| remorse | 3 | grief | 10.019650 |
| remorse | 4 | annoyance | 10.059118 |
| remorse | 5 | disappointment | 10.120614 |
| fear | 1 | remorse | 9.526731 |
| fear | 2 | annoyance | 9.545621 |
| fear | 3 | anger | 9.748065 |
| fear | 4 | sadness | 10.031736 |
| fear | 5 | embarrassment | 10.224888 |

grief–sadness：9.74513373

grief–remorse：10.01964996

relief–joy：10.68728999

relief–optimism：10.63686756

relief–gratitude：10.17204607

## 完整原作者标签预训练代码（仅文本读取，绝不导入/执行）
```python
"""
Runs our hyperbolic entailment cones on the synthetic data representing a uniform tree of some
fixed branching factor and some fixed depth. These trees are in data/toy/. This code will produce
an animation when embedding dimension is 2, and the animation will be opened in your browser at the
end of the training.
"""

# +
import csv
from collections import OrderedDict
import os
import pickle
import random
import re
from random import *

from smart_open import smart_open
import argparse
from utils.poincare_model import *
from utils.relations import *
from utils.eucl_cones_model import *
from utils.hyp_cones_model import *
from utils.poincare_viz import *
from utils.eval import *

import logging
import numpy as np
import sys
import plotly
from plotly.offline import plot
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# TODO: fill here your plotly details before running.
plotly.tools.set_credentials_file(username='', api_key='')

from config import get_dicts

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=str, help='go_emotion | ED | ED_easy_4 | ED_hard_a | ED_hard_b | ED_hard_c | ED_hard_d ', default='ED')
parser.add_argument("--model", type=str, help='poincare | hyp_cones ', default='poincare')
parser.add_argument("--dim", type=int, default=100)
args = parser.parse_args()

default_params = {
    'model' : args.model, # poincare or hyp_cones (initialized with Poincare)
    'tree_depth' : 3, # a number between 1 and 7
    'level_branch' : 4,  # a number between 3 and 4
    'remove_root': True,

    'print_every': 5,
    'size': args.dim, # embedding dimension. If 2, this code will produce an animation.
    'lr': 0.025, # Learning rate
    'opt': 'rsgd', # rsgd or exp_map or sgd

    'burn_in': 20,  # Number of epochs to use for burn-in initialization
    'epsilon': 1e-5,
    'seed': 7, # random seed
    'num_negative': 5, # Number of negative samples to use (5 for go_emotion, 10 for empathatic dialogs)
    'neg_sampling_power': 0.75,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

    'neg_sampl_strategy': 'true_neg',  # 'all', 'true_neg' , 'all_non_leaves' or 'true_neg_non_leaves'
    'where_not_to_sample': 'ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
    'neg_edges_attach': 'child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
    'always_v_in_neg': True,  # always include the true edge (u,v) as negative.

    'loss_type': 'nll', # 'nll', 'neg', 'maxmargin'

    'maxmargin_margin': 2,
    'neg_r': 1,
    'neg_t': 2,
    'neg_mu': 1.0,  # Balancing factor between the positive and negative terms

    'epochs': 2000,  # Number of epochs to use
    'batch_size': 16,  # Size of batch to use for training
}

params = default_params.copy()
param_str_list = ['%s:%s' % (key, params[key]) for key in sorted(params.keys())]
figure_name = '; '.join(param_str_list)
animation = create_animation(figure_name)

if args.tree == 'go_emotion':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'go_emotion')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'
    
elif args.tree == 'ED':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'    
    
elif args.tree == 'ED_easy_4':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_easy_4')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_a':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_a')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_b':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_b')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'     

elif args.tree == 'ED_hard_c':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_c')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'

elif args.tree == 'ED_hard_d':
    data_directory = os.path.join((os.path.abspath('')), 'data', 'ED_hard_d')
    data_file_path = os.path.join(data_directory, 'label_tree.tsv')
    root_label = 'root'
    
# Recovers the tree from the transitive closure of a DAG
def recover_tree_from_transitive_closure(relations):
    all_nodes_set = set()
    for rel in relations:
        all_nodes_set.add(rel[0])
        all_nodes_set.add(rel[1])

    ancestors = {}
    for node in all_nodes_set:
        ancestors[node] = []
    for rel in relations:
        if rel[0] != rel[1]:
            ancestors[rel[1]].append(rel[0])

    new_relations = []
    for node in all_nodes_set:
        num_ancestors = len(ancestors[node])
        for ancestor in ancestors[node]:
            if len(ancestors[ancestor]) == num_ancestors - 1:
                new_relations.append((ancestor, node))
    return new_relations

def read_tree_data(data=args.tree):
    # Load the tree data:
    transitive_relations = Relations(file_path=data_file_path, reverse=True)
    tree_relations = recover_tree_from_transitive_closure(transitive_relations)
                                                                                 
    # All direct children of root
    transitive_relations_without_root = []
    tree_relations_without_root = []
    for rel in tree_relations:
        if rel[0] != root_label:
            tree_relations_without_root.append(rel)

    for rel in transitive_relations:
        if rel[0] != root_label:
            transitive_relations_without_root.append(rel)

    return transitive_relations_without_root, tree_relations_without_root

transitive_relations, tree_relations = read_tree_data()

label2idx, idx2label = get_dicts(args.tree, return_emb_dicts=False)
class_names = [v for k, v in sorted(idx2label.items(), key=lambda item: item[0])]

# Create the Poincare model
model = PoincareModel(train_data=transitive_relations,
                      dim=params['size'],
                      lr=params['lr'],
                      opt=params['opt'],
                      burn_in=params['burn_in'],
                      epsilon=params['epsilon'],
                      seed=params['seed'],
                      # logger=logger,
                      num_negative=params['num_negative'],
                      ### How to sample negatives for an edge (u,v)
                      neg_sampl_strategy=params['neg_sampl_strategy'],
                      # 'all' (all nodes used for negative sampling) or 'true_neg' (only not connected nodes)
                      where_not_to_sample=params['where_not_to_sample'],
                      # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                      always_v_in_neg=params['always_v_in_neg'],  # always include the true edge (u,v) as negative.
                      neg_sampling_power=params['neg_sampling_power'],  # 0 for uniform, 1 for unigram, 0.75 for word2vec
                      ### How to use the negatives in the loss function
                      neg_edges_attach=params['neg_edges_attach'],
                      # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                      loss_type=params['loss_type'],
                      maxmargin_margin=params['maxmargin_margin'],
                      neg_r=params['neg_r'],
                      neg_t=params['neg_t'],
                      neg_mu=params['neg_mu'],
                      )

for i in range(int(params['epochs'] / params['print_every'])):
    print('Starting epoch ' + str(params['print_every'] * i))

    model.train(epochs=params['print_every'], batch_size=params['batch_size'], print_every=params['print_every'])

    # Animation
    if params['size'] == 2:
        figure = poincare_2d_visualization(
            model,
            animation=animation,
            epoch=(params['print_every'] * i),
            eval_result='',
            avg_loss=0,
            avg_pos_loss=0,
            avg_neg_loss=0,
            tree=list(tree_relations),
            show_node_labels=class_names,
            figure_title=figure_name,
            num_nodes=None)

if default_params['model'] == 'poincare':
    if args.dim == 2: plot(animation)
    word2vec = {}
    for k in label2idx.keys():
        if k in model.kv:
            word2vec[k] = model.kv[k].astype(float)

    # word2vec['sad'].astype(float)
    bin_directory = os.path.join((os.path.abspath('')), 'label_tree')
    bin_file_path = os.path.join(bin_directory, args.tree + '.bin')
    pickle.dump(word2vec, open(bin_file_path, 'wb'))
    # model.save(f'saved_poincare_model_{args.tree}.pt')
    os._exit(0)


def poincare_ball_dist(u, v):
    euclidean_dists = np.linalg.norm(u - v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    poincare_dists = np.arccosh(
        1 + 2 * (
            (euclidean_dists ** 2) / ((1 - u_norm ** 2) * (1 - v_norm ** 2))
        )
    )
    return poincare_dists

model = HypConesModel(transitive_relations,
                      dim=params['size'],
                      init_range=(-0.1, 0.1),
                      lr=0.0001,
                      seed=params['seed'],
                      logger=logger,
                      num_negative=5,
                      ### How to sample negatives for an edge (u,v)
                       neg_sampl_strategy='true_neg',
                      # 'all', 'true_neg' , 'all_non_leaves' or 'true_neg_non_leaves'
                       where_not_to_sample='ancestors',
                      # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                       neg_edges_attach='child',
                      # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                       neg_sampling_power=0,
                      # 0 for uniform, 1 for unigram, 0.75 for word2vec
                       margin=0.01,  # Margin for the loss.
                       opt=params['opt'],
                       K=0.1,
                       epsilon=1e-4,
                      cvpr_loss='sim'
                       )

vecs = model.kv.syn0
model.kv.syn0 = model._clip_vectors(vecs * 0.8)

print('Finished initialization. Now training the hyperbolic cones..')

# Train the model
for i in range(int(params['epochs'] / params['print_every'])):
    print('Starting epoch ' + str(params['print_every'] * i))
    # Train
    if i < 1:
        # No training here, just to plot the initial state.
        avg_loss, avg_pos_loss, avg_neg_loss = \
            model.train(epochs=0, batch_size=params['batch_size'], print_every=params['print_every'])
    else:
        avg_loss, avg_pos_loss, avg_neg_loss = \
            model.train(epochs=params['print_every'], batch_size=params['batch_size'], print_every=params['print_every'])

    # Animation
    if params['size'] == 2:
        figure = poincare_2d_visualization(
            model,
            animation=animation,
            epoch=(params['epochs'] + params['print_every'] * (i+1)),
            eval_result='',
            avg_loss=avg_loss,
            avg_pos_loss=avg_pos_loss,
            avg_neg_loss=avg_neg_loss,
            tree=list(tree_relations),
            show_node_labels=show_node_labels,
            figure_title=figure_name,
            num_nodes=None)

if params['size'] == 2:
    plot(animation)

word2vec = {}
for k in label2idx.keys():
    if k in model.kv:
        word2vec[k] = model.kv[k].astype(float)
        
if 'sad' in word2vec:
    # little buggy in ED dataset
    word2vec['sad'].astype(float)
pickle.dump(word2vec, open(os.path.join((os.path.abspath('..'), "label_tree", args.tree + ".bin") , 'wb')))

# ### If one wants to save and reload a model later:
#     pickle.dump(model.kv, open(args.root_path + "model_" + str(dim) + "D.bin", 'wb'))
#
#     model = PoincareModel(train_data=transitive_relations, dim=dim, logger=logger)
#     model.kv = pickle.load(open(args.root_path + "model_" + str(dim) + "D.bin", 'rb'))
#
#     if dim == 2:
#         figure = poincare_2d_visualization(
#             model,
#             animation=animation,
#             epoch=(0),
#             eval_result='',
#             avg_loss=0,
#             avg_pos_loss=0,
#             avg_neg_loss=0,
#             tree=list(),
#             show_node_labels=show_node_labels,
#             figure_title=figure_name,
#             num_nodes=None)
#     plot(animation)



```
## 原始树
```tsv
semi-positive	root
ambiguous	root
semi-negative	root
admiration	semi-positive
amusement	semi-positive
caring	semi-positive
gratitude	semi-positive
desire	semi-positive
approval	semi-positive
relief	semi-positive
positive	semi-positive
admiration	root
amusement	root
caring	root
gratitude	root
desire	root
approval	root
relief	root
positive	root
excitement	positive
joy	positive
love	positive
optimism	positive
pride	positive
excitement	semi-positive
joy	semi-positive
love	semi-positive
optimism	semi-positive
pride	semi-positive
excitement	root
joy	root
love	root
optimism	root
pride	root
surprise	ambiguous
surprise	root
curiosity	ambiguous
curiosity	root
realization	ambiguous
realization	root
confusion	ambiguous
confusion	root
disappointment	semi-negative
disapproval	semi-negative
nervousment	semi-negative
embarrassment	semi-negative
negative	semi-negative
nervousness	semi-negative
disappointment	root
disapproval	root
nervousment	root
embarrassment	root
negative	root
nervousness	root
disgust	negative
fear	negative
grief	negative
remorse	negative
sadness	negative
anger	negative
annoyance	negative
disgust	semi-negative
fear	semi-negative
grief	semi-negative
remorse	semi-negative
sadness	semi-negative
anger	semi-negative
annoyance	semi-negative
disgust	root
fear	root
grief	root
remorse	root
sadness	root
anger	root
annoyance	root
```
## config 与球边界实现
```python
import os
import pickle
import argparse
from utils.train_utils import add_flags_from_config

all_dataset_list = ['go_emotion', 'ED', 'ED_easy_4', 'ED_hard_a', 'ED_hard_b', 'ED_hard_c', 'ED_hard_d']
ENCODER_TYPE = 'roberta-base'
temperature = 0.3

config_args = {
    'training_config': {
        'batch_size': (100, 'batch size for training'),
        'epochs': (20, 'maximum number of epochs to train for'),
        'seed': (1234, 'seed for training'),
        'alpha': (0.9, 'weight for BCE*poincare_dist loss'),
        'gamma': (0.1, 'weight for poincare loss')
    },
    'poincare_model_config': {
        'dim': (100, 'hyperbolic label embedding dimension'),
        'feat_dim': (768, 'hidden size of text embedding'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'dropout': (0.1, 'dropout probability'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'a': (0.2, 'alpha for leakyrelu in graph attention networks'),
    },
    'data_config': {
        'dataset': ('go_emotion', 'which dataset to use'),
    }
}

def get_dicts(dataset, return_emb_dicts=False):
    assert dataset in all_dataset_list
    if dataset == 'ED':
        label2idx = {'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
             'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
             'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13, 
             'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
             'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
             'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
             'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31}
        
    elif dataset == 'go_emotion':
        label2idx = {'admiration': 0, 'amusement': 1, 'anger': 2,
              'annoyance': 3, 'approval': 4, 'caring': 5,
              'confusion': 6, 'curiosity': 7, 'desire': 8,
              'disappointment': 9, 'disapproval': 10, 'disgust': 11,
              'embarrassment': 12, 'excitement': 13, 'fear': 14,
              'gratitude': 15, 'grief': 16, 'joy': 17,
              'love': 18, 'nervousness': 19, 'optimism': 20,
              'pride': 21, 'realization': 22, 'relief': 23,
              'remorse': 24, 'sadness': 25, 'surprise': 26}
    
    elif dataset == 'ED_easy_4':
        label2idx = {'sad': 0, 'joyful': 1, 'angry': 2, 'afraid': 3}
    
    elif dataset == 'ED_hard_a':
        label2idx = {'anxious': 0, 'apprehensive': 1, 'afraid': 2, 'terrified': 3}
    
    elif dataset == 'ED_hard_b':
        label2idx = {'sad': 0, 'devastated': 1, 'sentimental': 2, 'nostalgic': 3}
    
    elif dataset == 'ED_hard_c':
        label2idx = {'angry': 0, 'ashamed': 1, 'furious': 2, 'guilty': 3}
    
    elif dataset == 'ED_hard_d':
        label2idx = {'anticipating': 0, 'excited': 1, 'hopeful': 2, 'guilty': 3}

    idx2label = {v: k for k, v in label2idx.items()}
    
    if return_emb_dicts:
        word2vec = pickle.load(open(os.path.join((os.path.abspath('')), 'label_tree', f'{dataset}.bin'), 'rb'))
        idx2vec = {k: word2vec[v] for k, v in idx2label.items()}
        return (label2idx, idx2label), (word2vec, idx2vec)
        
    return label2idx, idx2label
    
d, _ = config_args['data_config']['dataset']
label_dicts, emb_dicts = get_dicts(d, return_emb_dicts=True)
parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)

```
```python
"""Poincare ball manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)


```
