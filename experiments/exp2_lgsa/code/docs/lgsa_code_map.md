# LGSA 论文到代码：冻结版本逐项定位

实验：FYP 第二个实验 LGSA-GoEmotions Adaptation。此文记录理解阶段；没有宣称官方模型已在 GPU 上通过。

所有行号相对 `original_lgsa/`，commit `4ea3dfb39dcee82654f443811471dd29be32e87e`。
论文：[ACL 2026 正式 PDF](https://aclanthology.org/2026.acl-long.1322.pdf)，§3.3–3.5，公式 (1)–(15)。

## 1. 张量和术语

`B`=batch，`T`=序列长度（官方入口固定 512），`H`=hidden size（BERT base 为 768），
`C`=所有被监督的层级标签数，`D`=层数，`V`=原词表大小，`E`=图边数。
ROOT 是逻辑根，通常不占类别 ID。注意代码中的 `depth2label[0]` 指顶层标签，不是 ROOT。

## 2. 完整调用链

```text
train.py:23 import HierarchicalGofLoss
train.py:97–131 value_dict.pt + slot.pt -> value2slot/depth2label/path_list
train.py:144–184 JSONL -> data_map_function -> [B,512] tokens + [B,D*C] targets
train.py:190 import Prompt
train.py:202–204 Prompt.from_pretrained -> Prompt.__init__ -> init_embedding
  prompt.py:206–214 标签名 token 的 BERT lookup 向量求平均 -> [C,H]
  prompt.py:217–221 建 HGS parent map -> semantically_anchored_gof_initialization
    prompt.py:128–135 _gram_schmidt
    prompt.py:141–148 顶层正交化并恢复范数
    prompt.py:150–182 按深度遍历，分解 parent projection / child residual
  prompt.py:235–244 GraphEmbedding 替换输入层，OutputEmbedding 替换输出层
    graph.py:211–214 GraphEncoder -> GraphLayer -> GATConv
train.py:210–216 HierarchicalGofLoss 实例化（用 train.py 的另一份 parent map）
train.py:263 model(**batch) -> Prompt.forward
  prompt.py:298–330 BERT -> MLM head -> MLM loss + multi-label loss
train.py:265–268 再调用 GraphEmbedding.weight() -> [C,H] -> HGA
train.py:279–286 lambda warmup -> total loss -> backward
train.py:314 / prompt.py:356–376 generate -> 每个深度取对应类别，logit>0
train.py:338–349 best_macro / best_micro / last checkpoint
```

## 3. 十五项组件对应表

| 要求 | 文件、符号、行号 | 输入 → 输出 / shape | 数学作用及实际行为 |
|---|---|---|---|
| Semantic initialization | `models/prompt.py` `Prompt.init_embedding` 186–214 | 标签名 → token IDs → `[C,H]` | BERT **输入 embedding lookup** 均值，非标签句子的 contextual BERT pooled output；`tokenizer.encode` 默认含 special tokens。描述初始化是可选分支，默认不用。 |
| Global orthogonalization | 同文件 `semantically_anchored_gof_initialization` 137–148 | 顶层 `[K,H]` → `[K,H]`（满秩时） | Gram–Schmidt 得单位方向后乘原始范数。 |
| Gram–Schmidt | 同文件 `_gram_schmidt` 128–135 | `[K,H]` → `[rank,H]` | 顺序相关，低于 `1e-8` 的残差跳过；空 basis 会报错，没有完整退化处理。 |
| Recursive local orthogonalization | 同文件 150–184 | `[C,H]`、depth map、child→parent map → `[C,H]` | 按层遍历；不是递归函数调用。只处理 child 数>1 的家庭。 |
| Parent projection | 同文件 167–169、178–179 | child `[H]`、parent `[H]` → `[H]` | `(c·p)/(p·p)*p`；HGS 处无零范数保护。 |
| Child residual | 同文件 169–182 | `[K,H]` → 正交残差再加投影 `[K,H]` | 保留原残差长度；不能说最终 child 向量互相正交。 |
| GAT refinement | `models/graph.py` `GraphLayer` 152–207，`GraphEncoder` 210–334；`prompt.py` 38–48 | 标签+depth prompt `[C+D,H]`，edge_index `[2,E]` → 同形状 | 默认单层单头 GATConv + dropout + residual + LayerNorm。图含 parent→child 及 depth prompt→该层标签边；并非只包含分类标签。没有手动补反向边。 |
| HGS | `Prompt.init_embedding` 186–254 + 上述函数 | `[C,H]` initial → structured → embedding subsystem | 结构初始化一次；GAT 在调用 `weight()` 时动态运行，训练期 dropout 生效。 |
| HGA | `models/loss.py` `HierarchicalGofLoss` 20–133；`train.py` 210–216、265–268 | 当前 GAT label rows `[C,H]` → scalar + component dict | 正则化当前 label 几何关系；没有启用与固定 prototype 逐点距离对齐。 |
| Global loss Lg | `HierarchicalGofLoss.forward` 65–79 | 顶层 `[K,H]` → scalar | 不同顶层的 cosine² 平均。 |
| Parent-child loss Lpc | 同文件 81–90 | 所有 parent/child `[P,H]` → scalar | `(cos(c,p)-cos(theta))²` 平均，目标是角度值，不是只处罚超过角度上限。 |
| Norm loss Ln | 同文件 92–97 | 范数 `[P]` → scalar | `mean(relu(gamma*norm(parent)-norm(child)))`；下界约束，非强制 child 小于 parent。 |
| Sibling residual loss Ls | 同文件 99–131 | 每家庭 residual `[K,H]` → scalar | 残差 cosine 绝对值先按 pairs 平均，再按家庭平均；诊断字典 `brother_res` 实际只留下最后一个家庭，需修日志。 |
| Classification loss | `models/loss.py` 6–18；`Prompt.forward` 319–330 | `[B*D,C]` logits、带 -100 targets → scalar | multi-label categorical cross-entropy（logsumexp 形式），**不是 BCE**，更不是单标签 softmax CE。MLM 是另一项 CE。 |
| Total loss | `train.py` 279–286 | 三类 scalar → scalar | `Lmlm + Lcls + alpha(t)*gof_loss_weight*(Lg+Lpc+Ln+Ls)`；前10% steps 线性增加 alpha。四项默认权重1。`align_w` 没有接入执行中的 total。 |

## 4. Dataset / hierarchy contract

| 文件 / 对象 | 格式 | 使用位置 |
|---|---|---|
| `value_dict.pt` | `dict[int,str]`，ID 连续 `0..C-1` | train 97；prompt 206；graph 219 |
| `slot.pt` | `dict[parent_id, iterable[child_id]]` | train 100–114 |
| `DATA_train.json`, `DATA_dev.json`, `DATA_test.json` | JSONL，每行 `{"token": "text", "label": [id,...]}` | train 144–147 |
| `value2slot` | `child -> parent`；顶层 `-1` | train 101–127；HGA |
| `depth2label` | `depth -> list[label_id]` | prompt construction / loss mask / generate |
| `path_list` | `(parent,child)` 边；附加 `(C+depth,label)` | train 111、129–131；GAT |
| batch labels | `[B,D*C]`，对应层0/1，其余 -100 | train 156–177 |
| taxonomy 文本 | 每行 parent TAB children | Prompt.__init__ 98–120；WoS 文件名有拼写不一致，详见问题清单 |

输入 JSONL 必须已经包含需要监督的祖先标签。`train.py` **不会**为每个细标签自动增加祖先。
`value2slot` 字典只能保留一个 parent；多父 DAG 会被覆盖。数据进入模型前必须查环、重复 child、多父、缺失 ID、未覆盖标签。

## 5. 论文与当前公开代码必须分开解释

1. **方向反转已用原始 AST 复现**：`train.py:111` 的 `(parent,child)` 在 `prompt.py:217` 被反向读取。两层 toy 中细标签完全没有经过预期的 sibling residual 正交化。
2. **prototype 梯度路径**：`prompt.py:225` 将 `ideal_label_embeddings` 注册为 Parameter，但 235–240 的拼接随后在 `GraphEmbedding:33` 被 `Embedding.from_pretrained(..., True, 0)` 冻结成另一个参数。玩具实验确认该构造切断 ideal Parameter 梯度；不能因变量名就说它是被训练的固定目标。
3. **输入/输出 weight 不同**：输入走 `weight()`（GAT 后），输出 `OutputEmbedding.weight` 绑定 `raw_weight`（GAT 前），见 239–244。论文 §3.5 的共享表述不能原封不动套到这段代码。
4. **几何正则来自第二次 GAT 调用**：train 263 forward 后 266 再调用 `weight()`；两次 dropout 样本可能不同。
5. **没有附加 hyperbolic 模块**。`HypergraphConv` 是代码可选图分支，不等于 hyperbolic；本实验默认 GAT。

完整修复分类与下一阶段验证见 [upstream_findings.md](upstream_findings.md)。
