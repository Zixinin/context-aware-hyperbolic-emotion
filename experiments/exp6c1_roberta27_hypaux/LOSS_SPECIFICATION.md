# Exp6C-1 A1：Formula A auxiliary-only，训练前冻结

## 研究边界与明确选择

本轮唯一计划新增变量：在已经完成的 A0 27-label 多标签 BCE baseline 上，加原 HypEmo HNN 投影、原 tangent decoder、原固定27情绪向量及 Exp6B 已验收 Formula A 的多正标签辅助监督。新增分支含原 HNN/decoder 的参数和原 dropout；分类路径完全复用 A0。

Exp6B **总目标不能原样复用**：它同时包含 detached 距离权重乘以 BCE。本轮只复用已验收辅助项，普通 BCE 不乘 alpha=0.9 或任何样本/类别/距离权重。辅助系数 gamma=0.1 沿用用户指定此前适配设置。它不是重新校准后的最佳系数，也没有假定跨协议相对梯度尺度自动相同。

## 精确数学公式

batch大小 B；共享RoBERTa末层首token表示 h_i；A0原分类路径输出 g_i∈R^27；multi-hot y_i∈{0,1}^27；P_i={k:y_ik=1}，m_i=|P_i|≥1。

本次所有样本均有至少一个非neutral正标签，所以 Exp6B 的有情绪 mask M_i 恒为1，不再需要 neutral-only 分支。neutral+emotion 来源样本仍保留，移除 neutral 后只对其剩余正情绪求平均。

```math
q_{ik}=y_{ik}/m_i
z_i=HNN_{c=1}(h_i) \in \mathbb{B}^{100},\quad
u_i=Decoder_{tangent}(z_i) \in \mathbb{R}^{27}
H_i=-\sum_{k=0}^{26}q_{ik}\log softmax(u_i)_k
D_i=\sum_{k=0}^{26}q_{ik}\langle z_i,e_k\rangle
L_{hyp\_aux}=\frac1B\sum_{i=1}^{B}(H_i-D_i)
L_{BCE}=\frac1{27B}\sum_{i=1}^{B}\sum_{k=0}^{26}
    BCEWithLogits(g_{ik},y_{ik})
L_{A1}=L_{BCE}+0.1 L_{hyp\_aux}
```

e_k 为原27个100维固定向量，按 A0 canonical label names 对齐；非训练参数，保留原值、dtype、维数与球内性质。不新增 neutral 向量。

H_i 是对多正标签均匀分布的 cross-entropy，不是从正标签中选 argmax，也不是给辅助 decoder 独立 sigmoid。**正式分类仍是27个独立 sigmoid**，阈值0.30，辅助decoder不用于最终分类预测。

点积是原 Formula A 所用球坐标的普通内积；不擅自换成双曲距离、cosine、温度缩放、对比损失或其他对齐目标。HNN、球投影与 logmap0 tangent decoder 直接导入已冻结原作者源码。

## 归约与尺度

- BCE 使用标准 `binary_cross_entropy_with_logits(..., reduction='mean')`，等价标准 `BCEWithLogitsLoss`，在 B×27 元素上平均。
- 每样本 H、D 对其全部真实正标签按 1/m_i 平均，再在整个 batch 上平均；不是逐batch求和，也不是只挑一个正标签。
- 最后不足16条的batch使用其实际B；训练梯度仍按A0 `(mean_total_loss*B).backward; gradients/=B; clip_all_norm1; AdamW; scheduler`。
- 不按正标签数、batch大小、距离、类频或 rare-label 支持度另行加权。
- raw aux 可能因 `-D_i` 为负；不 clamp、不取绝对值、不加常数、不动态归一化。若非有限值或出球，停止并报告，不修公式继续跑。
- 同原有 ball 表示，理论上 |D_i|<1；H_i≥0，但这个界不能保证辅助项相对于 BCE 的有效梯度尺度小。
- gamma0.1 的原始含义是显式辅助系数；不能将它解释为辅助贡献固定占总损失10%。实际比例由当前loss值决定。

训练前 smoke 必须报告普通 BCE、raw aux、0.1×aux、总 loss、带符号与绝对的 weighted_aux/BCE 比例、H/D均值以及共享encoder上的 BCE 与 weighted-aux 梯度范数及其比例。逐batch和逐epoch持续记录实际数值；不会根据数值或Test结果自动调系数。

## 梯度路径

```text
BCE -> A0 classifier/dropout -> shared RoBERTa encoder
H_i -> auxiliary decoder -> HNN projection -> shared RoBERTa encoder
-D_i -> HNN projection -> shared RoBERTa encoder
fixed label vectors e_k: no gradient / no optimizer state
auxiliary -> classifier weights/bias: no direct path
BCE -> HNN/auxiliary decoder: no path
```

aux中的 z 与 u 不 detach；q 是固定数据目标。原 Exp6B detached 距离权重不参与本轮实际训练目标，alpha0.9不使用。

smoke 会把新辅助项与已冻结 Exp6B `approved_multilabel_loss.loss_components(...)["auxiliary"]` 比较数值及对z/u的梯度，要求精确一致。为调用旧函数，仅在此对照诊断临时追加全0 neutral列；它不是训练数据，也不创建neutral分类参数。旧函数计算的 weighted classification/total 被丢弃，不反传。本轮训练只反传上述 A1 总目标。

## 与A0严格匹配及初始化

直接调用原 `model27.load_model` 和 `Roberta27.forward`，包括首token、原独立标签dropout、权重与bias及历史初始化消耗约定。不改 A0 源码；只在新实例的encoder上使用一次性输出hook，把同一次forward的 `last_hidden_state[:,0,:]` 传入辅助分支。不会多跑一次encoder或增添pooler/dense/tanh。

先构建共享A0模型，再在 `torch.random.fork_rng(devices=[0])` 中导入/构建原HNN并恢复Python/NumPy状态；构造前后共享torch CPU/CUDA状态一致。额外分支初始化不能改变共享权重、分类头或共享初始化后的RNG。

历史A0没有单独命名initial checkpoint；但A0 smoke使用warmup首步lr=0并保存了其state。本轮不假定该事实足够：实际逐张量比较新seed42原始A0模型与历史smoke_weights.pt，必须精确相同。随后证明A1共享state相同，初始train/eval分类logits逐元素完全相同。**不加载A0最佳已训练checkpoint作为A1初始化**，也不使用A1 smoke更新后的权重正式训练。

每轮first-batch IDs与历史A0的对应文件精确对照；shuffle DataLoader直接复用A0函数及独立torch.Generator(42)。HNN保留原dropout，因此训练期间辅助分支会多消耗随机数，后续共享dropout掩码轨迹不承诺与A0完全相同。此事实必须随结果报告，不声称几何本身已被单独因果隔离。

## 原双曲模块与向量来源

- 原作者commit：`97ec6af6b19b831f69cb35a82dc65ea1395fa157`。
- `FGTCModel`只实例化HNN，`encode_graph=False`；原100维、c=1、num_layers=2、relu、bias=1、dropout0.1与tangent 27-way decoder保持原样。
- 原 `label_tree/go_emotion.bin` SHA-256：`2695b98e160e79c62533d6bee55fdbcfdbbcba87cbcde8ab6db2734dc16ab20a`。
- 与已验收 Exp6B 原始向量矩阵逐值匹配；其 `.npy` SHA-256：`648b6ecec551df27b02b70d378f59e0a879d662ae0dd9413c857fbc89030a1db`。
- 原源码就地只读导入，不复制第三方源码到FYP。

原始实现可定位：[HNN及ball输入映射](https://github.com/dinobby/HypEmo/blob/97ec6af6b19b831f69cb35a82dc65ea1395fa157/models/encoders.py)、[tangent decoder](https://github.com/dinobby/HypEmo/blob/97ec6af6b19b831f69cb35a82dc65ea1395fa157/models/decoders.py)、[原辅助项](https://github.com/dinobby/HypEmo/blob/97ec6af6b19b831f69cb35a82dc65ea1395fa157/models/base_models.py)。本轮多正标签/整批均值采用此前已批准Formula A，不声称原单标签loss归约完全不变。

## 停止与验收规则

A0自身旧global protection inventory早于A1，授权新增A1文件会使旧“全仓库不允许新增”的检查产生差异。本轮不改该历史清单：用A0封存artifact inventory、A0源码固定hash和完整保存预测重算确认旧A0，再用新的A1保护清单完整包含A0代码、checkpoint和结果。A1新增文件/缓存只写独立目录。

任何来源/公式/归约/标签身份/运行环境/初始state/logit一致性问题均停止，不自行换公式、改球投影或启动正式训练。若本轮smoke通过，先返回实际loss比例/梯度报告验收，再明确启动seed42四轮训练。

正式选择只看validation Macro-F1@0.30，完全相同最高分取更早epoch。最佳checkpoint在全部3834条验证样本重载一致后才做一次正式Test；之后只读完整预测重算指标并与A0同ID配对比较27类。失败不自动重训。完成后停止，不增加seed或其他方法。
