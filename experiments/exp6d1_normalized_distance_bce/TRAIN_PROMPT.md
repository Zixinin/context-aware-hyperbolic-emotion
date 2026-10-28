# Exp6D-1 — Normalized Hyperbolic Distance-weighted BCE Pilot

## 0. 实验背景与目标

当前 FYP 为：

**Context-aware Multi-label Emotion Recognition with Hyperbolic Emotion Label Embeddings**

已有实验：

- Exp6C-0（A0）：RoBERTa27 + BCE。
- Exp6C-1（A1）：RoBERTa27 + BCE + Hyperbolic Auxiliary。
- Exp6C-1S（A1-S）：在 A1 的双曲辅助入口增加固定缩放 `s=0.03`。
- Exp6D Audit：完整核对原版 HypEmo 与 Exp6C 的代码、Loss 和训练协议，并完成距离加权的只读数值与梯度审计。

Exp6D 审计发现，直接采用原始双曲距离作为 BCE 权重，会将分类梯度整体放大约十倍，而归一化 N1 可以控制这一尺度。

**本轮正式执行 Exp6D-1，只增加一个机制：Normalized Hyperbolic Distance-weighted BCE（N1）。**

研究问题：

> 在保持 Hyperbolic Auxiliary、RoBERTa 和数据协议不变的条件下，加入基于文本—正情绪标签双曲距离的归一化样本加权，能否改善多标签情绪识别？

本轮是单 seed、Validation-only 探索实验，不预设一定提升。

---

## 1. 实验位置与保护要求

FYP repository：

`/home/zixinfeng/Documents/context-aware-hyperbolic-emotion`

主要匹配对照：

`results/exp6c1s_scaled_train/run_20261022T035554Z_cZrAxW97`

其他冻结实验：

- Exp6C-0 A0
- Exp6C-1 A1
- Exp6D Audit
- Exp6A / Exp6B 及所有历史实验

新目录：

```text
experiments/exp6d1_normalized_distance_bce/
results/exp6d1_normalized_distance_bce/run_<timestamp>/
```

严格禁止修改任何已有代码、checkpoint、预测结果或实验报告。

不修改 CUDA、驱动、系统 Python 或已验证环境依赖。

不自动 Git push、merge、reset、clean。

正式执行前后均核查冻结文件保护清单与哈希。

---

## 2. 唯一允许的模型修改

以 A1-S 为基础。

保留：

- 原有 RoBERTa-base snapshot/tokenizer。
- 原有 27-label 分类头。
- 原有 Hyperbolic Projection / HNN / Auxiliary Decoder。
- 固定的 27 个 Parrott 预训练标签向量。
- 双曲输入缩放 `s=0.03`。
- 原始 Multi-positive Hyperbolic Auxiliary Loss。
- 辅助系数 `gamma=0.1`。

唯一修改：

**将普通 BCE 改为按正标签双曲距离进行 N1 样本加权的 BCE。**

### 数学定义

对于一个 batch 中第 i 条文本：

\[
P_i=\{k:y_{ik}=1\}
\]

计算所有真实正情绪标签的双曲距离：

\[
d_i=\frac{1}{|P_i|}\sum_{k\in P_i}d_{\mathbb H}(z_i,e_k)
\]

其中：

- \(z_i\) 是当前文本的双曲投影。
- \(e_k\) 是固定的 Parrott 预训练情绪标签向量。
- 多个正标签必须全部参与距离平均。

对当前 batch 进行尺度归一化：

\[
w_i=\operatorname{stopgrad}
\left(\frac{d_i}{\frac1B\sum_{j=1}^{B}d_j}\right)
\]

每条样本的分类 BCE：

\[
\ell_i=\frac1{27}\sum_{k=1}^{27}
\mathrm{BCEWithLogits}(g_{ik},y_{ik})
\]

归一化距离加权 BCE：

\[
L_{\mathrm{N1}}=\frac1B\sum_iw_i\ell_i
\]

最终损失：

\[
\boxed{L=L_{\mathrm{N1}}+0.1L_{\mathrm{aux}}}
\]

必须核实公式与 Exp6D 审计通过的 N1 设计完全一致。

重要要求：

1. 距离必须是真实 Poincaré Distance，不是普通 dot product。
2. 权重必须 detached，不允许通过权重分支反向传播。
3. 对 27 个输出的 BCE 先按标签维度求平均，再按样本权重聚合。
4. 同一个样本中的全部正标签必须参与距离聚合。
5. 权重按当前 minibatch 的距离均值归一化；包括最后一个不足 batch16 的 minibatch。
6. 不额外引入 `alpha=0.9`，因为本实验要保持 A1-S 的 BCE 系数不变。
7. 不增加 distance clipping、temperature、rare weighting、ASL 或其他正则化。
8. 不改变辅助损失归约方式。

如果存在与既有 N1 数值审计不一致的实现，停止并报告。

---

## 3. 固定实验协议

| Setting | Value |
|---|---|
| Backbone | RoBERTa-base |
| Dataset | GoEmotions simplified |
| Task | 27-label Multi-label |
| Neutral | 排除 neutral-only，不输出 neutral |
| Train / Validation / Test | 30587 / 3834 / 3821 |
| Seed | 42 |
| Epochs | 4 |
| Batch Size | 16 |
| Max Length | 50 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Scheduler | Linear decay |
| Warmup Steps | 765 |
| Total Training Steps | 7648 |
| Gradient Clip | 1.0 |
| Hyperbolic Input Scale | 0.03 |
| Auxiliary Coefficient | 0.1 |
| Fixed Threshold | 0.30 |
| Best Checkpoint | Validation Macro-F1 |

必须从与 A1-S 相同的预训练模型及初始化状态开始独立训练。

**禁止从 A1-S 已训练 checkpoint 继续训练。**

核查公共 encoder、classifier、HNN/decoder 的初始参数与 A1-S 一致，分类初始 logits 一致。

---

## 4. 训练前 Smoke Test

正式训练前完成以下验收。

### 4.1 Loss 正确性

检查：

- 27 维 multi-hot label 正确。
- Single-positive 和 Multi-positive 样本都正确计算。
- 每条样本至少有一个正情绪标签。
- 距离矩阵和标签映射一致。
- 权重无 NaN、Inf。
- 当前 batch 权重均值约为 1。
- N1 与 Exp6D 审计中的独立参考公式数值一致。

使用固定训练样本检查：

```text
ordinary_bce
mean_positive_hyperbolic_distance
normalized_weight
weighted_bce
auxiliary_loss
total_loss
```

### 4.2 梯度正确性

分离计算：

```text
BCE gradient
N1 weighted BCE gradient
0.1 auxiliary gradient
combined gradient
```

检查：

- 共享 RoBERTa Encoder 梯度有限。
- 分类头梯度有限。
- HNN / Auxiliary Decoder 梯度有限。
- 标签向量仍然固定。
- Detached weights 不产生错误的梯度路径。
- N1 不会意外将 BCE 分类梯度整体扩大十倍。

保持 A1-S 相同诊断条件，对照 Exp6D 的数值审计。

如果出现非有限值、映射不正确、异常尺度放大，立即停止正式训练。

---

## 5. 正式训练

仅训练：

**Exp6D-1 N1，seed42，4 epochs。**

每个 epoch 保存：

```text
training_bce
training_weighted_bce
training_auxiliary
training_total_loss
validation_macro_f1
validation_micro_f1
validation_rare_f1
validation_macro_ap
learning_rate
gradient_norm
gradient_clip_fraction
```

同时新增 N1 权重诊断：

```text
distance_mean
distance_std
distance_median
distance_p95
distance_max

weight_mean
weight_std
weight_min
weight_p95
weight_max

weighted_bce / ordinary_bce
weight_vs_bce_correlation
```

记录 Single-positive 和 Multi-positive 样本的权重分布差异。

优先从训练 batch 记录这些统计，不使用 Test 来判断调整方法。

继续复用 Exp6C-1S 的固定 train-only 几何探针，记录每轮：

- 首次 expmap clip rate。
- 后续 HNN clip rate。
- 最终球半径分布。
- 范数大于 0.99 的比例。
- 加权 BCE 与辅助 Loss 的梯度范数及方向。

训练过程中如果再次发生球边界饱和，只记录，不自动更改缩放。

若出现明显异常或训练失败，保留日志并停止，不自行尝试另一种权重公式。

---

## 6. Validation 评估与模型选择

每轮使用完整 Validation split 进行评价。

固定：

`sigmoid(logits) >= 0.30`

以最高 Validation Macro-F1 选择 checkpoint，平局选最早 epoch。

保存最佳 checkpoint，并重新加载验证全部 3834 条样本。

必须核实：

- Canonical IDs 一致。
- Gold labels 一致。
- Logits 重载一致。
- 阈值预测一致。
- Aggregate/per-label 指标重算一致。

**本轮不执行新的 Test 推理。**

不得依据已知 Test 表现调整权重设计。

---

## 7. 必须完成的对比

比较：

| Model | Description |
|---|---|
| A0 | RoBERTa27 + ordinary BCE |
| A1 | RoBERTa27 + BCE + Hyperbolic Auxiliary |
| A1-S | A1 + input scaling s=0.03 |
| Exp6D-1 | A1-S + N1 distance-weighted BCE |

已有 Validation 结果：

| Metric | A0 | A1 | A1-S |
|---|---:|---:|---:|
| Micro-F1 | 63.4011% | 63.9513% | 63.9839% |
| Macro-F1 | 58.1356% | 56.0209% | 56.8995% |
| Rare-F1 | 55.0248% | 44.7456% | 48.8913% |
| Macro AP | 61.3012% | 59.5187% | 59.7749% |

所有历史值必须重新读取并核实封存结果，不能仅依赖本 Prompt 的小数。

本轮主要匹配对照是：

**Exp6D-1 vs A1-S。**

A0、A1 只作为历史参考。

正式输出：

- Micro Precision / Recall / F1。
- Macro Precision / Recall / F1。
- Weighted-F1。
- Rare-F1。
- Macro Average Precision。
- 27 类 Per-label F1/AP/TP/FP/FN。
- 七个 Rare 标签详细结果。

注意分析 grief、relief、pride 的实际变化，但不得因已知历史表现而专门给它们调整阈值、Loss 或训练参数。

---

## 8. 结果解释标准

### 若 N1 的 Macro-F1 优于 A1-S

可以记录：

“归一化双曲距离加权在当前固定实验设置下改善了 A1-S 的 Validation Macro-F1。”

但还要检查 Rare-F1、Macro AP 是否改善，以及是否仅由个别低支持标签主导。

### 若 N1 超过 A0

可以记录：

“在当前 seed42、27-label Multi-label Validation 协议下，N1 取得初步性能优势。”

不能称为已证明双曲几何具有独立价值。

### 若 N1 没有改善

如实记录负结果。

不得继续自动尝试 raw F、其他归一化、gamma、新缩放值或新随机种子。

### 若 N1 有明显数值异常

停止训练，保留数据并说明原因。

不得自行修改公式后继续，以免一个实验同时包含多个未经批准的改动。

---

## 9. 交付文件

生成独立报告：

```text
EXP6D1_IMPLEMENTATION_AUDIT.md
EXP6D1_N1_LOSS_VERIFICATION.md
EXP6D1_TRAINING_SUMMARY.md
EXP6D1_VALIDATION_COMPARISON.csv
EXP6D1_PER_LABEL_COMPARISON.csv
EXP6D1_DISTANCE_WEIGHT_DIAGNOSTICS.csv
EXP6D1_GEOMETRY_REPORT.csv
EXP6D1_VERIFICATION_REPORT.md
```

保存：

```text
config.json
environment.json
source_manifest.json
training_log.csv
checkpoint_selection.json
best_checkpoint/
validation_predictions.npz
validation_metrics.json
validation_per_label.csv
rare_label_metrics.json
```

最后提供一份简单中文总结，回答：

1. N1 是否正确实现？
2. N1 是否仍然存在约十倍的整体分类梯度放大？
3. 加入 N1 后 Validation Macro/Micro/Rare-F1 如何变化？
4. 是否超过 A1-S？是否超过 A0？
5. 七个 Rare 标签有无恢复？
6. Macro AP 是否改善？
7. 权重分布是否具有足够的样本差异？
8. 训练过程中双曲边界饱和是否仍然存在？
9. 目前能否说明距离加权对 Multi-label 任务有帮助？
10. 后续还缺什么证据才能认定 Hyperbolic 具有独立价值？

---

## 10. 严格停止规则

本轮只执行：

**Exp6D-1 — A1-S + N1 Distance-weighted BCE，seed42。**

禁止自动：

- 启动 Raw F 实验。
- 训练其他 seed。
- 调整 gamma。
- 调整输入缩放。
- 修改标签层级。
- 加入 Rare weighting。
- 引入 Contrastive Loss、ASL、HyCoEM。
- 重新训练 A0/A1/A1-S。
- 运行新的正式 Test。
- 搜索更多参数组合。

完成后输出：

```text
==================================================
EXP6D1_N1_SEED42_VALIDATION_PILOT_COMPLETE
Task: RoBERTa27 Multi-label
Control: Exp6C-1S
Distance weighting: N1 normalized
Hyperbolic input scale: 0.03
Auxiliary coefficient: 0.1
Seed: 42
Checkpoint: Validation Macro-F1
Formal threshold: 0.30
New Test inference: NONE
==================================================
```

如果任何必要验收失败，明确报告失败状态，不打印成功标记。

**Research principle: Restore one core HypEmo mechanism at a time, preserve matched comparisons, and do not interpret a single-seed validation improvement as proof of hyperbolic superiority.**