# Codex Research Skill — Exp6C-1D: Hyperbolic Mechanism Diagnosis

## 0. 任务背景

当前 FYP：

**Context-aware Multi-label Emotion Recognition with Hyperbolic Emotion Label Embeddings**

研究目标：探索双曲情绪标签表示是否能够改善细粒度多标签情绪识别，特别关注 Macro-F1、Rare-F1 和相似情绪之间的区分。

目前已完成：

- Exp6A：Original HypEmo 单标签复现。
- Exp6B：28-label Multi-label HypEmo Formula A。
- Exp6C-0（A0）：RoBERTa + 27-label Multi-label BCE。
- Exp6C-1（A1）：在 A0 上增加 HypEmo-inspired Hyperbolic Auxiliary Loss。

**本周的核心任务不是重新设计模型，而是诊断 A1 为什么没有提升 Macro-F1，以及当前 Hyperbolic 方法是否还有合理的改进空间。**

本次实验命名：

**Exp6C-1D — Hyperbolic Auxiliary Mechanism Diagnosis**

本轮仅进行已有产物审计、只读分析和必要的无更新梯度诊断。

不得启动任何新训练。

---

## 1. 已有实验与路径

FYP 项目：

```text
/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
```

A0：普通 RoBERTa baseline

```text
results/exp6c0_roberta27_baseline/run_20261020T033547Z_vQWCGEh8
```

A1：Hyperbolic Auxiliary

```text
results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL
```

Original HypEmo：

```text
/home/zixinfeng/Documents/HypEmo-Reproduction
```

### 已确认的正式结果

| Test Metric | A0 | A1 | Delta |
|---|---:|---:|---:|
| Micro-F1 | 63.6241% | 63.9277% | +0.3036 pp |
| Macro-F1 | 57.9276% | 56.6248% | −1.3028 pp |
| Rare-F1 | 53.3160% | 46.1239% | −7.1921 pp |
| Macro Precision | 57.7245% | 58.5312% | +0.8067 pp |
| Macro Recall | 60.2233% | 56.9279% | −3.2955 pp |

Validation Macro-F1：

```text
A0: 58.135583%
A1: 56.020868%
```

两者均为：

```text
Backbone: roberta-base
Task: 27-label multi-label classification
Neutral output: excluded
Neutral-only samples: excluded
Train: 30,587
Validation: 3,834
Test: 3,821
Seed: 42
Epochs: 4
Batch size: 16
Fixed threshold: 0.30
```

注意：这里是 **27 类，不是 28 类**。不得与以前的完整 28 类实验混淆。

---

## 2. 安全与执行限制

严格要求：

1. A0、A1 源码、checkpoint、预测结果、训练日志全部冻结。
2. 不修改 Exp1–Exp6C 已有实验。
3. 不安装或升级 PyTorch、CUDA、NVIDIA Driver、Transformers 或系统环境。
4. 不启动训练、不执行 optimizer.step()、不更新模型参数。
5. 不运行新随机种子。
6. 不做新的正式 Test 推理。
7. 不根据 Test 结果调整阈值、gamma、Loss 或模型。
8. 不自动实现 HyCoEM、Parrott 新层级、ASL、Contrastive Loss 或新的网络架构。
9. 不重新计算历史测试集来挑选改进方案。
10. 不自动 Git push。

新建独立诊断目录：

```text
results/exp6c1d_hyperbolic_diagnostics/
    run_<timestamp>/
```

只向该目录写入诊断产物。

可读取已有训练集、验证集、配置、checkpoint 和封存指标。

**本轮所有进一步的预测分布分析，以 Validation 数据为主。已封存的 Test 结果只用于描述历史观察，不用于寻找最优改进方案。**

---

## 3. Phase 1 — 审计 A1 的真实数学实现

首先完整阅读：

```text
Exp6C-1 model implementation
Loss specification
Training configuration
Gradient logs
Checkpoint selection
Verification reports
```

不能只读 Summary。

需要回答：

### Q1：A1 的实际 Loss 是什么？

目前报告为：

\[
L = L_{\mathrm{BCE}} + 0.1 L_{\mathrm{hyp\_aux}}
\]

请从代码逐项还原完整数学公式。

尤其检查：

- 普通 BCE 如何归约？
- 多个 Positive Emotion 如何聚合？
- Auxiliary CE/NLL 项如何计算？
- Dot-product 项如何计算？
- 不同项是否按 batch mean、positive count 或类别数量归约？
- 每条样本的正标签数量是否影响辅助损失的有效尺度？
- 是否存在意料之外的类别权重或样本权重？
- 固定标签向量是否真的被冻结？
- 梯度是否正常传递到 RoBERTa 与 Hyperbolic Projection？

不得直接认定该 Loss 是错误的；要区分作者原有机制、我们的多标签适配以及可能的实现缺陷。

### Q2：当前辅助 Loss 数值是否明显偏大？

已知 A1 Test Loss：

```text
BCE = 0.08456518
Raw Auxiliary = 1.52376396
Weighted Auxiliary = 0.15237640
Total = 0.23694158
```

辅助项标量数值约为 BCE 的 1.80 倍。

但绝对不能仅凭这个比例宣称它主导了梯度。

需要结合训练日志和后续梯度诊断判断。

**Phase 1 输出：**

```text
01_LOSS_IMPLEMENTATION_AUDIT.md
```

如发现标签映射错误、Loss 公式和预注册设置不一致、梯度意外断开等实质性问题，记录证据并停止后续诊断，等待人工确认。

---

## 4. Phase 2 — 双曲辅助 Loss 梯度诊断

这是本轮最重要的一部分。

研究问题：

**Hyperbolic Auxiliary Loss 是否向共享 RoBERTa 参数施加了过强或相互冲突的梯度？**

### 4.1 优先使用已有日志

先检查 A1 的 `batch_log.csv`、`training_log.csv`、已有梯度产物。

区分：

- 已经记录的完整 Total Loss 梯度。
- BCE 单独梯度。
- Auxiliary 单独梯度。

不得把总梯度误认为辅助梯度。

### 4.2 如果日志不足，进行无参数更新梯度诊断

允许读取已经封存的 A1 最佳 checkpoint。

在训练数据中选定一个固定、可复现的小样本集合，记录全部 canonical IDs 和抽样规则。

样本必须包含：

- 单一正情绪样本。
- 多个正情绪样本。
- 常见情绪样本。
- 具有低频正标签的样本（若训练集中存在）。

使用 `torch.autograd.grad` 或等价方式，分别计算：

\[
g_{\mathrm{BCE}}=\nabla_\theta L_{\mathrm{BCE}}
\]

\[
g_{\mathrm{aux}}=\nabla_\theta(0.1L_{\mathrm{hyp\_aux}})
\]

其中 θ 为共享 RoBERTa 参数。

计算：

**1. 梯度范数**

```text
||g_BCE||
||g_aux||
||g_aux|| / ||g_BCE||
```

**2. 梯度方向一致性**

\[
\cos(g_{\mathrm{BCE}},g_{\mathrm{aux}})
\]

判断梯度是否：

- 大体同向。
- 基本无关。
- 明显相反。

**3. 分层梯度**

至少检查：

```text
RoBERTa early layers
RoBERTa middle layers
RoBERTa final layers
Classification head
Hyperbolic projection
Auxiliary decoder
```

对于不属于某项 Loss 计算图的组件，记录 `not connected by design`，不要将其误报为梯度故障。

**4. 数值稳定性**

检查有限性、梯度裁剪记录以及超曲投影是否接近数值边界。

全部操作不更新参数。

为了可复现，记录 `model.eval()` 状态、batch IDs、随机状态、梯度计算精度和任何人为设定的诊断条件。

不能把最佳 checkpoint 处的梯度直接当作整个训练过程所有步骤的梯度；它只能用于机制诊断。

**Phase 2 输出：**

```text
02_GRADIENT_INTERACTION_REPORT.md
02_gradient_summary.csv
```

---

## 5. Phase 3 — Validation 逐标签错误分析

研究问题：

**A1 为什么整体 Precision 提升，但 Recall 下降？是否少数标签主导了 Macro-F1 的变化？**

使用 A0、A1 已保存的 Validation 预测。

先验收：

```text
Same canonical IDs
Same gold labels
Same 27-label order
Same fixed threshold 0.30
```

为全部 27 类输出：

| Field | A0 | A1 | Delta |
|---|---|---|---|
| Positive support | | | |
| Predicted positive count | | | |
| Precision | | | |
| Recall | | | |
| F1 | | | |
| TP | | | |
| FP | | | |
| FN | | | |
| Mean positive probability | | | |

重点分析七个预先定义的 Rare 标签：

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

并查看：

- A0 和 A1 预测结果发生变化的样本比例。
- True Positive 转为 False Negative 的数量。
- False Negative 转为 True Positive 的数量。
- 预测概率整体上移还是下移。
- 低支持标签是否普遍发生 Recall 下降。
- Rare-F1 变化是否主要由一两个标签贡献。

可以增加一个只读的 Validation PR 曲线或不同阈值的敏感性图，**但这些仅用于判断预测分数分布及排序能力，不得据此选择新阈值作为本轮正式成绩**。

不要只分析 grief，因为我们已经知道它在 Test 上下降明显。必须完整分析七个 Rare 标签及全部 27 类，避免结论被测试集观察所引导。

所有结论需注明支持数。

**Phase 3 输出：**

```text
03_VALIDATION_LABEL_DIAGNOSTICS.md
03_per_label_validation_comparison.csv
03_rare_label_diagnostics.csv
```

如有必要，可制作少量清楚的可视化图表。

---

## 6. Phase 4 — Parrott 标签向量与几何关系审计

研究问题：

**我们现在的 Hyperbolic Label Embeddings 究竟从 Parrott 学到了什么？**

目前 A1 使用 HypEmo 原版预训练的 27 个固定标签向量。

先阅读原作者文件：

```text
data/go_emotion/label_tree.tsv
label_tree/go_emotion.bin
train_label_embedding.py
config.py
```

需要核实：

1. 实际 GoEmotions 层级有几层？
2. 哪些标签共享相同的父节点？
3. 哪些关系是直接父子边，哪些只是传递闭包？
4. 27 个正式标签与向量是否一一匹配？
5. 原树存在的 `nervousment` 拼写异常是否对预训练标签空间产生了影响？
6. Exp6C-1 是否确实使用了作者封存的预训练向量？
7. 当前标签向量是否固定，是否在 A1 训练中更新？

### 只读几何分析

利用已经保存的标签向量计算 27×27 的双曲距离矩阵。

检查：

- 同一父类别内标签的距离分布。
- 不同父类别之间的距离分布。
- 七个 Rare 标签的最近邻。
- Grief 与 sadness、remorse 等标签的距离。
- Relief 与正向情绪标签的距离。
- 是否存在向量范数异常或接近边界的情况。

这里不需要重新训练标签向量。

如已有 Exp4A 数据驱动层级报告，可作概念性对照：

```text
Exp4A:
Training-data-derived static hierarchy

HypEmo:
Parrott-style pretrained static embeddings

HyCoEM:
Learned label embeddings + dynamic hard negatives
```

不得把三种方法视为等价。

**特别注意：A1 标签向量是固定的，不能声称它们在训练过程中被常见标签挤走。**

我们可以分析固定标签之间的空间关系，以及经过训练的文本投影相对标签的位置，但必须将其区别开。

**Phase 4 输出：**

```text
04_PARROTT_EMBEDDING_AUDIT.md
04_label_distance_matrix.csv
04_rare_nearest_neighbors.csv
```

---

## 7. Phase 5 — 综合诊断和下一轮决策

汇总前四阶段，回答：

**Q1. 辅助 Loss 的梯度是否明显强于 BCE？**

要报告实际梯度范数和比例，而不是只依赖 Loss 数值。

**Q2. 双曲辅助监督与 BCE 的梯度方向是否存在冲突？**

区分共享层、不同 batch 和个别层的情况，不用单个 cosine 概括全部训练。

**Q3. Rare-F1 下降是否只是由 grief 主导？**

用 Validation 的完整标签结果与支持数回答，不能仅引用 Test 上的 grief 变化。

**Q4. Parrott Label Embeddings 是否显示出合理的层级结构？**

只报告实际距离和父子关系的关联，不把几何相关性当成性能贡献的证明。

**Q5. 目前有没有证据说明 Hyperbolic 本身无效？**

明确区分：

- 双曲辅助分支的整体作用。
- 辅助 Loss 的数学设计。
- 标签层级先验的作用。
- 双曲空间本身的贡献。

当前 A0/A1 不能完全隔离这些因素。

### 提出下一轮实验，但不要执行

根据诊断结果选择最值得验证的**一个**假设。

候选方向：

**方案 A：辅助监督强度问题**

如果辅助梯度明显过强或与 BCE 冲突，建议只改变辅助 Loss 系数 γ，其他设置全部不变。

可提出一个预先定义的较小候选系数，但不要批量搜索。

**方案 B：标签先验问题**

如果梯度基本健康，但怀疑固定 Parrott 先验贡献不足，考虑一个保持维度、参数和 Loss 相同的标签向量对照。

例如原始标签向量与受控打乱映射的向量比较。

**方案 C：多正标签对齐问题**

如果证据更支持辅助目标本身不适合 Multi-label，先给出一个受 HyCoEM 启发的、数学明确的多正标签对齐方案。

要求不会将真实正标签误当成 Hard Negative。

本次不实现。

**方案 D：随机种子不稳定**

如果诊断没有明确机制，且下降主要由低支持标签的波动造成，优先建议 A0/A1 的配对多随机种子复核，而不是马上调整模型。

### 最终报告必须包含

```text
05_FINAL_DIAGNOSTIC_REPORT.md
05_NEXT_EXPERIMENT_PROPOSAL.md
```

使用以下结构：

```text
1. What was verified?
2. What did we observe?
3. Which explanations are supported?
4. Which explanations remain hypotheses?
5. Is there evidence of implementation problems?
6. What is the single most useful next experiment?
7. What should NOT be changed yet?
```

最后附简洁中文总结，方便我向 FYP 导师汇报。

---

## 8. 完成条件与严格停止

本次任务结束条件：

```text
[ ] A0/A1 冻结状态核实
[ ] Loss 数学实现审计完成
[ ] BCE 与辅助项梯度诊断完成或说明证据不足
[ ] 27 个标签的 Validation 预测分析完成
[ ] Rare 标签预测变化分析完成
[ ] Parrott 标签层级与向量审计完成
[ ] 综合诊断报告完成
[ ] 提出一个最优先的后续实验建议
[ ] 没有启动新训练
[ ] 没有进行新的正式 Test
[ ] 没有修改旧代码或 checkpoint
```

如果某阶段遇到无法可靠验证的阻碍，明确报告，不得伪造结果或自动更改实验范围。

最后输出：

```text
=================================================
EXP6C1D_HYPERBOLIC_DIAGNOSTICS_COMPLETE
A0 / A1: FROZEN
27-label protocol: VERIFIED
Gradient diagnosis: REPORTED
Rare-label analysis: REPORTED
Parrott hierarchy: AUDITED
New training: NONE
Test tuning: NONE
=================================================
```

完成后停止，等待我的下一步授权。

**Research principle: Diagnose first, modify one factor at a time, evaluate on validation, and preserve the integrity of all existing experiments.**