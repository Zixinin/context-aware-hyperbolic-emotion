# Codex Execution Skill — Exp6C-0: RoBERTa 27-label Multi-label Baseline

## 0. 研究背景与目标

当前 FYP：

**Context-aware Multi-label Emotion Recognition with Hyperbolic Emotion Label Embeddings**

已有实验：

- Exp6A：成功完成 ACL 2023 HypEmo 原版单标签五 seed 复现。
- Exp6B Stage A：完成 Corrected Single-label HypEmo。
- Exp6B Stage B：完成 BERT + 28-label Multi-label HypEmo Formula A，但没有优于匹配 CLS baseline。

Exp6B 的问题在于，从原版 HypEmo 到多标签适配，同时改变了 backbone、标签数量、数据、训练设置和 loss，难以分析各因素的影响。

因此开启一个新的受控实验系列 Exp6C。

**本次仅执行 Exp6C-0（A0）：RoBERTa-base + 27-label Multi-label BCE baseline。**

核心研究问题：

> 在保留 RoBERTa-base 和 27 种具体情绪标签、但允许每条文本具有多个情绪标签的情况下，普通分类器可以达到怎样的性能？

本轮暂时不使用任何双曲几何或 HypEmo 辅助损失。

本轮不尝试创新，也不要求性能超过已有实验。

目的是建立一个干净、可信、可复用的 baseline，供后续 Exp6C-1、Exp6C-2 进行控制变量比较。

---

## 1. 项目位置与安全要求

现有 FYP 项目：

```bash
/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
```

原版 HypEmo 复现：

```bash
/home/zixinfeng/Documents/HypEmo-Reproduction
```

MVFA 项目：

```bash
/home/zixinfeng/Documents/MVFA-AD
```

**严格要求：**

1. 不修改 Exp1–Exp6B 已有代码、配置和实验结果。
2. 不修改原作者 HypEmo 仓库。
3. 不覆盖任何旧 checkpoint、CSV、JSON 或日志。
4. 不安装或更改 NVIDIA Driver、CUDA、系统 Python。
5. 优先复用已验证兼容 RoBERTa 的现有独立环境；先核实依赖，再决定是否需要新虚拟环境。
6. 不自动执行 Git push、merge、reset、clean。
7. 所有新代码和产物使用独立目录。
8. 不复制原 HypEmo 第三方源码到 FYP 项目。

建议目录：

```text
experiments/
└── exp6c0_roberta27_baseline/

results/
└── exp6c0_roberta27_baseline/
    └── run_<timestamp>/
```

---

## 2. 实验范围

本次只训练一个模型：

**A0 — RoBERTa-base + 27-label Multi-label BCE**

模型结构：

```text
Input Sentence
       ↓
RoBERTa-base
       ↓
Last Hidden State
       ↓
First-token Representation (<s>)
       ↓
Dropout
       ↓
Linear(768 → 27)
       ↓
27 Emotion Logits
       ↓
BCEWithLogitsLoss
```

预测时对每个 logit 单独使用 sigmoid。

**禁止加入：**

- Hyperbolic geometry
- Poincaré Ball
- Hyperbolic label embeddings
- Hyperbolic auxiliary loss
- Distance-weighted BCE
- Label-query attention
- LGSA
- Emotion hierarchy
- Graph / GNN
- Contrastive loss
- ASL / Focal Loss
- Class weights / resampling
- Data augmentation
- Extra projection layers
- 任何新的创新模块

这是一个纯粹的 RoBERTa 多标签分类 baseline。

---

## 3. 数据处理：27-label Multi-label GoEmotions

这一部分特别重要，必须先检查并验证。

### 3.1 数据来源

使用现有 FYP 中的 GoEmotions simplified 官方 train / validation / test split。

原始 FYP 数据包含：

```text
27 emotions + neutral = 28 labels
```

不能使用 HypEmo 官方单标签 CSV 替代本次数据。

不能重新随机划分数据集。

不能转换成单标签任务。

### 3.2 本次标签定义

只保留 27 个具体 emotion labels。

删除分类输出中的：

```text
neutral
```

保留原 FYP 中其他 27 个标签的相对顺序。

必须从现有 canonical label mapping 获取顺序，不得手动假定 neutral 永远位于某个固定下标。

### 3.3 样本保留规则

这是本轮预先确定的主要实验协议：

**保留所有至少包含一个非 neutral 正标签的样本。**

例子一：

```text
Original labels:
[joy]

New labels:
[joy]

保留
```

例子二：

```text
Original labels:
[excitement, nervousness]

New labels:
[excitement, nervousness]

保留
```

例子三：

```text
Original labels:
[neutral]

New labels:
[]

从本轮主要训练、验证和测试数据中排除
```

例子四：

```text
Original labels:
[neutral, sadness]

New labels:
[sadness]

保留
```

如果数据中存在 neutral 与具体 emotion 共存的情况，不能将整条样本删除。

不得删除多标签样本，也不得把多个正标签压缩成一个标签。

不得对标签执行单标签 argmax。

### 3.4 数据验收

生成：

```text
DATASET_AUDIT.md
DATASET_MANIFEST.json
```

至少报告：

- 原始 28 类各 split 样本数。
- 过滤后的 27 类各 split 样本数。
- 删除的 neutral-only 样本数量。
- 保留的单情绪样本数量。
- 保留的多情绪样本数量。
- 多标签样本所占比例。
- 每个 emotion 的 train/val/test positive support。
- 多标签平均正例数。
- 标签索引与原 FYP 的映射关系。
- 原始 canonical sample IDs 和过滤后的 IDs。

验证：

- 仍为 27 维 multi-hot labels。
- 每条保留样本至少有一个正标签。
- 训练、验证、测试互不混用。
- 未产生新的数据泄漏。
- 保留的样本内容和标签未经额外修改。

如果数据来源、label IDs 或 split 不能可靠核对，停止训练并报告问题。

**注意：本次数据协议与原版 HypEmo 的单标签数据协议不同，也与之前完整 28 类数据协议不同。必须在结果中明确标注。**

---

## 4. 模型与训练设置

### 4.1 Backbone

使用：

```python
roberta-base
```

记录实际加载的：

```text
Model revision
Tokenizer revision
Configuration
Checkpoint SHA / snapshot identity
```

使用 RoBERTa 对应的原始 tokenizer。

主实验文本最大长度：

```text
max_length = 50
```

本轮采用已有 FYP 的统一训练预算，而不是原作者单标签 20-epoch、batch100 的运行协议。

这是为后续 Exp6C-1 / Exp6C-2 建立统一多标签对照，并不构成对原论文单标签协议的精确复现。

### 4.2 统一训练超参数

```text
Backbone: roberta-base
Max length: 50
Batch size: 16
Epochs: 4
Seed: 42
Optimizer: AdamW
Learning rate: 5e-5
Weight decay: 0.01
Warmup ratio: 0.10
Scheduler: linear decay
Classification loss: BCEWithLogitsLoss
Formal threshold: 0.30
```

学习率调度总步数必须根据新的过滤后训练集大小动态计算。

Warmup 步数按照现有 FYP 的 ceil 规则计算并记录。

确认其余细节与现有 Exp5A CLS / Exp6B 匹配协议一致：

```text
Optimizer parameter groups
Gradient normalization
Gradient clipping
Dropout
Deterministic settings
DataLoader generator
Checkpoint selection
Evaluation implementation
```

如果发现既有实验记录中存在不一致，先在报告中说明。

不要静默采用不一致的实现。

### 4.3 分类头设计

使用 RoBERTa 最后隐藏层的第一个 token：

```python
h = last_hidden_state[:, 0, :]
```

随后经过与现有 FYP CLS baseline 一致的 dropout/classifier 结构。

不要直接调用未经核查的 `RobertaForSequenceClassification` 默认分类头，因为其内部可能引入额外的 dense / tanh 等操作。

**必须将 A0 的 encoder、文本池化与普通分类头设计固定下来。**

未来 A1 / A2 必须复用这条完全相同的分类路径，只增加受控的辅助组件。

否则无法隔离 HypEmo 的实际贡献。

---

## 5. Loss

使用标准：

```python
BCEWithLogitsLoss
```

即：

```text
Model logits: [B, 27]
Targets:      [B, 27]
```

分类目标：

\[
L_{\mathrm{A0}}=\mathrm{BCEWithLogitsLoss}(z,y)
\]

使用正常的标准平均损失归约，不进行样本加权或类别加权。

特别禁止：

```text
Distance weighting
Positive class weighting
Special rare-label weighting
Neutral weighting
Auxiliary loss
```

原因是本次需要先建立最干净的多标签分类参照。

---

## 6. Checkpoint Selection and Evaluation

每个 epoch 结束后，在 validation split 上评估。

固定阈值：

```text
sigmoid probability >= 0.30
```

正式 checkpoint 选择标准：

**Highest Validation Macro-F1 at fixed threshold 0.30**

出现完全相同的最高分时选择更早 epoch。

不能根据 Test 选择：

- Checkpoint
- Threshold
- Epoch
- Seed
- Hyperparameters

完成训练后：

1. 载入最佳 validation checkpoint。
2. 在整个验证集上重新推理。
3. 确认重载结果与先前保存结果一致。
4. 再执行一次正式 Test。
5. 保存所有必要数据与指标。

确保验证和测试调用 `model.eval()` 与 `torch.no_grad()`。

---

## 7. Evaluation Metrics

本轮输出 27 个具体 emotion labels。

必须报告：

```text
Micro Precision
Micro Recall
Micro-F1

Macro Precision
Macro Recall
Macro-F1

Weighted-F1

Rare-F1
```

Rare labels 沿用此前固定的七类：

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

每个标签输出：

```text
Label
Support
TP
FP
FN
Precision
Recall
F1
```

同时报告：

```text
Predicted positive counts
True positive counts
Number of predictions per example
```

所有指标由保存的完整测试预测重新计算验证。

严格使用：

```python
metric(y_true, y_pred)
```

不要继承原版 HypEmo 反向传入 F1 参数的问题。

---

## 8. 可选的 Neutral-only 诊断

虽然本轮主要任务不预测 neutral，但为了后续研究，可在不影响正式训练和模型选择的前提下增加一项只读诊断。

在完成正式模型训练和 checkpoint 选择后，使用原始 test split 中被排除的 neutral-only 文本进行推理。

观察：

```text
模型是否对这些文本错误预测具体 emotion
```

输出：

- Neutral-only 样本数。
- 至少误报一个具体情绪的样本比例。
- 平均每个样本的错误正情绪预测数量。

这项结果必须标注为：

**Supplementary neutral-only diagnostic**

不能并入主要 27-label Test Macro-F1，也不能用它调整 threshold。

如实现会影响主实验代码或需要额外训练，则跳过。

---

## 9. 与历史实验的关系

本轮主要目标不是直接超过之前的 Exp1–Exp6B，而是建立独立的 RoBERTa + 27-label baseline。

可以附带一个历史实验表，记录：

- Exp2 LGSA
- Exp5A CLS
- Exp5A Mean
- Exp6B Multi-label HypEmo
- Exp6C-0 RoBERTa 27-label BCE

但必须同时注明：

```text
Backbone
Label count
Sample filtering
Training split
Loss
Epochs
Seed
Threshold
Checkpoint rule
```

**禁止直接将 Exp6C-0 的 27 类 Macro-F1 与 Exp6B 的 28 类 Macro-F1 相减，声称模型提升或下降。**

两者的输出标签和测试样本已经不同。

**同样不能将它与原始 HypEmo 单标签 Weighted-F1 66.3% 直接比较。**

未来最重要的比较将是：

```text
Exp6C-0:
RoBERTa + 27 Multi-label + BCE

vs

Exp6C-1:
RoBERTa + 27 Multi-label + BCE
+ Hyperbolic Auxiliary Loss
```

两个模型必须复用完全相同的数据协议、分类路径和训练设置。

---

## 10. 训练前 Smoke Test

正式训练前，必须检查：

- 正确读取新的 27-label 数据。
- Multi-hot targets 维度正确。
- 多标签样本仍然有多个正值。
- RoBERTa tokenizer 和 model 加载成功。
- 所有张量维度匹配。
- BCE Loss 有限。
- Backward 正常运行。
- BERT/RoBERTa Encoder 和分类头梯度有限非零。
- Optimizer / Scheduler 正常。
- GPU 运行正常。
- 不影响已有项目和实验结果。

Smoke 完成并验收后，才启动完整 seed42 训练。

不得以 Smoke 结果替代正式实验。

---

## 11. 正式训练与输出

本轮只执行：

```text
Exp6C-0
Model: RoBERTa 27-label Multi-label BCE
Seed: 42
Epochs: 4
```

保存：

```text
results/exp6c0_roberta27_baseline/
└── run_<timestamp>/
    ├── config.json
    ├── environment.json
    ├── dataset_manifest.json
    ├── training_log.csv
    ├── checkpoint_selection.json
    ├── best_checkpoint/
    ├── validation_metrics.json
    ├── test_metrics.json
    ├── test_per_label_metrics.csv
    ├── rare_label_metrics.json
    ├── test_predictions.npz
    ├── EXP6C0_SUMMARY.md
    └── VERIFICATION_REPORT.md
```

保存的预测必须能够与 canonical sample IDs 对齐，便于之后 A1/A2 进行逐样本比较。

不要只输出控制台中的最终 F1。

---

## 12. 必须回答的研究问题

生成：

```text
EXP6C0_SUMMARY.md
```

包含以下内容：

1. 新的 27 类多标签数据集有多少样本？保留多少单标签/多标签样本？
2. RoBERTa-base 是否成功完成 4 epochs？
3. 最佳 validation checkpoint 位于哪个 epoch？
4. 正式 Micro / Macro / Weighted / Rare-F1 是多少？
5. 哪些 emotion 表现最好，哪些最差？
6. 七个 rare emotion 表现如何？
7. 与已有 28-label 实验有哪些协议差异？
8. 该 baseline 是否已经足够规范，可以作为后续 A1/A2 的匹配对照？
9. 有没有异常 loss、异常预测或梯度问题？
10. 如何在下一阶段保持完全相同的分类路径，只增加双曲辅助监督？

不得因为指标不如预期而自动修改模型或重新调参。

---

## 13. 完成条件

全部满足以下要求才视为完成：

```text
[ ] 27-label 数据过滤与标签顺序验证通过
[ ] Multi-label 样本保留正确
[ ] RoBERTa 模型加载正确
[ ] Forward/backward smoke 通过
[ ] Seed42 完整训练 4 epochs
[ ] Validation Macro-F1 选轮正确
[ ] 最佳 checkpoint 保存成功
[ ] Checkpoint 重载结果一致
[ ] 正式 Test 只执行一次
[ ] 27-label 指标与 Rare-F1 重算一致
[ ] 数据、配置、模型版本和结果完整归档
[ ] Existing Exp1–Exp6B 不受影响
```

最终成功标记：

```text
==================================================
EXP6C0_ROBERTA27_MULTILABEL_BASELINE_VERIFIED
Backbone: roberta-base
Task: 27-label multi-label emotion classification
Loss: BCEWithLogitsLoss
Seed: 42
Status: COMPLETE
==================================================
```

如果失败，明确输出错误原因和被阻断的阶段，不得打印成功标记。

如果 Ubuntu 已有经过验证的 ntfy / Telegram / 企业微信通知方式，可以在实验完成时复用，但不得泄露 token、凭据或修改通知系统。

---

## 14. 严格停止规则

**完成 Exp6C-0 后立即停止。**

不要自动运行：

```text
Exp6C-1: Hyperbolic Auxiliary Loss
Exp6C-2: Distance-weighted BCE
Exp6C-3: Normalized Distance Weighting
```

不要自动启动额外 seed。

不要因为 Macro-F1 较低就重新训练另一个模型。

不要为了接近 HypEmo 原论文结果而改变数据或指标。

本次研究的唯一任务：

**建立一个经过完整验证的 RoBERTa + 27-label Multi-label BCE baseline，为后续受控引入 HypEmo 机制打好基础。**