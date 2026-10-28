# Exp6C-1S-Train — Hyperbolic Input Scaling Pilot (s=0.03)

## 0. 实验目标

继续 FYP 的 Hyperbolic Emotion Recognition 研究。

本轮已经完成：

- Exp6C-0（A0）：RoBERTa27 + ordinary BCE
- Exp6C-1（A1）：RoBERTa27 + BCE + 0.1 Hyperbolic Auxiliary Loss
- Exp6C-1D：辅助损失、梯度、Parrott 标签空间诊断
- Exp6C-1S：输入缩放的无训练几何敏感性分析

Exp6C-1S 已发现：当前 A1 的文本双曲映射存在明显边界截断。将辅助分支的输入尺度从 `s=1` 调整为 `s=0.03`，可在已检查的训练探针中消除截断，并恢复径向敏感性。

**但尚未证明改善分类性能。**

本轮正式执行一个单因素 pilot：

**Exp6C-1S-Train：RoBERTa27 + Hyperbolic Auxiliary Loss + Input Scaling (s=0.03)**

核心研究问题：

> 减少 Hyperbolic 文本投影的边界截断，能否改善多标签情绪分类的 Validation Macro-F1 和 Rare-F1？

本轮不得引入任何其他改动。

---

## 1. 项目位置及实验保护

FYP 项目：

`/home/zixinfeng/Documents/context-aware-hyperbolic-emotion`

已冻结 A0：

`results/exp6c0_roberta27_baseline/run_20261020T033547Z_vQWCGEh8`

已冻结 A1：

`results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL`

已有 Exp6C-1S 只读诊断结果请在现有目录查找。

本次新建独立目录：

```text
experiments/exp6c1s_scaled_train/
results/exp6c1s_scaled_train/run_<timestamp>/
```

禁止：

- 修改任何 A0、A1 或之前实验的源代码和 checkpoint。
- 覆盖已有 CSV、JSON、NPZ、日志或诊断结果。
- 修改 HypEmo 官方原始仓库。
- 更新 CUDA、NVIDIA Driver 或系统 Python。
- 未经批准改变依赖版本。
- 自动 Git push、reset 或 clean。
- 启动其他实验。

执行前核查旧实验保护清单，结束后再次核查。

---

## 2. 唯一允许的模型修改

保持 Exp6C-1 A1 完全相同的模型架构和训练目标。

原本辅助分支输入：

```python
hyperbolic_input = h
```

候选修改：

```python
hyperbolic_input = 0.03 * h
```

这里 `h` 必须是 A1 原本送入首次双曲 `expmap0` 的文本向量。

请先查明源码中的准确调用位置。

**只能在原双曲辅助入口做一次缩放。**

不得：

- 改动普通分类头接收的文本向量。
- 修改 RoBERTa 网络结构。
- 修改 HNN 层数、隐藏维度或初始化。
- 修改 expmap0 / proj 的数学实现。
- 修改 Poincaré 曲率。
- 修改 Parrott 预训练标签向量。
- 修改标签向量的冻结状态。
- 在后续 HNN 各层重复乘以 0.03。
- 增加新的归一化层或距离函数。

如果源码结构使单点输入缩放无法实现，停止并报告，不得擅自重写模型。

---

## 3. 必须完全一致的实验协议

沿用 A1：

| Setting | Value |
|---|---|
| Backbone | roberta-base |
| Dataset | GoEmotions simplified |
| Labels | 27-label multi-label |
| Train / Val / Test | 30587 / 3834 / 3821 |
| Neutral | 无输出，neutral-only 排除 |
| Seed | 42 |
| Epochs | 4 |
| Batch Size | 16 |
| Max Length | 50 |
| Optimizer | AdamW |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Scheduler | Linear decay |
| Warmup Steps | 765 |
| Total Steps | 7648 |
| Gradient Clipping | 1.0 |
| Formal Threshold | 0.30 |
| Checkpoint | Highest Validation Macro-F1 |

Loss 保持：

\[
L=L_{\mathrm{BCE}}+0.1L_{\mathrm{hyp\_aux}}
\]

沿用原有多正标签 auxiliary objective。

**禁止改 gamma=0.1，禁止调整 BCE 或辅助损失的 reduction，禁止加入 class weighting、ASL、contrastive loss 或其他新方法。**

---

## 4. 训练前必须完成的验收

正式训练前进行以下检查。

### 4.1 模型初始化一致

从与 A1 完全相同的预训练 RoBERTa snapshot 开始独立初始化。

不得从 A1 已训练的最佳 checkpoint 继续训练。

核验：

- 相同的 RoBERTa 权重。
- 相同的分类头初始权重。
- 相同的 HNN / decoder 初始化参数。
- 相同输入的初始分类 logits 一致。
- 数据顺序、标签映射、tokenizer 一致。

如果额外缩放导致 RNG 消耗发生变化，要记录，不得虚称所有训练步骤的 dropout mask 完全相同。

### 4.2 正向与梯度 Smoke Test

同一批固定训练样本，检查：

- 27-label multi-hot targets 正确。
- BCE 与 A1 定义完全相同。
- 仅辅助映射入口发生缩放。
- 标签 embedding 无意外更新。
- Loss 有限。
- 共享 encoder、HNN 和 decoder 梯度有限且非零。
- 分类头正常接受 BCE 梯度。
- 没有意外梯度断开。

### 4.3 几何 Smoke Test

记录：

```text
Before scaling: input norm
After scaling: input norm
First expmap norm
First clip rate
Later HNN clip rate
Final ball radius
Auxiliary/BCE gradient norm ratio
Gradient cosine similarity
```

要求与 Exp6C-1S 已有诊断合理一致。

同时验证 `s=1` 的只读前向行为可对应原 A1，不重新训练 s=1。

若出现严重异常，停止正式训练。

---

## 5. 正式训练：仅 seed42

Smoke 通过后，执行四轮完整训练。

每一轮保存：

```text
Training BCE
Training raw auxiliary
Training weighted auxiliary
Training total loss
Validation BCE
Validation auxiliary
Validation Macro-F1
Validation Micro-F1
Validation Rare-F1
Learning rate
Gradient clipping statistics
```

特别增加每个 epoch 的双曲几何诊断。

使用预先固定的少量训练样本 ID，追踪：

- 首次映射 clip 比例。
- HNN 后续 clip 比例。
- 最终双曲球坐标范数分布。
- 最终范数大于 0.99 的比例。
- 共享 encoder 辅助/BCE 梯度范数比例（可用固定探针检查，不进行额外更新）。

**重点检查：训练四轮后是否重新出现边界饱和。**

若后续层重新饱和，记录证据，不自动调低缩放系数。

---

## 6. Checkpoint Selection

每个 epoch 在完整 Validation 上评估。

固定：

`sigmoid probability >= 0.30`

仅使用最高 Validation Macro-F1 选择 checkpoint。

平局时选择较早 epoch。

保存最佳模型，训练结束后完整重载，并验证：

- Canonical sample IDs 一致。
- Gold labels 一致。
- 保存与重载 logits 一致。
- 固定阈值预测一致。
- 指标重算一致。

**本轮定位为探索性 pilot，先不执行新的 Test 推理。**

理由：此前 A0/A1 Test 已经观察过，本轮主要是判断一个由训练集几何诊断提出的缩放假设是否值得进一步验证。后续正式 Test 应在候选方法和确认协议冻结后进行，避免用 Test 反复挑模型。

---

## 7. Validation 主结果比较

比较三个模型：

- A0：普通 RoBERTa + BCE
- A1：RoBERTa + Hyperbolic Auxiliary，s=1
- A1-S：RoBERTa + Hyperbolic Auxiliary，s=0.03

已有 Validation：

| Metric | A0 | A1 |
|---|---:|---:|
| Micro-F1 | 63.4011% | 63.9513% |
| Macro-F1 | 58.1356% | 56.0209% |
| Rare-F1 | 55.0248% | 44.7456% |

从原始保存文件读取这些历史值，核验后再比较。

新模型输出：

```text
Micro Precision / Recall / F1
Macro Precision / Recall / F1
Weighted-F1
Rare-F1
Macro Average Precision
Per-label Average Precision
```

对 27 个标签保存：

```text
Support
TP / FP / FN
Precision
Recall
F1
Average Precision
Predicted positive count
Mean positive probability
```

七个 Rare 标签沿用：

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

尤其关注：

1. Macro-F1 是否从 A1 的 56.02% 恢复。
2. 是否达到或超过 A0 的 58.14%。
3. Rare-F1 是否恢复。
4. Rare 标签 AP 是否恢复，而不只是阈值下 F1 改变。
5. Micro-F1 是否存在明显代价。
6. 训练后边界截断是否持续减少。

不得因为 grief 没有恢复而删除 grief 或调低它的专属阈值。

---

## 8. 如何解释结果

提前规定以下分析规则。

### 情况 A：A1-S 超过 A1，但没超过 A0

结论应为：

“输入缩放缓解了当前 A1 的部分性能退化，但尚未超过普通 BCE baseline。”

### 情况 B：A1-S 超过 A0

结论应为：

“在当前 seed42、27-label 任务及固定评估协议下，缩放辅助分支取得了初步收益。”

不能称为已证明双曲几何优于欧氏空间。

需要后续多随机种子以及匹配的 Euclidean Auxiliary Control。

### 情况 C：A1-S 仍低于 A1

结论应为：

“几何饱和可以缓解，但单独缩放没有改善验证分类表现。”

不要自动继续尝试 s=0.01、0.05、0.1 或调整 gamma。

### 情况 D：几何改善，但 F1 没有改善

明确区分：

- 数值几何改进成立。
- 分类性能收益不成立。

不能为了证明方法有用而忽略负结果。

---

## 9. 需要交付的报告

创建：

```text
EXP6C1S_TRAIN_IMPLEMENTATION_AUDIT.md
EXP6C1S_TRAIN_SUMMARY.md
EXP6C1S_A0_A1_SCALED_COMPARISON.csv
EXP6C1S_GEOMETRY_EPOCH_REPORT.csv
EXP6C1S_RARE_LABEL_ANALYSIS.csv
EXP6C1S_VERIFICATION_REPORT.md
```

同时保存：

```text
config.json
environment.json
source_manifest.json
dataset_manifest.json
training_log.csv
checkpoint_selection.json
best_checkpoint/
validation_metrics.json
validation_per_label.csv
validation_predictions.npz
geometry_diagnostics.json
```

最终中文报告必须用简单清晰的语言回答：

1. 本次仅修改 s=0.03 是否得到验证？
2. 从头训练后双曲边界饱和是否仍然缓解？
3. 与 A0、A1 相比，Validation Macro/Micro/Rare-F1 怎么变化？
4. Grief、Relief、Pride 是否有改善？
5. AP 是否改善，还是只有固定阈值预测变化？
6. 梯度强度及方向发生了什么变化？
7. 目前是否有足够理由继续研究输入缩放？
8. 下一步最值得做的一个对照是什么？

不要只输出成功标记，必须输出真实结果和限制。

---

## 10. 严格停止规则

仅完成：

**Exp6C-1S-Train，s=0.03，seed42。**

不得自动：

- 训练 seed123 / seed2026。
- 搜索新的缩放系数。
- 调整 gamma。
- 加入新 Loss、Parrott 改造或 HyCoEM。
- 加入 Rare weighting。
- 运行正式 Test。
- 挑选表现最好的多个 run。
- 自动启动下一阶段。

完成后输出：

```text
=================================================
EXP6C1S_SCALED_SEED42_PILOT_COMPLETE
Backbone: roberta-base
Task: 27-label multi-label
Hyperbolic input scale: 0.03
Auxiliary coefficient: 0.1
Checkpoint: Validation Macro-F1
Formal threshold: 0.30
New Test inference: NONE
=================================================
```

如果未通过验收，输出失败原因，不打印完成标记。

如已有经过验证的实验完成通知机制，可安全复用，不得暴露凭据。

**本轮原则：只改一个因素，用 Validation 验证，先回答明确的问题，不因为结果不理想而连续修改模型。**