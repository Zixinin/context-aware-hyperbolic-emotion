# Exp6E-3 — Matched-Gamma Coordinate Alignment Control + Mandatory Backup

## 0. 任务目标

当前项目：**Hyperbolic Emotion Label Embeddings for Multi-label Emotion Recognition**

现在需要完成两项工作：

**Task A：执行 Exp6E-3 单种子匹配系数对照实验。**

**Task B：保护现有 FYP 项目，并将本轮新生成的代码、配置、checkpoint、日志和结果同步到外接硬盘，完成文件完整性校验。**

请将备份视为本轮的重要交付内容，而不是可选步骤。

用户预计在 **2026-10-31** 之后无法继续使用 D3020-A3，因此必须确保最新研究产物可以迁移到其他 Ubuntu 机器。

本轮不执行其他模型训练，不自动扩大到多随机种子。

---

## 1. 背景与研究问题

已经完成的相关实验：

- Exp6C-0 A0：RoBERTa27 + ordinary BCE。
- Exp6C-1 A1：RoBERTa27 + Hyperbolic Auxiliary，原坐标内积对齐，gamma=0.1。
- Exp6E-1 A1-G：Geodesic Alignment 无更新诊断，原 gamma=0.1 触发预注册尺度停止门，没有训练。
- Exp6E-2 A1-G-C：Geodesic Alignment + train-only calibrated gamma_G，seed42 Validation pilot。

Exp6E-2 使用：

\[
L_{\mathrm{A1GC}}
=
L_{\mathrm{BCE}}
+
\gamma_G\operatorname{mean}(C_i+G_i)
\]

其中：

\[
\gamma_G=0.02091538317504222
\]

\(C_i\) 为原有的 soft-target auxiliary CE，\(G_i\) 为多正标签真实 Poincaré 测地距离平均值。

Exp6E-2 seed42 Validation：

| Model | Micro-F1 | Macro-F1 | Rare-F1 | Macro AP |
|---|---:|---:|---:|---:|
| A0 | 63.4011% | 58.1356% | 55.0248% | 61.3012% |
| A1 | 63.9513% | 56.0209% | 44.7456% | 59.5187% |
| A1-G-C | 63.4171% | 58.6162% | 56.3096% | 60.0204% |

A1-G-C 的 Macro/Rare-F1 改善，但相对于 A1 同时改变了：

1. Alignment objective：coordinate dot product → geodesic distance。
2. Auxiliary coefficient：gamma 0.1 → gamma_G 0.02091538317504222。

因此不能单独把改善归因于真实测地距离。

**本轮研究问题：**

> 当辅助系数完全相同时，真实双曲测地距离对齐是否比原坐标内积对齐取得更好的多标签分类结果？

---

## 2. 本轮唯一新训练模型

实验名称：

**Exp6E-3 — A1-C-Gamma：Matched-Gamma Coordinate Alignment Control**

本轮以原 A1 的架构与 coordinate alignment 为基础。

原 A1：

\[
L_{\mathrm{A1}}
=
L_{\mathrm{BCE}}
+
0.1\operatorname{mean}(C_i-D_i)
\]

本轮：

\[
\boxed{
L_{\mathrm{A1CG}}
=
L_{\mathrm{BCE}}
+
0.02091538317504222
\operatorname{mean}(C_i-D_i)
}
\]

其中：

\[
q_{ik}=\frac{y_{ik}}{\sum_j y_{ij}}
\]

\[
C_i=-\sum_k q_{ik}\log\operatorname{softmax}(u_i)_k
\]

\[
D_i=\sum_k q_{ik}\,z_i^\top e_k
\]

严格保持原 A1 的：

- RoBERTa-base encoder。
- 原分类头及 logits。
- HNN Poincaré projection。
- Tangent-space auxiliary decoder。
- 多正标签 q 定义。
- 固定的27个 Parrott emotion label embeddings。
- 曲率 c=1。
- Hyperbolic input scale s=1。
- BCE 的原归约方式。
- Auxiliary CE 与 coordinate dot 的原归约方式。

**唯一修改：整个辅助目标的系数从 0.1 变为 gamma_G。**

不能使用 geodesic alignment，也不能引入 N1 distance-weighted BCE。

本轮与 A1-G-C 比较时，两个模型的 gamma_G 相同，主要区别是 alignment objective。

注意：同 gamma 不代表两个目标具有完全相同的实际梯度强度；报告中仍须说明这一限制。

---

## 3. 项目位置与实验保护

Ubuntu 项目：

`/home/zixinfeng/Documents/context-aware-hyperbolic-emotion`

Frozen A1：

`results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL`

Frozen A1-G-C：

`results/exp6e2_calibrated_geodesic_train/run_20261028T083257Z_RR6oxmlJ`

本轮创建：

```text
experiments/exp6e3_matched_gamma_coordinate/
results/exp6e3_matched_gamma_coordinate/run_<timestamp>/
```

不能修改已有 A0、A1、A1-S、N1、Exp6D-2、Exp6E-1/2 的任何代码或结果。

新代码必须独立存放，不得直接覆盖 A1 或 A1-G-C 实现。

记录：

- Git commit/status。
- Source SHA256。
- Python、PyTorch、Transformers 和 Geoopt 等相关版本。
- CUDA/GPU 信息。
- 配置和随机种子。
- 预训练模型来源。
- 27维 label order 与 Parrott label embedding 文件 SHA256。

不要自动升级依赖、重装环境、修改系统配置或 Git push。

---

## 4. 固定训练协议

| Setting | Value |
|---|---|
| Backbone | RoBERTa-base |
| Dataset | GoEmotions simplified |
| Task | 27-label multi-label |
| Train / Validation / Test | 30587 / 3834 / 3821 |
| Seed | 42 |
| Epochs | 4 |
| Batch Size | 16 |
| Max Length | 50 |
| Optimizer | AdamW |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Warmup | 765 steps |
| Total Steps | 7648 |
| Gradient Clipping | 1.0 |
| Input Scale | s=1 |
| Auxiliary Coefficient | 0.02091538317504222 |
| Threshold | 0.30 |
| Checkpoint Selection | Highest Validation Macro-F1 |

从与 A1-G-C 和 A1 匹配的同一预训练 RoBERTa snapshot 独立初始化。

**不得续训任何现有 best checkpoint。**

核对固定 seed42 下共享模型参数、分类头初始 logits、数据 ID 和标签映射的一致性。

如果历史初始化保存信息不足，应明确说明能够验证到什么程度，不得伪称逐步训练随机噪声完全一致。

---

## 5. 训练前验收

在正式运行 GPU 训练之前：

### 5.1 Forward parity

比较原 A1 代码与新实现：

- 原始 classification logits 应一致。
- 原始 coordinate auxiliary 应一致。
- BCE 应一致。
- 正标签 q 应一致。
- 标签向量及冻结状态应一致。

对于相同的固定输入，新实现与原 A1 的唯一预期差别应是总 Loss 中的辅助乘数。

### 5.2 Gradient parity

检查：

- 新旧辅助项在未乘系数之前数值一致。
- 新 weighted auxiliary gradient = 原 auxiliary gradient × (gamma_G / 0.1)，在相同参数和输入下成立。
- 分类 BCE 梯度保持一致。
- Encoder、HNN、aux decoder、classification head 梯度有限。
- 没有意外 detach。
- 没有 NaN/Inf。
- 同一 forward 下双曲投影未被额外修改。

请记录主要模块的梯度范数、BCE–aux cosine、裁剪相关统计。

不得因为梯度与 A1-G-C 不同而擅自重新校准 gamma。

### 5.3 Backup preflight

正式训练前检查：

1. Ubuntu 项目路径存在。
2. 实验脚本和当前研究结果可以正常读取。
3. 本轮输出目录空间足够。
4. 外接硬盘是否仍然挂载。
5. 外接硬盘剩余空间是否足以容纳本轮新产物及需要同步的文件。
6. 上一轮 A1-G-C、Exp6D-2 和本次源码是否已有至少一个可验证的迁移副本。

用户此前使用的外接硬盘路径为：

`/media/zixinfeng/外接硬盘/fzx`

必须先通过挂载表核验实际设备及可写状态。不能仅凭目录存在就假设外接硬盘已经挂载，更不能把备份误写到 Ubuntu 本机同名目录。

如该路径发生变化，请检查已挂载磁盘并报告真实位置，不凭名称猜测新的目标设备。

若外接硬盘不可用，停止自动正式训练并明确报告备份阻塞原因，不要显示备份成功。保持所有文件不变。

---

## 6. 正式训练

Smoke test 和备份预检通过后：

**仅执行 A1-C-Gamma，seed42，4 epochs。**

记录每一轮：

```text
Training BCE
Training original coordinate auxiliary
Training weighted auxiliary
Training total loss

Validation Micro-F1
Validation Macro-F1
Validation Rare-F1
Validation Macro AP

Learning rate
Gradient norm
Gradient clipping frequency
```

另外保存双曲几何诊断：

```text
First expmap clip rate
Later HNN clip rate
Final hyperbolic radius distribution
Ratio of weighted aux to BCE gradients
BCE–aux gradient cosine
```

如出现边界饱和，正常记录，不改 s。

如出现训练不稳定、异常 loss 或非有限梯度，按既有保护规则停止，不能自动调整 gamma 或其他训练设置后重跑。

---

## 7. 结果对照

比较四个模型：

| Model | Alignment | Gamma |
|---|---|---:|
| A0 | None | — |
| A1 | Negative coordinate dot | 0.1 |
| **A1-C-Gamma（本轮）** | Negative coordinate dot | **0.02091538317504222** |
| A1-G-C | Positive geodesic distance | **0.02091538317504222** |

分别回答：

**Question A：辅助系数降低有没有帮助？**

比较：

A1-C-Gamma vs A1

**Question B：在相同 gamma 下，测地对齐是否更好？**

比较：

A1-G-C vs A1-C-Gamma

**Question C：哪一个模型超过普通 RoBERTa？**

比较：

A1-C-Gamma vs A0

A1-G-C vs A0

主要关注：

- Validation Macro-F1（主指标）。
- Rare-F1。
- Micro-F1。
- Macro Average Precision。
- 27个标签逐类 F1/AP。
- grief、relief、pride、nervousness 等既定 Rare 标签。

特别报告：

- 有多少标签改善、多少下降。
- Macro-F1 变化是否主要来自少量 Rare 标签。
- Grief、Relief 的 AP 是否也改善，而不只是固定阈值 F1 提升。
- BCE 和辅助梯度相对贡献是否改变。

不能仅因为某种模型在 seed42 表现最好，就宣布其机制已经得到证明。

---

## 8. Validation-only 与停止边界

严格执行：

- 完整四轮训练。
- 每轮完整 Validation。
- 固定阈值 0.30。
- 只按 Validation Macro-F1 选最佳 checkpoint。
- 平局选最早 epoch。
- 保存、重载并核验完整 3834 条 Validation 预测。
- 不进行任何新的 Test 推理。

不得自动：

- 训练 seed123 / seed2026。
- 改变 gamma。
- 修改输入尺度。
- 添加 N1。
- 加入其他 Loss。
- 修改 Parrott 标签层级。
- 更换模型或扩大训练预算。
- 运行欧氏对照。
- 重新训练旧模型。

---

## 9. 交付文件

本轮独立输出：

```text
EXP6E3_IMPLEMENTATION_AUDIT.md
EXP6E3_INITIALIZATION_PARITY.json
EXP6E3_LOSS_GRADIENT_AUDIT.md
EXP6E3_TRAINING_SUMMARY_CN.md
EXP6E3_A1_A1CG_A1GC_COMPARISON.csv
EXP6E3_RARE_LABEL_ANALYSIS.csv
EXP6E3_GEOMETRY_REPORT.csv
EXP6E3_CHECKPOINT_RELOAD_VERIFICATION.json

config.json
environment.json
source_manifest.json
training_log.csv
checkpoint_selection.json
validation_metrics.json
validation_per_label.csv
validation_predictions.npz
best_checkpoint/
```

所有必要模型参数、优化器状态与配置若需用于可复现训练或续训，应按当前实验规范保存；任何未保存状态必须列明。

最终中文报告回答：

1. 本轮是否严格只修改辅助系数？
2. Macro-F1 是否比原 A1 改善？
3. A1-G-C 与相同 gamma 的坐标内积模型相比谁更好？
4. Rare 标签是否普遍受益？
5. 辅助梯度与边界截断发生什么变化？
6. 这次结果能否支持测地对齐优于原坐标内积？
7. 后续是否值得做三 seed 验证？

---

## 10. 强制文件备份与迁移保护

用户预计 2026-10-31 后无法再使用 D3020-A3。

请在本轮实验中实施独立的备份步骤。

### 10.1 备份范围

至少包含：

```text
experiments/exp6e3_matched_gamma_coordinate/
results/exp6e3_matched_gamma_coordinate/

experiments/exp6e2_calibrated_geodesic_train/
results/exp6e2_calibrated_geodesic_train/

experiments/exp6d2_paired_multiseed_v1_1/
results/exp6d2_paired_multiseed/

原有Exp6C、Exp6D相关代码、配置与结果
项目根目录必要配置
数据集manifest、label映射、标签向量
所有重要checkpoint
训练及验证日志
复现实验所需环境说明
Git元数据（如有）
```

优先对整个 FYP repository 做增量备份，而不是只保存最新 Markdown。

可排除：

```text
__pycache__/
*.pyc
可重建的Python虚拟环境
不包含训练产物的临时cache
```

**绝不能排除模型checkpoint、原始预测数组、label embedding、CSV、NPZ、配置、源码或实验日志。**

如果存在指向项目外部文件的符号链接，列出目标并核对备份是否覆盖真实文件；不能只备份一个无法使用的链接。

### 10.2 外接硬盘备份

在验证挂载和磁盘空间后，将项目增量同步到外接硬盘上独立的 FYP 备份目录。

优先复用已核实的现有备份根目录，不要误覆盖其他项目。

建议使用 rsync，禁止 `--delete`。

不得删除或清空之前的备份。

执行后必须核验：

- 新实验源码存在。
- 新结果目录存在。
- Best checkpoint 存在且大小合理。
- 配置、报告和全部日志存在。
- 关键文件 SHA256 一致。
- 文件数量和总容量符合预期。
- 未出现被忽略的 rsync 错误。

应额外生成：

```text
BACKUP_MANIFEST_SHA256.txt
BACKUP_SOURCE_AND_DESTINATION.json
BACKUP_RSYNC_LOG.txt
BACKUP_VERIFICATION_REPORT.md
```

备份验证需独立读取目标磁盘上的文件，不只验证源目录。

### 10.3 本地 Mac 备份界限

不要假设 Ubuntu 可以直接 SSH 连接到用户 MacBook。

不要声称已经完成 Mac 本地备份，除非有真实访问权限、实际传输记录和目标文件核验结果。

Ubuntu 端应生成：

```text
MAC_BACKUP_TRANSFER_INSTRUCTIONS.md
```

其中记录准确的 Ubuntu 源路径、必需文件、总大小以及用户从 Mac 通过现有 `ubuntu4090` SSH alias 拉取文件所需的步骤。

若已成功备份外接硬盘，但尚未复制到 Mac，必须标明：

```text
EXTERNAL_DRIVE_BACKUP: PASS
MAC_LOCAL_BACKUP: PENDING_USER_TRANSFER
```

不得将外接硬盘备份冒充 Mac 本地备份。

### 10.4 以后新实验的备份机制

创建一个安全、可重复运行的项目备份脚本，放在本轮新实验目录或单独的新工具目录中。

脚本要求：

- 自动核实外接硬盘挂载状态。
- 增量复制新实验代码与结果。
- 不删除历史文件。
- 保留 checkpoint、日志和运行配置。
- 同步完成后校验文件。
- 打印真实 PASS/FAIL。
- 允许用户在后续几天再次执行。

不要建立后台定时任务，不修改系统 crontab，不自动远程上传到未经批准的服务。

---

## 11. 最终成功条件

科学实验与备份应分别报告状态。

只有实验训练与验收完成，才能打印：

```text
EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE
```

只有外接硬盘数据真正校验通过，才能打印：

```text
EXP6E3_EXTERNAL_BACKUP_VERIFIED_PASS
```

Mac 本地备份必须由真实 Mac 端传输与校验结果确认，不允许提前打印成功。

最终终端必须展示：

```text
================================================
EXP6E3 FINAL STATUS

Experiment:
  Model: A1-C-Gamma
  Seed: 42
  Gamma: 0.02091538317504222
  Alignment: Coordinate Dot Product
  Test inference: NONE

Validation:
  Micro-F1: <actual>
  Macro-F1: <actual>
  Rare-F1: <actual>
  Macro AP: <actual>

Verification:
  Checkpoint reload: PASS/FAIL
  Historical protection: PASS/FAIL

Backup:
  External drive: PASS/FAIL
  Mac local: PENDING / VERIFIED
  Source path: <actual>
  Backup destination: <actual>
================================================
```

若正式训练失败或备份失败，应保留已产生的文件，明确报告部分完成状态，禁止伪造成功标记。

**最高优先级：实验可复现、已有结果不被覆盖、在电脑迁移前保全研究资料。**