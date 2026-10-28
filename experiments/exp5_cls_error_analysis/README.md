# Exp5 CLS 三种子错误分析（诊断工具，尚非实际结果）

只读输入：CLS seed42 位于 `results/exp5a_controls/run_20261013T103802Z/cls_seed42`，
seed123/2026 位于 `results/exp5a_controls_multiseed/run_20261013T113904Z/cls_seed*`。
三套原实验代码及运行产物保持不变。不读取 Mean/Query 的预测，不加载 checkpoint、
不创建模型/优化器，不进行 GPU 前向、训练、阈值搜索或 test 调参。

使用 Ubuntu 的绝对 `.venv-bert/bin/python`，只依赖既有 numpy/pandas 与 Python 标准库，
不安装任何包。启动器隐藏 CUDA、禁止写 pycache，带有原有 ntfy 的开始/成功/失败通知。
通知失败不改变诊断退出码。`notify-test` 必须成功且手机收到后再执行 `analyze`。

## 运行

```bash
bash experiments/exp5_cls_error_analysis/launch.sh notify-test
bash experiments/exp5_cls_error_analysis/launch.sh analyze
```

输出位于新的 `results/exp5_cls_error_analysis/run_*_analysis`；启动日志、测试日志与通知
位于另一个新的 `run_*` 目录，不加入封存分析目录。已有目录绝不覆盖。
只有出现 `EXP5_CLS_ERROR_ANALYSIS_COMPLETE_VERIFIED` 且
`ANALYSIS_SUCCESS.json` 存在，才能称为实际诊断完成。本地 synthetic tests 不代表该状态。

## 核验

- 三个 checkpoint 的原 SUCCESS 全文件清单/SHA256、原代码 manifest 与依赖链、完整配置。
- 原训练完成状态、10,856 updates、4 epochs、记录的 GPU/梯度/数值重载证明。
- 选择记录等于 validation Macro-F1 的最早最大值，不强制选 epoch4。
- 28 标签顺序、官方三个 TSV 的 SHA256、跨 split ID 与训练标签频率。
- 每个 NPZ 的 `[N,28]` logits、逐条 targets 与 ID 顺序完全对应官方 split。
- validation/test 的固定0.30整体和逐标签指标与保存报告重算一致。
- 分析前后逐一比较原三个 CLS 目录、三套源代码、TSV、preflight、asset manifest 的清单/SHA256。

不会加载 checkpoint 来再次验证 checkpoint→logits 数值一致性，只使用封存哈希和原重载记录。
若必要输入缺失/不一致，停止并输出 `ANALYSIS_FAILED.json` 与具体缺失说明，绝不新增推理补齐。

## 输出与定义

主报告：`EXP5_CLS_ERROR_ANALYSIS.md`。

| CSV | 内容 |
| --- | --- |
| label_counts.csv | 28 类 train/validation/test 正例数与训练负正比 |
| label_metrics_by_seed.csv | validation/test × 3 seeds × 28 labels 的 P/R/F1、TP/FP/FN/TN、Pred |
| label_summary.csv | 每标签均值、样本SD、种子各值、共同漏检、预测分歧、描述性诊断 |
| rare_group_metrics.csv | rare7、rare5、grief+relief、nonrare21、all28 逐种子组内 macro/micro |
| pair_overlap.csv | 两个 split × 3 seeds × 全378对的共现/重合与分层交叉错误 |
| pair_patterns.csv | 预定义三对的4个真实层 × 4个预测模式计数 |
| validation_pair_examples.csv | 仅 validation，预定义三对的源顺序前3个交叉错误实例/方向/seed |
| validation_probability_summary.csv | 正/负例及TP/FP/FN/TN的概率均值、分位数 |
| validation_probability_bins.csv | 正/负例固定概率直方图区间的数量 |
| validation_threshold_neighborhood.csv | FN<.25、.25≤FN<.30、.30≤FP<.35、FP≥.35的数量 |
| validation_BCE_proxy.csv | 已有 validation 概率的正负 BCE 分解，不是训练梯度 |

CSV 中 rates 为0–1，counts 为计数；单模型计数为整数，跨种子计数均值为浮点数。
零分母的原指标按 frozen zero_division=0 复现，同时提供 defined 标记；条件错误率的零分母
留空/报告为 n.a.。统计单位是固定 split 上的三种子，不合并预测、不当成三份独立数据。

诊断规则（描述性，非统计显著性）：R<.5 且 P−R≥.1 为 recall_dominant，Precision对称；
其余 P/R 同时<.5 为 both_low。F1 SD≥.05 或极差≥.10 为 seed_sensitive。
这些固定规则不是用test寻优，也不自动指定下一轮训练标签/参数。

相似情绪预先固定 anger/annoyance、sadness/disappointment、fear/nervousness。
真实 A+B 双正例的双预测是合法命中。仅 A 标注层漏A多报B，记为“相对标注的交叉错误”，
不能据此断言标注错误、语义误解或强制互斥。其余375对只作完整描述，不按test排名选对。

ASL、语义标签嵌入仅基于 validation 观测给出有条件假设，并链接原论文。没有实现新方法。
不报告 training P/R 或梯度，标准产物没有相关预测/记录。本任务完成后停止。
