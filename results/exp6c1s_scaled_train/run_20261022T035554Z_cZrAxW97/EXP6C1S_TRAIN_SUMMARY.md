# Exp6C-1S-Train：s=0.03，seed42，Validation-only pilot

输入缩放缓解了当前A1的部分性能退化，但尚未超过普通BCE baseline。

This result verifies only the declared computational claim under the recorded environment and protocol.
本轮仅将A1双曲辅助入口h改为0.03*h，冻结的A0/A1/原始HypEmo源码未改。普通分类输入、架构、初始化、固定27类标签向量、BCE/Formula A、gamma0.1与原训练协议不变。
四轮7648步已完成，765 warmup；按完整Validation Macro-F1严格大于选轮，平局取早。最佳epoch=4；3834条Validation完整重载逐元素logits/IDs/gold/固定阈值预测和指标一致。新Test推理=0。

## 1. 真实Validation比较（阈值>=0.30）

| 指标 | A0 | A1(s=1) | A1-S(s=.03) | 对A0变化pp | 对A1变化pp |
|---|---:|---:|---:|---:|---:|
| micro_precision | 60.4761% | 63.1013% | 62.5595% | +2.0834 | -0.5417 |
| micro_recall | 66.6233% | 64.8244% | 65.4746% | -1.1487 | +0.6502 |
| micro_f1 | 63.4011% | 63.9513% | 63.9839% | +0.5829 | +0.0327 |
| macro_precision | 57.7707% | 57.8379% | 57.9733% | +0.2026 | +0.1354 |
| macro_recall | 60.1895% | 56.2556% | 57.2359% | -2.9537 | +0.9802 |
| macro_f1 | 58.1356% | 56.0209% | 56.8995% | -1.2361 | +0.8786 |
| weighted_f1 | 63.0712% | 63.4254% | 63.4158% | +0.3446 | -0.0095 |
| rare_f1 | 55.0248% | 44.7456% | 48.8913% | -6.1335 | +4.1457 |
| macro_average_precision | 61.3012% | 59.5187% | 59.7749% | -1.5263 | +0.2562 |

历史值从封存Validation数组重算并核对原始metrics文件，不使用prompt中的四舍五入值作为结果。Macro AP是全部27类定义有效的逐标签Average Precision平均，基于完整概率排序；逐标签mean_positive_probability只在gold-positive样本上取均值，不是预测正例均值。

## 2. Grief / Relief / Pride与其余Rare标签

| 标签 | Val support | A0 F1 | A1 F1 | A1-S F1 | A0 AP | A1 AP | A1-S AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| grief | 13 | 57.14% | 0.00% | 0.00% | 57.14% | 43.15% | 40.54% |
| pride | 15 | 59.26% | 50.00% | 58.33% | 61.41% | 53.86% | 55.35% |
| relief | 18 | 22.22% | 9.52% | 32.00% | 28.39% | 17.97% | 26.02% |
| nervousness | 21 | 39.02% | 46.15% | 47.37% | 39.30% | 35.89% | 40.54% |
| embarrassment | 35 | 59.15% | 61.29% | 59.02% | 66.13% | 65.59% | 61.94% |
| remorse | 68 | 77.02% | 76.82% | 74.48% | 71.00% | 68.12% | 73.45% |
| fear | 90 | 71.35% | 69.43% | 71.04% | 78.10% | 76.63% | 78.33% |

F1涉及固定阈值决策，AP检查概率排序；二者变化分别解读，不因F1上升就宣称排序能力恢复。Rare-F1仍是七个固定标签的未加权平均，未删除grief、未调专属阈值；少量支持样本导致的波动需要如实保留。

## 3. 从头训练后，是否再次饱和

| 时点 | 首次clip | 后续任一text clip | 最终>0.99 | 半径均值 | 半径IQR | 辅助/BCE共享梯度比中位数 | 共享cos中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.00% | 0.00% | 0.00% | 0.175727 | 0.015439 | 0.0722 | -0.0038 |
| epoch1 | 0.00% | 23.45% | 56.55% | 0.987100 | 0.013253 | 1.0553 | +0.5017 |
| epoch2 | 0.00% | 46.21% | 73.79% | 0.990502 | 0.006031 | 0.5491 | +0.5537 |
| epoch3 | 0.00% | 59.31% | 73.79% | 0.992195 | 0.005712 | 0.4474 | +0.6795 |
| epoch4 | 0.00% | 51.03% | 75.17% | 0.992318 | 0.005492 | 0.4723 | +0.6714 |
| best_reload | 0.00% | 51.03% | 75.17% | 0.992318 | 0.005492 | 0.4723 | +0.6714 |

最后一轮：first clip=0.00%，later clip=51.03%。选中轮：later clip=51.03%。近边界比例与实际proj截断分开报告；若后续重新饱和，不能用此前只读s=.03探针结论代替从头训练的真实结果。
每个时点复用相同10组train-only探针，160个位置/145个独立ID；不是随机全train分布估计。梯度是eval状态下全部27类BCE与0.1aux对共享encoder的局部梯度，非单情绪梯度。记录器仅观察，不额外缩放；每轮无更新诊断恢复Python/NumPy/CPU/CUDA RNG和所有module.training状态。

## 4. 如何解释、是否继续

主要指标判定：输入缩放缓解了当前A1的部分性能退化，但尚未超过普通BCE baseline。
Macro改善且Rare未下降，存在继续确认该候选的初步理由；仍需看AP、几何轨迹和后续独立对照，不能直接宣称稳定收益。
若几何改善但Macro/Rare未改善，必须保留负结果，不能把数值改进写成识别收益。即使主指标超过A0，也只是一个已观察历史Test后的探索性Validation pilot；一个seed、一个checkpoint选择协议不能证明显著性、泛化或Hyperbolic优于Euclidean。
输入缩放还改变辅助梯度强度和方向，gamma固定不代表有效梯度贡献固定。因此性能变化不能只归因于径向信息恢复。初始化与诊断RNG检查只能覆盖记录的状态，不宣称逐训练步历史dropout mask已逐个核验。
下一步最值得考虑的单个对照是容量和辅助目标尽可能匹配的Euclidean Auxiliary Control，用于区分双曲几何与额外监督/参数作用；是否继续应同时看Macro/Rare/AP与实际饱和轨迹。如果本轮主要指标未改善，停止继续搜索缩放系数或gamma。
本轮没有自动启动欧氏对照、多seed、新尺度或正式Test；任何后续实验需要独立冻结方案。

## 5. 证据位置

INITIALIZATION_PARITY.json、SCALED_INITIALIZATION_PARITY.json、smoke/、geometry/、training_log.csv、batch_log.csv、checkpoint_selection.json、checkpoint_reload_verification.json、validation_predictions.npz、validation_per_label.csv、protected_before.json、source_manifest.json与SUCCESS.json。
