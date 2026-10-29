# Exp6D — 03_DISTANCE_WEIGHT_NUMERICAL_AUDIT

PARTIAL：已审阅源码、协议和保存的训练样本数值；真实冻结模型参数梯度检查待 Ubuntu 执行。不得据此宣称审计已全部通过。

固定train-only样本：10×16 cohort positions，145 unique IDs。分层覆盖单正、多正、常见及7稀有情绪，cohort有交集。不能用这些分层样本估计整套训练分布，也不能根据单个cohort的局部梯度断言某一情绪整轮训练获益。
原始逐标签距离矩阵保存在各numeric目录all_pairwise_distances.npz，逐样本/逐组统计在samples.csv/cohorts.csv与NUMERICAL.json。
|冻结模型/时点|距离mean|median|std|p95|max|距离与正标签数corr|F分类/BCE范围|N1分类/BCE范围|
|---|---|---|---|---|---|---|---|---|
|A1_best|11.196999|11.321894|0.554534|11.947737|12.468785|0.021258|9.5775–10.5346|0.9788–1.0305|
|A1S_initial|6.527501|6.562835|0.525312|7.189147|7.532845|-0.108788|5.3589–6.3702|0.9996–1.0002|
|A1S_best|10.899947|11.024035|0.768497|11.844086|12.487885|0.003168|9.4704–10.5695|1.0014–1.0499|

F绝对尺度约十倍与权重的相对离散程度是两个问题。必须同时查看加权前后最大样本/前10% loss share；只有权重尾部相对基准增加集中度才支持“distance outlier主导”，不能把本来BCE难例占比较大误认为距离造成。单正/多正完整percentiles与每个rare/common cohort统计均保存。
去重采用固定cohort顺序首次出现，梯度与损失用原cohort batch不改写。GPU batch上下文导致同ID轻微差异被记录：历史A1-S距离最大差约2.48e-5、BCE约8.20e-8、aux约1.19e-6；近边界距离对FP32坐标敏感。未平均不同batch logits来伪造一致性。
FP64 Torch Eq6与独立逐pair NumPy实现核对。距离精度改变仅用于本次数值审计，不擅自改已冻结实验代码。
Exp6B历史28类Formula A使用emotion weight=.9d、neutral-only weight=1；训练记录emotion均权约9.288/9.894/9.863/9.864。这里只把它作为过去真实实现证据，27类所有保留样本均有情绪，不能照搬neutral-only相对弱化的解释。
以上未使用Test分数选方案；不构成识别性能提升证据。
