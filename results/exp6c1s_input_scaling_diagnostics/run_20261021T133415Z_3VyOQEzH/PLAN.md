# Exp6C-1S — 固定 checkpoint 的输入缩放敏感性诊断

当前授权：先做一次不训练的小诊断，判断输入缩放能否减少径向饱和，再决定后续实验。由用户在 Ubuntu D3020-A3 运行；Mac 只编辑代码并执行 CPU 合成测试。没有连接远端，没有申请或启动多 seed 训练。

## 固定方案与预算

- 独立目录 `results/exp6c1s_input_scaling_diagnostics/run_<UTC>_<random>`；A0/A1/Exp6C-1D 只读。
- 固定 A1 best checkpoint、固定原始 27 类标签向量、原 RoBERTa/tokenizer、c=1、FP32、eval，原辅助目标和系数 0.1 不变。
- 复用上次 Exp6C-1D 的 10 组训练探针，每组 16 条，160 个位置、145 个不同 train ID。不按本次结果选样，不新增 Validation/Test 前向。
- 比较预先固定的 s = 1、0.3、0.1、0.03、0.01，仅令传入 `hyper.encode` 的 h 变成 s*h。分类路径不变，固定标签向量不缩放，不移动。
- 60 次整模型前向：10 次未记录的原实现参考前向 + 50 次带记录器的缩放前向。150 次 autograd.grad：每组/尺度计算首次径向导数、BCE 参数梯度、0.1×aux 参数及输入梯度。无 optimizer、无 backward、无参数更新。
- GPU 阶段 timeout 45 分钟，超时终止，不自动重试。前后进行历史源文件、checkpoint、结果、输入的 SHA256 复验；保留原退出码。
- 不安装依赖，不改环境；使用原 `.venv-bert`，离线载入；复用历史锁，GPU 被占用则停止。通知开始、成功、失败，并生成相应审核包。

## 观察什么

1. 每个实际 expmap0/proj/Möbius 矩阵乘/Möbius 加法/logmap0 的输入输出范数。初始映射、HypLinear 与 HypAct 分别记录，bias 与文本分别计数。真实 proj 截断条件与单纯 >0.99 分开。
2. 首次与最终半径的均值、标准差、IQR、范围；初始 clip 比例、后续任一文本投影 clip 比例。总体按 ID 首次出现去重，逐 cohort 的完整数据保留。
3. 初始映射半径对原 h 径向变化的 autograd 导数，以及加权辅助目标对 h 的径向/切向梯度。截断区 FP32 微小残差不视为有意义的敏感性。
4. BCE、aux、0.1aux/BCE 标量比；共享 encoder、分层、分类头、HNN、decoder 的两种梯度范数/范数比/cosine 及连通性。
5. 所有尺度分类 logits/BCE 完全一致；s=1 的 z/decoder logits 与无记录器参考完全相同；s=1 复现原 Exp6C-1D 同 cohort Loss 和全部模块梯度（rtol 1e-6，梯度 atol 1e-9，Loss atol 1e-8）。不匹配即停止，不能跳过门禁。

## 数学与解释边界

c=1 时首次未截断半径为 tanh(s||h||)；FP32 原投影 cap≈0.996，阈值 atanh(cap)≈3.1063。h 范数约22时，s 大于约0.1412可能首先触发截断；实际下游仍可能重新饱和，因此必须观察每次调用。

原目标不变：q_ik = y_ik / sum_j y_ij；BCE 是 B×27 元素平均；aux 是样本平均的 [-sum_k q_ik log softmax(u_i)_k - sum_k q_ik(z_i·e_k)]；L=BCE+0.1aux。

BCE→共享encoder/分类头；aux→共享encoder/HNN/decoder，不连接分类头，固定 e 无梯度。只有辅助路径用 s*h；该路径的 h 梯度含缩放的链式因子 s。

这些尺度是几何探针，不按辅助 Loss 或分类成绩选最优值。固定 checkpoint 在 s=1 下训练，其余尺度是事后干预。缓解截断且梯度可传递仅说明训练候选在机制上可检查；不等于改善 Macro/Rare-F1，更不能证明双曲几何独立价值。接近边界本身不是错误。

完成后停止，返回审核包审阅。**没有自动训练单 seed 或多 seed 的阶段。**

## 主要输出

`05_SCALING_SENSITIVITY_REPORT.md`、`04_scale_summary.json/csv`、逐样本半径与径向导数、逐调用范数/截断表、梯度与 Loss 表、50 个数值缓存、原始调用位置与源文件副本、RNG 前后记录、`BASELINE_PARITY.json`、`NO_UPDATE_PROOF.json`、`SUCCESS.json` 与独立 CPU 复核结果。失败时单独生成 `Exp6C1S_failure_review.tar.gz`，不把失败当成完成。
