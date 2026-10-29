# Exp6D — 02_GRADIENT_PATH_AUDIT

PARTIAL：已审阅源码、协议和保存的训练样本数值；真实冻结模型参数梯度检查待 Ubuntu 执行。不得据此宣称审计已全部通过。

|模块|原版距离加权分类|原版aux|A1/A1-S BCE|A1/A1-S aux|候选F/N分类|
|---|---|---|---|---|---|
|共享encoder|有；距离权重视为常数|有，经pooler/HNN|有|有，经raw h/HNN|有；重加权BCE梯度|
|原版pooler_fc|有|有|不存在|不存在|不新增|
|分类头|有|无|有|无|有|
|HNN projection|无|有|无|有|无|
|aux decoder|无|有|无|有|无|
|固定标签向量|无|无|无|无|无|

`source_review/upstream/hypbert.py:25`核实detach；`source_review/exp6c1_roberta27_hypaux/a1model.py:54`核实分类与aux的分叉位置。候选每例权重w时，`d L_class / d g_ik = w_i(sigmoid(g_ik)−y_ik)/(B×27)`；正标签梯度方向均为吸引。

距离每次随当前模型重算，但当次反传没有dw/dtheta。这是stop-gradient自适应样本加权，不是距离完全与训练无关，也不是端到端距离最小化。

真实参数梯度以`GRADIENT_PATHS.csv`、`MODULE_GRADIENTS.csv`和`NO_UPDATE_PROOF.json`为准。Mac缓存只有输出，不冒充共享encoder/HNN参数梯度已算出。局部FP64距离检查与合成导数检查不能替代这一步。
