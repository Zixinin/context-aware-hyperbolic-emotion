# Exp6D — 04_EXECUTIVE_SUMMARY_CN

完整只读审计；没有训练或新 Test 推理。

完整只读数值与真实参数梯度检查通过；唯一建议是以A1-S为控制、保持s=.03，下一步单独批准N1距离加权BCE seed42 pilot。没有实施训练，也没有新Test。

**Q1 原版与Exp6C有哪些重要差异？**

按相关性排序：①核心分类distance weighting缺失；②单标签CE/原数据改为27多标签BCE/更多训练样本；③aux点积原来batch求和，当前batch均值；④原版pooler/head与HNN输入取点不同；⑤双optimizer和每batch StepLR改为统一AdamW/linear warmup，batch/length/预算不同；⑥原版eval和Weighted-F1实现存在已知问题，当前已按独立协议修正。以上是源码事实，哪项造成成绩变化尚未逐一控制证明。

**Q2 之前是否测了原版核心方法？**

Exp6B Formula A确实包含distance-weighted BCE，但为BERT28且含neutral专门规则，不能当当前匹配RoBERTa27结论。Exp6C-1和A1-S没有分类distance weighting，仅测试适配aux；后者还单独改变输入尺度。

**Q3 CE改BCE得到/失去什么？**

BCE允许同一句多个情绪各自为真，保留全部正标签；softmax CE只允许一个总概率分布，不能直接把一条多标签样本挑一个标签。mean-27 BCE、单标签CE的数值/梯度单位不同，原alpha/gamma不可照搬解释为相同强度。距离取所有真阳性均值，不只挑一个。

**Q4 当前aux是否忠实？**

继承了HNN、固定27标签向量、decoder CE和ball-coordinate dot思想；但multi-positive q目标和dot的batch sum→mean改变了目标与比例。aux不是纯测地distance loss；softmax的多正标签仍竞争概率。分类与aux的梯度路径分开，固定向量与distance权重不反传。

**Q5 为什么论文涨分而当前没有？**

可核实的是实验任务/数据/目标/head/优化/评估不同，当前几何存在后续clip；距离加权机制也尚未在匹配RoBERTa27上测试。它们都是待检验解释，不是已证明原因。原版released-F1还有指标差异，不能直接把原版数字当标准Weighted-F1对照；更不能说双曲一定能改善稀有情绪。

**Q6 如何受控恢复distance weighting？**

先完成本轮真实梯度与冻结审计。通过后唯一预登记候选为同s=.03的A1-S控制，新增N1 `dbar/当前train-batch平均dbar` detached样本权重。raw F缓存会将分类loss放大约十倍，N1明确控制共同倍数，aux.1和其余设置不变；它是FYP适配，不冒充原版严格复现。最终推荐状态：`ONE_PROPOSED_PILOT_A1S_N1`。严重放大门触发则先只读定位，不推进训练。

**Q7 什么才支持几何本身有用？**

需要容量与aux目标匹配的Euclidean control，同时匹配标签先验、参数量、训练预算、初始化与loss尺度；之后再用适当多seed看稳定性。当前aux/权重pilot无法隔离“几何”与额外监督、先验和随机轨迹的作用。此轮不跑额外seed，也不根据Test调参。
