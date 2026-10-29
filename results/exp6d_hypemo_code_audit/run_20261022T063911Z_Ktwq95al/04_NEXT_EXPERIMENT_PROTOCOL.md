# Exp6D — 04_NEXT_EXPERIMENT_PROTOCOL

完整只读审计；没有训练或新 Test 推理。

当前推荐状态：`ONE_PROPOSED_PILOT_A1S_N1`。

所有后续训练仍需单独用户授权；本目录没有训练runner。

如果完整审计通过且不触发严重梯度门，唯一候选为：**Exp6D-1，A1-S为主控制，N1距离归一化加权BCE pilot**。这里是预登记候选，不是当前已经批准/实施的训练。

只改变分类样本权重：`w=stopgrad(dbar/mean_batch(dbar))`；总loss=mean(w*ell)+.1*aux。固定s=.03，原固定27向量、A1辅助loss、head与全部训练协议不变。N1明确放弃raw绝对distance倍数，检验batch内relative distance-weighting。不得称为完整原版复现。保留A0作为背景参考，但主要因果比较为同s的A1-S。

保持同snapshot/tokenizer、train30587/validation3834/test3821及所有canonical IDs、27顺序、maxlen50、seed42、batch16/eval64、4轮7648步/warmup765、AdamW5e-5/WD分组、linear scheduler、clip1、FP32/eager、固定.30、Validation Macro-F1 earliest tie；从同raw pretrained初值独立训练，不从已训A1-S best续训。额外距离计算/归一化不消耗RNG；构造与初始分类logits做匹配检查，不能宣称整个训练轨迹相同。

Validation主要指标Macro-F1，联合Rare-F1、Macro AP判断方向，Micro-F1辅助；按27类P/R/F1/AP检查grief/relief等，不能因某个稀有标签坏就删除它或换指标。只把单seed结果称pilot，不声称显著或普适。报告weight全分布、outlier贡献、module梯度/BCE-aux cos/norm、clip频率、首次及后续projclip与radius轨迹、4轮与best reload证据。

风险：距离可能主要反映半径/标签范数而非语义困难度；N1相对信号可能很弱；多正aux仍有softmax竞争；后续HNN仍可能clip；梯度重分配可能伤害稀有类，归一化对batch组成敏感。当前cached数值不证明未来性能。

若严重不稳定，唯一下一步改为更简单的training-only只读距离/梯度定位，**不启动训练**。无论哪种结论，均不自动跑新Test或更多seed；不同时修改s/gamma/head/hierarchy。后续几何独立价值需要容量、aux目标、训练预算、标签信息与尺度尽可能匹配的Euclidean control。
