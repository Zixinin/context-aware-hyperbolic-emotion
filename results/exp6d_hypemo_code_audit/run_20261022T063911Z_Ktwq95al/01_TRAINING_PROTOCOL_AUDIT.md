# Exp6D — 01_TRAINING_PROTOCOL_AUDIT

完整只读审计；没有训练或新 Test 推理。

论文§4.1/4.2：原单标签非neutral数据23485/2956/2984；RoBERTabase、main Adam1e-5/WD.01、双曲侧描述RiemannianAdam.01、维度100。Table1说明效率计时batch16，未充分确立精度实验所有batch与选轮细节，不将其猜成config100或构造默认16。

公开config与Exp6A实际profile：batch100、20轮、alpha.9/gamma.1、默认seed1234。Exp6A五个预定seed见`EXP6A_EXECUTION_FILES.json`（Ubuntu生成）及`evidence/Exp6A_local`的原始终端证据；每轮235更新，共4700。实际原版旧环境torch1.13.1+cu116，FYP torch2.6.0+cu118，不能当同运行时严格因果对照。

`source_review/upstream/hypemo.py:89`：StepLR每batch而非每epoch。首轮235步后branch LR=.01×.5^23≈1.1921e-9；主Adam没有此scheduler。优化器名RiemannianAdam不代表所有参数在球上更新：`source_review/upstream/optimizers/radam.py:99`依`ManifoldParameter`分支决定；本HNN/decoder权重为普通nn.Parameter。源码group step还在参数循环中增长，不能将它描述成标准torch.Adam等价实现。保留源码，不修复或重跑。

原版已知评估问题：`source_review/upstream/hypemo.py:107`只有no_grad，没有eval；f1_score预测与真值反传使Weighted-F1按预测support加权；valid_loss仅最后batch。`source_review/upstream/train.py:45`每轮Test并记录best Test，entry不存可重载checkpoint。Exp6A因此维持Partial reproduction；其released-F1不能替代论文标准Weighted-F1。Exp6B Stage A明确修正eval/metric/selection，属于单独分支证据，不回写Exp6A。

FYP冻结协议：数据30587/3834/3821，27顺序、snapshot/tokenizer、raw first token独立dropout head、maxlen50、seed42、batch16/eval64、4轮7648step、warmup765、AdamW5e-5/WD.01且bias/LayerNorm免WD、clip1、FP32/eager、阈值>=.30、validation Macro-F1 earliest tie。A1新增branch参数79627；A1-S唯一改s=.03。A0/A1历史各一次Test；A1-S及本审计新Test均0。审计不以任何Test值选公式。

训练预算比较必须同时看样本、步数与学习率；“原版20轮而我们4轮所以必然没训够”未经证实。
