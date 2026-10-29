# 本地诊断工具QA：真实Ubuntu诊断待执行

13项CPU检查通过，完整输出见LOCAL_TESTS.log：配对TP/FN/FP/TN变化、0.30含等号、零支持/AP缺失、非法概率拒绝；Poincare距离的已知解析值、出球拒绝；实际75边层级的方向/约简/拼写异常及合成有向环拒绝；train-only固定cohort可复现及覆盖七Rare；梯度norm/cosine/合并norm与设计不连接、autograd不更新；不调用train/step/backward/predict的静态检查；合成validation/原树/随机向量的报告全流程与审核包独立重算/篡改拒绝。

真实A0/A1本地冻结源码manifest再次通过；七个缓存原作者源码hash匹配原A1冻结引用。Bash与Python语法检查通过。没有本地真实RoBERTa forward、真实checkpoint梯度或新的模型训练。合成测试不构成Ubuntu真实机制观察。

真实旧checkpoint/validation数组/训练日志/固定向量都在Ubuntu。诊断shell仅由用户执行，参数前后hash、完整27类验证集结果及实际cosine只有该阶段可产生。不得提前宣称EXP6C1D完成或几何有效/无效。
