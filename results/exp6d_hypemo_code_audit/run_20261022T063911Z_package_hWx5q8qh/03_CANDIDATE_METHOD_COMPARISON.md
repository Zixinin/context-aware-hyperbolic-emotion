# Exp6D — 03_CANDIDATE_METHOD_COMPARISON

PARTIAL：已审阅源码、协议和保存的训练样本数值；真实冻结模型参数梯度检查待 Ubuntu 执行。不得据此宣称审计已全部通过。

|方案|分类项|原版忠实程度|全局尺度|距离梯度|主要局限|
|---|---|---|---|---|---|
|A1/A1-S|mean ell|只含适配aux|分类系数1|无分类distance|尚未检验距离权重|
|F|mean .9*dbar*ell|更接近公开分类weight；CE仍已换BCE，aux亦变|缓存约9.5–10.6倍|detach|同时改变相对样本与分类/aux平衡；clip可能不同|
|N09，仅诊断|.9 mean(dbar/mean_batch dbar)*ell|额外归一化偏离原版|仍全局缩小10%|整个ratio detach|额外改变A1分类系数|
|N1|mean(dbar/mean_batch dbar)*ell|明确为FYP尺度控制适配|mean weight1；loss仍可因协方差变化|整个ratio detach|跨batch比例不保持，batch依赖|
|固定mu0，仅讨论|mean(dbar/mu0)*ell|额外归一化偏离原版|训练距离变化会漂移|mu0常数，dbar detach|初始/最佳缓存均距约6.53/10.90，不宜假设后期仍均值1|

没有温度、裁剪distance、learned scale、gamma搜索或per-label阈值。这里的多方案只读比较不是并行训练计划。最终只在真实模块梯度核实后提出一个下一步，见04文件。
