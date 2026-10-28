# Exp6E-1：Fresh train-only 梯度尺度校准

**只完成校准，没有训练。** 本轮CPU只读提取上轮已封存的fresh初始化真实梯度统计；没有重新forward/backward、optimizer、Validation/Test推理或F1选择。
来源：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/results/exp6e_geodesic_alignment/run_20261027T115121Z_clE7F1kT；仅fresh_initial参与数值校准，old A1-best不参与系数或方向判定。
本轮计算结果由完整精度CSV产生，非六位小数终端表估算；文件完整性、train身份、初始化和无更新证明均复验。

## 唯一系数及预定规则

`gamma_G = 0.02091538317504222`（double hex `0x1.56ad79ae165d2p-6`）。
令a_b=∇encoder mean(CE−dot)、g_b=∇encoder mean(CE+distance)。每组精确匹配系数r_b=.1||a_b||/||g_b||。
gamma_G=exp(median_b(log(r_b)))；10个原有固定cohorts等权，偶数中位数取两个中央log的平均。没有候选grid、裁剪系数或按方向/性能择优。
保存梯度已经乘.1：实现使用r_b=.1*(saved_old_aux_norm/saved_new_aux_norm)，避免漏乘或重复乘.1。
同batch BCE不变，所以匹配aux范数也匹配aux/BCE范数比；这并不匹配梯度方向，也不保证逐组精确匹配或训练全程匹配。

|cohort|单组精确系数|固定gamma下aux/旧aux|旧aux/BCE|新aux/BCE|BCE与旧aux cos|BCE与新aux cos|旧与新aux cos|
|---|---:|---:|---:|---:|---:|---:|---:|
|single_positive|0.009219|2.268677|0.192654|0.437070|-0.047146|0.030078|0.092546|
|multi_positive|0.008449|2.475357|0.189107|0.468108|-0.002213|0.053995|0.039229|
|common_top7_train|0.011059|1.891239|0.233975|0.442503|-0.011587|0.030808|0.063503|
|rare_grief|0.020621|1.014283|0.442536|0.448857|0.001791|0.031490|0.406810|
|rare_pride|0.022409|0.933337|0.435658|0.406615|0.055652|0.044206|0.250703|
|rare_relief|0.019928|1.049560|0.458791|0.481528|0.110777|0.079054|0.383059|
|rare_nervousness|0.021976|0.951757|0.460577|0.438357|-0.085621|0.041559|0.263928|
|rare_embarrassment|0.026214|0.797876|0.551153|0.439751|0.048007|0.039266|0.168473|
|rare_remorse|0.021214|0.985918|0.446296|0.440011|-0.095708|0.020984|0.273124|
|rare_fear|0.024090|0.868205|0.500351|0.434407|-0.191789|-0.061401|0.382314|

## 方向、边界与局限

新辅助与BCE出现负cos的组：rare_fear。正系数不会改变这些夹角；没有因方向不好删除组或另挑系数。方向不是正确性证明，也不等于该单标签的梯度/未来F1变化。
固定样本有重叠：160位置、145个独立train IDs，分层覆盖稀有类；等cohort校准不是整套训练分布的无偏估计。eval/dropout关闭下得到的匹配不保证train/dropout与未来更新轨迹匹配。
这是对同一fresh前向的梯度Gram重加权：||BCE+gamma*g||²由范数及cos精确重构。GLOBAL_GRADIENT_IMPLICATIONS.csv中的clip1标记只是局部反事实，不能冒充真实训练clip比例或Adam更新幅度。
s=1保持不变，已有首次映射严重截断问题仍存在；降低aux系数不会修复前向截断。本轮没有改标签、曲率、投影、距离、BCE、CE公式或分类头。
注意：gamma_G乘整个CE+distance，因此辅助CE系数由.1变为0.02091538317504222；CE内部公式与targets保持，但实际权重不同。已有统计未分别保存CE与distance的梯度，不能宣称两分量各自匹配。
## 后续仅为待批准协议

若后续批准，L=BCE+gamma_G*mean(CE+mean_positive d_1)，使用本文件冻结的唯一常数，不逐batch/epoch重估。
命名为A1-G-C：测地目标+整个aux梯度尺度匹配；与A1比较改变了对齐形式和系数，不能称严格仅替换内积。
原预训练snapshot独立初始化；27类原IDs/划分/标签顺序、seed42、s=1、batch16/eval64、4epochs、7648步、warmup765、AdamW5e-5、原WD、clip1、阈值.30、Validation Macro-F1选轮/earliest tie均保持。
Validation报告Macro/Micro/Rare-F1、P/R、Macro AP和全部27标签/固定Rare7；记录CE/距离分量、梯度方向/强度、实际训练clip及首/后续投影截断。无新Test、更多seed、s=.03或系数搜索。
FROZEN_GAMMA_PROPOSAL.json冻结计算候选，不授权训练。需要审阅本报告后决定是否训练；现有原Exp6E暂停记录不修改、不绕门。
若结果不佳保留负结果，不连续搜系数；后续几何独立价值仍需matched Euclidean control。
