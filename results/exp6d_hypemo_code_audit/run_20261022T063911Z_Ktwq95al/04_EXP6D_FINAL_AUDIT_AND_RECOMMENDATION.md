# Exp6D — 04_EXP6D_FINAL_AUDIT_AND_RECOMMENDATION

完整只读审计；没有训练或新 Test 推理。

状态：`ONE_PROPOSED_PILOT_A1S_N1`。当前数值表和梯度表见03文件，源码定位见SOURCE_LINE_INDEX.json；所有重要差异逐行见01 CSV。

|历史Validation|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|
|---|---|---|---|---|
|A0|63.4011|58.1356|55.0248|61.3012|
|A1|63.9513|56.0209|44.7456|59.5187|
|A1-S|63.9839|56.8995|48.8913|59.7749|

表中是冻结历史Validation证据，来自已核验A1-S归档；没有新评估模型，也没有使用Test选公式。A1-S虽然部分恢复Macro/Rare，仍低于A0，不能称已证实Hyperbolic有独立收益。

结论一：Exp6C-1/A1-S检验的是普通BCE上增加适配双曲aux，并未检验原版distance-weighted classification；因此负向Macro/Rare结果不能等价为“完整HypEmo方法无效”。Exp6B曾在不同BERT28任务中检验过距离权重，但其neutral规则、数据/head与当前任务不匹配，不能替代Exp6D的控制实验。

结论二：除CE→BCE外，还有分类head/HNN输入取点、aux点积batch sum→mean、optimizer/scheduler/learning-rate/batch/训练轮次、eval/metric与checkpoint选择等差异。A0/A1内部仍保持匹配，原论文对比不能跨这些变化直接归因。

结论三：当前训练样本raw距离权重约十倍分类尺度是已计算事实；原因或收益尚无训练证据。N1是清楚标注的尺度控制扩展，不能默称忠实原版。其是否可进入pilot，取决于真实参数梯度和冻结检查。

已知原版评估缺陷仅审计不修改；尚未发现需要修复当前冻结FYP的实现缺陷。若新检查发现源/协议/label/梯度异常，直接停止，保留FAILURE而不生成完成标记。不因任何Test值推荐系数。

只读完整检查成功时才可推荐04_NEXT_EXPERIMENT_PROTOCOL中的一个候选；仍未实施。进一步随机种子或Euclidean control各需独立授权。本轮停止于审计报告。
