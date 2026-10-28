# A1-G-C：校准系数后的测地对齐 seed42 Validation结果

本轮使用测地距离对齐，并将整个辅助项系数固定为校准值gamma_G=0.02091538317504222；普通BCE、CE内部公式、固定27向量、s=1与其余训练设置保持。辅助CE的实际系数也改变，因此不是仅替换内积的单变量消融。新Test=0。

|模型|Alignment|Micro-F1 %|Macro-F1 %|Rare-F1 %|Macro AP %|
|---|---|---:|---:|---:|---:|
|A0|none|63.4011|58.1356|55.0248|61.3012|
|A1|negative coordinate inner product|63.9513|56.0209|44.7456|59.5187|
|A1-G-C|positive unsquared geodesic distance|63.4171|58.6162|56.3096|60.0204|

全部数值均为同一3834条Validation、27标签、seed42、阈值.30；没有把历史Test与本轮Validation混表。A0/A1读取封存数组重算，A1-G-C完整四轮后按Validation Macro-F1选轮并exact reload。

## A1-G-C相对A1的变化（pp）

|指标|变化|
|---|---:|
|micro_precision|-2.6494|
|micro_recall|+1.8639|
|micro_f1|-0.5341|
|macro_precision|+1.5135|
|macro_recall|+3.8175|
|macro_f1|+2.5953|
|weighted_f1|-0.2000|
|rare_f1|+11.5640|
|macro_average_precision|+0.5016|

预定主指标Macro-F1相对原A1提高，变化+2.5953pp。全部27类P/R/F1/AP/TP/FP/FN及逐类贡献见CSV，包含负向结果。

## 逐标签变化

|情绪|support|F1变化pp|AP变化pp|
|---|---:|---:|---:|
|admiration|488|-0.7887|+0.1047|
|amusement|303|+0.9918|+0.9277|
|anger|195|+0.5683|+2.3560|
|annoyance|303|-3.1681|-1.9425|
|approval|397|+0.0805|+1.0443|
|caring|153|-0.1835|+2.0687|
|confusion|152|-0.2535|-0.2633|
|curiosity|248|-1.8018|+0.6822|
|desire|77|-1.7033|-6.4106|
|disappointment|163|+6.2009|+3.7732|
|disapproval|292|+1.1849|-0.1719|
|disgust|97|+0.7614|+0.9028|
|embarrassment|35|-3.7146|-0.5821|
|excitement|96|-0.1942|+1.1987|
|fear|90|+0.0887|+0.1786|
|gratitude|358|-0.2151|-0.7964|
|grief|13|+52.6316|+9.4041|
|joy|172|+0.2566|-1.7628|
|love|252|+0.3406|+1.2224|
|nervousness|21|-9.3117|-0.7731|
|optimism|209|-3.0184|+3.2260|
|pride|15|+10.8696|-1.2653|
|realization|127|-6.8966|-2.9081|
|relief|18|+30.4762|+0.7307|
|remorse|68|-0.0916|+3.6033|
|sadness|143|-1.0740|-3.7359|
|surprise|129|-1.9626|+2.7327|

## Loss尺度与数值稳定性

training_log.csv记录4轮BCE/CE/旧坐标项诊断/新距离项/aux及clip频率；batch_log.csv保存全部7648步。probes各目录包含同forward下旧/新aux与BCE的四模块梯度、cosine、边界截断及RNG/state不变证明。
gamma_G=0.02091538317504222来自fresh train-only预定log-median规则，训练前冻结、全程不调。它匹配的是初始共享aux梯度典型尺度，未匹配方向，也不保证未来训练尺度匹配。若有改善，只能解释为测地目标与校准权重的联合设置效果。
## FYP Methodology建议

把原方法准确写为HNN/Poincare分支 + tangent-space decoder soft-target CE + coordinate inner-product alignment；原A1对齐不是geodesic distance。
A1-G-C单独列为探索性的测地目标与整个aux梯度尺度匹配实验，给出c=1、正标签均值/完整batch均值、gamma_G=0.02091538317504222与FP64局部距离计算。辅助CE系数也变，主分类仍是RoBERTa欧氏分类头。
27类neutral-only过滤实验属于preliminary条件任务；计划中的28类任务及neutral mask要另述，不能当作已完成。
说明固定发布标签先验及完整预训练日志不足的来源限制，披露近边界与clip；单seed、反复探索Validation、无新Test，不声称显著或最终泛化。
测地距离式对齐优于坐标内积（若观察到）也不等于双曲几何优于matched Euclidean方法。四项审计与明确否定结果均应保留。
本轮到此停止，没有新seed、权重搜索、欧氏训练、28类重跑或正式Test。

## 固定Rare7（全部Validation）

|标签|support|A0 F1 %|A1 F1 %|A1-G-C F1 %|A1-G-C AP %|
|---|---:|---:|---:|---:|---:|
|grief|13|57.1429|0.0000|52.6316|52.5527|
|pride|15|59.2593|50.0000|60.8696|52.5994|
|relief|18|22.2222|9.5238|40.0000|18.6980|
|nervousness|21|39.0244|46.1538|36.8421|35.1126|
|embarrassment|35|59.1549|61.2903|57.5758|65.0051|
|remorse|68|77.0186|76.8212|76.7296|71.7197|
|fear|90|71.3514|69.4301|69.5187|76.8098|

## 真实训练投影截断与全局梯度裁剪

|epoch|首次text clip|后续text任一clip|最终半径>.99|平均半径|global clip1|最大未裁剪norm|
|---|---:|---:|---:|---:|---:|---:|
|1|100.00%|61.98%|0.00%|0.237962|10.41%|3.993003|
|2|100.00%|100.00%|0.00%|0.298978|8.89%|3.713246|
|3|100.00%|100.00%|0.00%|0.349370|12.97%|10.819259|
|4|100.00%|100.00%|0.00%|0.379075|19.87%|2.364274|

上表来自各轮全部30587个真实训练positions，不是eval小探针估计。ProjectionMonitor只观察原proj输入/输出，不改Tensor/RNG；其train/eval forward逐元素一致性在初始化parity中核验。
s=1的首次截断问题保持并如实报告，不能因为系数降低就声称几何饱和已修复。固定train-only probe给出额外局部梯度/方向诊断，不用这些probe更改gamma。

## 固定train-only探针的共享梯度诊断

|时点|weighted aux/BCE中位数|BCE与aux cos中位数|weighted CE norm中位数|weighted distance norm中位数|CE-distance cos中位数|
|---|---:|---:|---:|---:|---:|
|fresh_initial|0.439881|0.035378|0.033552|0.241499|0.073547|
|epoch1|0.040020|0.187305|0.006508|0.026682|0.292843|
|epoch2|0.033687|0.261889|0.009492|0.027270|0.331743|
|epoch3|0.029418|0.392349|0.011363|0.027118|-0.087269|
|epoch4|0.032199|0.275620|0.012365|0.028783|0.243397|
|best_reload|0.029418|0.392349|0.011363|0.027118|-0.087269|

每时点使用同10cohorts/160位置/145独立train IDs、eval模式；方向匹配与范数匹配分别解释。CE/distance分量梯度本轮实际重新测量，不能从此前校准摘要推断这些分量各自匹配。
