# Exp6D-1 N1 Loss验收

实现d_i为所有真阳性Poincaré geodesic均值，w_i=stopgrad(d_i/mean_current_batch d)，ell_i=mean_27 BCE，L=mean(w*ell)+.1aux。无alpha、eps、distance clip、temperature、rare weighting。
距离以FP64计算后将归一化权重cast FP32，与已完成Exp6D工具audit_math.candidates逐例/逐模块核对。辅助项直接保留冻结A1-S forward返回的Tensor，保持归约与梯度图；先27维mean再sample weighted mean。
smoke的single/multi-positive、B1和short11样本核对独立参考与sigmoid BCE解析导数。B1权重退化为1；最后训练batch11按自身均距归一化，在每epoch保存数组中复算。
Exp6D已训练A1-S checkpoint只用于只读replay：{'checkpoint': 'A1S best used only for no-update replay', 'all_10_cohorts': True, 'values_match': True, 'all_4_modules_gradients_match': True}。正式训练重新从raw snapshot seed42初始化，没有从此checkpoint续训。
fresh initial probe分类N1/BCE参数梯度比范围：0.997644–1.003805。四个module分别计算ordinary BCE、N1 BCE、.1aux及combined梯度并检验相加关系。距离权重不反传至HNN/decoder；aux不连接分类头；固定向量无grad。
原BCE(flat mean)与逐例mean再batch mean允许预声明FP32求和舍入差异atol1e-7/rtol1e-6；不把这类差异当新方法。权重和距离参考值、真实模块梯度replay及初始化证明详见smoke/、INITIALIZATION_PARITY.json和N1_INITIALIZATION_PARITY.json。
