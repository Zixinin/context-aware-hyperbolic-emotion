# Exp6D — 03_MULTILABEL_DISTANCE_WEIGHTING_SPEC

完整只读审计；没有训练或新 Test 推理。

这是待验证的multi-label extension，不是原论文宣称的公式；本轮只在冻结模型上计算候选标量与梯度，没有实现未来训练分支。

令P_i={k:y_ik=1}，q_ik=y_ik/|P_i|，d_ik=d_H(z_i,e_k)，dbar_i=sum_(k in P_i)d_ik/|P_i|，ell_i=mean_27 BCEWithLogits。所有样本|P_i|>=1，不需要neutral-only规则。aux完全沿用冻结A1的a_i，a=mean_i a_i。

**F：** `wF_i=stopgrad(.9*dbar_i)`；`L_F=mean_i(wF_i*ell_i)+.1*a`。

**N1：** `m_B=mean_j dbar_j`，`wN_i=stopgrad(dbar_i/m_B)`；`L_N=mean_i(wN_i*ell_i)+.1*a`。

N1的归一化尺度来自**当前training minibatch**，每次forward重新计算；无滑动统计或可训练参数，不使用validation/Test，不冻结常数。分子分母都detach，不经当前模型距离/均值反传。它可写成`.9*dbar_i / mean_j(.9*dbar_j)`，alpha.9代数抵消，实际分类系数保持A1的1、aux保持.1。

这是明确偏离原版的方案，不能命名为“严格原版距离权重复现”。它保留同一batch样本间相对权重w_i/w_j=dbar_i/dbar_j；不同batch的共同因子不同，跨batch比例不保持；对batch组成敏感，B=1退化为普通BCE。mean(w)=1不保证mean(w*ell)=mean(ell)，差异由weight/loss协方差决定。

额外报告`N09=.9*mean(wN*ell)+.1*a`只用于暴露alpha全局缩小10%的影响，不是另一个推荐实验。未默默把F的.9改为1。未来若选N1，理由是匹配A1控制BCE系数1而非冒充原始alpha/gamma比。

比较过的固定尺度定义：`mu0=mean_C dbar(theta_initial)`，C为预先固定train-only校准集；`w=dbar/mu0`。mu0冻结会保留跨batch相对信号，但训练时半径/距离增长使整体分类尺度漂移。现有145个分层样本初始mean≈6.5275、best≈10.8999，不能把这组分层均值称为全train无偏统计；也不能声称除以初始常数能在后期维持均值1。此方案没有被自动选作未来实验。

FP64 c=1距离采用论文Eq6，不加任意eps、clipping、温度或learned scale。任何向量范数>=1、空正标签、非有限距离直接停止。既有HNN自身proj/tanh数值规则保持冻结。
