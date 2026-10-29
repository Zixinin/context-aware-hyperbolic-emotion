# 01 Loss implementation audit

原始源码与 checkpoint/结果封存 hash 均通过；27-label、30587/3834/3821 与 A0 配置逐项匹配。只打开 train/validation 数据进行分析；历史 test 产物仅作封存 hash 核验，不重算其指标或推理。

对样本 i，m_i=sum_k y_ik >=1，q_ik=y_ik/m_i：
```math
L_{BCE}=\frac{1}{27B}\sum_{i,k}\mathrm{BCEWithLogits}(g_{ik},y_{ik})
H_i=-\sum_k q_{ik}\log\mathrm{softmax}(u_i)_k,\quad D_i=\sum_k q_{ik}\langle z_i,e_k\rangle
L_{aux}=\frac1B\sum_i(H_i-D_i),\quad L=L_{BCE}+0.1L_{aux}
```

z 为原 HNN 的100维 c=1球投影，u 为原 tangent decoder 的27-way logits；正式分类仍经 A0独立sigmoid。
BCE无距离权重、类别权重、pos_weight或样本权重。多正标签先按1/m平均；多标签数不作为额外样本权重，但会改变目标分布/梯度方向。
e 是register_buffer的detach/clone，requires_grad=False，不在optimizer参数中。aux经decoder/HNN到共享encoder；不直接连接分类头。BCE不连接额外分支。
训练的(mean_loss*B).backward再grads/=B与mean_loss梯度等价；之后对全部训练参数clip_norm=1。新增分支会改变整体裁剪系数以及后续dropout随机消耗，两者均是完整新增分支的影响，不能隔离为几何。

## 作者机制、多标签适配与待检验假设
HNN/球映射/tangent decoder、原固定标签向量来自冻结作者源码；正标签均值与batch均值沿用Exp6B Formula A多标签适配。
dot项是球坐标普通点积，不是负Poincare距离。辅助CE的label identity本身不使用固定向量的距离，固定向量通过dot项进入目标。
CE对u_ik的导数为(softmax(u_i)_k-q_ik)/B。多正标签中p_ik>1/m_i时，该真实正标签logit会收到向下的CE梯度；这是分布目标的数学性质，不等于把该标签误标为负类，也不是已证实的性能下降原因。
dot项对z的导数为-mean_{k in P_i}(e_k)/B；多个正标签的向量平均可发生方向抵消。这与几何距离对齐不能混同。
普通softmax CE的多正目标最低值为log(m_i)，故其标量尺度仍随标签数的目标熵变化；按m平均不意味着单/多正样本的辅助标量必相同。
本阶段未发现标签映射、预注册公式或断梯度的实质性问题。Formula A数值及z/u梯度用冻结实现再次通过构造输入核对；真实checkpoint梯度/方向见Phase2。

## 证据定位
evidence/A1_a1model.py: loss_components、A1Model.forward、build、train_step；evidence/A1_a1run.py: initial_parity、formula_and_gradient_case、train。
完整数学规范、原训练配置、选轮、日志和smoke证明均保存在evidence/；精确SHA见01_AUDIT_EVIDENCE.json。
