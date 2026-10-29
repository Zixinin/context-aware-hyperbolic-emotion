# A1 本地实现审计：已能回答的部分

本报告是诊断准备，不代表Ubuntu上的完整Exp6C-1D已完成。

## 公式与归约

从冻结 `a1model.py::loss_components` 还原：

```math
m_i=\sum_{k=1}^{27} y_{ik},\quad q_{ik}=y_{ik}/m_i
L_{BCE}=\frac{1}{27B}\sum_{i,k}\mathrm{BCEWithLogits}(g_{ik},y_{ik})
H_i=-\sum_k q_{ik}\log\mathrm{softmax}(u_i)_k
D_i=\sum_k q_{ik}z_i^T e_k
L_{aux}=\frac1B\sum_i(H_i-D_i)
L=L_{BCE}+0.1L_{aux}
```

z 是原HNN球投影；u 是tangent decoder logits。正式输出仍用原RoBERTa分类器的27独立sigmoid。向量e使用register_buffer(detach().clone())，不是参数。辅助CE与dot经HNN作用到共享encoder，不直接连接分类头；BCE不经过HNN/decoder。无距离权重、类别权重、重采样或新增样本权重。

每条样本对真实正标签平均，再对batch平均。多个正标签不会线性放大样本权重，但其均匀目标熵log(m_i)和方向会变化，不能说多正/单正Loss的有效尺度完全相同。

## 几何到底出现在哪里

HNN含expmap0/Möbius线性/球投影，decoder经logmap0回切空间。固定标签向量进入的是球坐标普通点积，而非负双曲距离。CE的类别目标仅用标签身份；它间接依赖双曲分支表示，但不直接利用标签间双曲距离。

因此A1不是“让分类logits直接按双曲距离打分”，不是对BCE作双曲距离加权，也不是纯粹的几何单因素对照。

## 两条可检验的机制假设

1. CE对u_ik的梯度为(p_ik−q_ik)/B。多正标签中p>1/m的真实正标签会收到向下梯度。这是辅助分布目标的性质，不能把它误报为目标标签编码错误；也尚未证明会损害正式分类Recall。
2. dot对z的梯度为−mean(e_positive)/B。多正向量可能抵消或形成中间方向。它的影响取决于实际向量和共享encoder梯度，需要测量，不凭几何直觉下结论。

历史训练clip的是全部参数总梯度，新分支可能改变共享梯度的整体裁剪系数；HNN dropout还会改变后续随机数轨迹。二者都使Micro小涨不能单独归因于几何。

## 当前证据与待补

已完整阅读本地冻结A1 model/lib/run/loss specification，以及原encoder/decoder/base model/config/label trainer/Poincare实现。原作者7个文件SHA与A1 UPSTREAM_REFERENCE一致。

用户此前返回的Ubuntu结果为Micro+0.3036pp、Macro−1.3028pp、Rare−7.1921pp；这是历史观察，本轮不再按test选择标签或调参。

**待Ubuntu机器证据：**当前冻结状态复查、逐轮真实loss和裁剪统计、全部27类validation转移与七Rare贡献、best处组件/分层梯度cosine、固定向量实际距离。当前不宣称梯度冲突、某类被固定标签挤走或几何无效。
