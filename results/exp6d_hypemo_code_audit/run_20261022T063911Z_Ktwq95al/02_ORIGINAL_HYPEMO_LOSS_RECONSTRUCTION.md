# Exp6D — 02_ORIGINAL_HYPEMO_LOSS_RECONSTRUCTION

完整只读审计；没有训练或新 Test 推理。

记分类logits为g，HNN输出z，固定标签球坐标e，辅助decoder logits为u，q_ik=y_ik/|P_i|。

**论文与源码分开。** 论文Eq5是mean CE，Eq7打印sum_i d_i CE_i，并未直接给出当前仓库辅助NLL-dot实现、alpha/gamma或detach细节。下面精确公式来自固定源码，不能冒充论文完整数学表述。

原版：CE_i=−log softmax(g_i)[y_i]；d_i=acosh(1+2||z_i−e_y||²/((1−||z_i||²)(1−||e_y||²)))。

`L_original=.9 mean_i[stopgrad(d_i) CE_i]+.1[mean_i(−log softmax(u_i)[y_i])−sum_i(z_i·e_yi)]`。

分类CE使用reduction=none，距离先乘逐例CE再mean：`source_review/upstream/hypbert.py:115`。距离路径torch.detach→CPU NumPy→逐例arccosh→新FloatTensor→device，因此无autograd：`source_review/upstream/hypbert.py:25`。不能说分类loss沿距离项直接把z拉向e。标签来自pickle固定数据，重新构造tensor，无标签向量训练。

辅助loss：`source_review/upstream/models/base_models.py:132`默认NLL已是标量batch mean；`source_review/upstream/models/base_models.py:140`将(B,1,1)点积沿batch求和，再与NLL相减，最终sum不再增加batch维。即**mean NLL − SUM dot**。在重复相同batch时，NLL不变而dot翻倍；不等于mean(NLL−dot)。`LOCAL_SYNTHETIC_CHECKS.json`与Ubuntu `SYNTHETIC_CHECKS.json`检查这一点。

`z·e`直接是球坐标的普通欧氏点积，不是测地距离/切空间点积。decoder先logmap0(z)，再普通Linear100→27、输出dropout，随后log_softmax/NLL：`source_review/upstream/models/decoders.py:63`。辅助分支包含双曲映射，但不能称每一项都是双曲距离损失。

当前A1/A1-S：`ell_i=(1/27)sum_k BCEWithLogits(g_ik,y_ik)`；`a_i=−sum_k q_ik log softmax(u_i)_k−sum_k q_ik(z_i·e_k)`；`L=mean ell+.1 mean a`：`source_review/exp6c1_roberta27_hypaux/a1model.py:15`。

Exp6B Formula A已批准正标签mean/batch mean；A1仅复用它的aux，去除分类距离权重。这不是原版辅助归约逐项忠实复刻：原点积的每例权重为.1，A1中为.1/B。NLL仍batch mean，所以不能仅靠把gamma乘B同时恢复两项；那会把NLL额外放大B倍。

多正标签softmax aux用q分配概率质量，所有正标签都在目标中，但仍有总概率1的竞争。单正时等价普通CE；多正时最小CE为log|P_i|，正标签不可同时趋近概率1。BCE则每列独立sigmoid，多个正确情绪都可趋近1。需要区分这项目标改变与真正“误把额外正标签当负标签”的实现错误。
