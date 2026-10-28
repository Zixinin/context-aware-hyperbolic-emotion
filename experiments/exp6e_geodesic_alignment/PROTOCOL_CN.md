# Geodesic Alignment Ablation — Exp6E / A1-G

这是在原Exp6C1 A1上做的一次seed42探索性消融。只把对齐项从负的正标签坐标内积改为正的正标签测地距离，不使用A1-S的.03输入缩放，也不使用N1距离加权BCE。无新标签预训练、neutral向量、28类训练、额外seed、Euclidean训练、Test推理或权重搜索。

固定协议：RoBERTa-base同snapshot/tokenizer，原27类多标签顺序，train30587/validation3834/test3821；maxlen50；batch16/eval64；4轮/7648步/warmup765；AdamW5e-5、WD.01且原bias/LayerNorm免WD；clip1；FP32/eager；固定sigmoid阈值.30；Validation Macro-F1 strict greater-than选轮，平局取最早。每个正式模型从预训练权重独立初始化，smoke和旧checkpoint不作为正式初始化。

令q_ij=y_ij/Σ_jy_ij（当前每例至少一个非neutral正标签），C_i=−Σ_jq_ij log softmax(u_i)_j。

旧A1：A_old=mean_i[C_i−Σ_jq_ij z_iᵀe_j]，L_old=BCEWithLogitsLoss(mean)+.1 A_old。

新A1-G：A_G=mean_i[C_i+Σ_jq_ij d_1(z_i,e_j)]，L_G=BCEWithLogitsLoss(mean)+.1 A_G。保留同一个旧CE及BCE Tensor。正标签均值后完整batch平均，所有正标签参加。没有显式负标签对齐；CE的softmax分母仍存在标签竞争。

d_c(x,y)=acosh(1+2c||x−y||²/[(1−c||x||²)(1−c||y||²)])/sqrt(c)，本实验c=1，曲率−1，有效球半径1。使用等价的2/sqrt(c)·asinh(sqrt(c)||x−y||/sqrt[(1−c||x||²)(1−c||y||²)])。FP64局部几何运算有live autograd，再转FP32参与loss；模型和optimizer保持FP32。不加epsilon、不裁剪距离、不重新投影固定标签、不平方、不归一化或detach距离。重合点距离0，采用vector_norm提供的零次梯度约定；未平方距离在该点不可微，不用gradcheck宣称其可微。

Phase1核对现有代码、文件SHA256、原A0/A1封存证据、发布标签字典/矩阵、固定参数、源码输入路线、27/28真实样本统计及未来欧氏控制。公开发布bin精确预训练日志不齐全，不能完整认证历史无泄漏；对所审标签树入口没有发现实例划分读入路径。若当前代码/文件显示实际实例泄漏、球外点、曲率不一致或梯度路径错误则停止。

Phase2在fresh及原A1 best上的同10个固定train-only cohorts、160位置/145独立ID做无更新探针。分别测BCE、.1×坐标aux、.1×geodesic aux和总loss的四模块梯度/连接/cosine，观察首次/后续投影clip、半径、CE/坐标/距离项。RNG、mode、模型state及.grad必须恢复。训练前严重停止门：aux绝对尺度或total梯度norm超过同forward旧值10倍；任何非有限、球外或错误路径均停止。超过门只报告并建议单独权重敏感性，不自动改gamma或scale。

通过后做两个丢弃的optimizer更新和64条Validation exact reload。Phase3新Python进程重置seed42，从相同预训练参数正式训练一次。每轮全ID序列核对原A1已保存的完整顺序。旧A0只核对真实保存的四个首batch，不补造其从未保存的完整顺序。初始化shared/hyper哈希、历史初始分类logits及新旧train/eval logits/z/decoder输出全部匹配，参数量/optimizer分组不变。历史每步随机mask未完整归档，不能声称已逐步直接观测其一致性。

训练后每轮无更新探针仍检查非有限/有效域/梯度路径。此时旧坐标目标可能接近零，所以旧/新loss比值只作描述，不因该比值自动改系数或停机；实际梯度全局clip及所有分量逐步记录。完整3834 Validation选轮与重载，重新核对ID/gold/logits/阈值预测和逐类指标。

四项报告、当前真实公式、新定义/代码、测试与稳定性记录、结果路径、Validation A0/A1/A1-G全表、27类变化及中文总结/Methodology建议均写入独立run。任何负结果完整保留。原Exp6C/D的源码、数据、checkpoint和日志只读，运行前后哈希保护。不安装或修改依赖、PyTorch、CUDA、驱动，不自行SSH。

本轮只授权一个seed42 A1-G pilot。完成后停止；优于坐标内积（若发生）只支持当前目标式对齐的比较，不证明双曲几何独立优于matched Euclidean方法。Validation已被反复探索，不作为最终泛化证据。

## 本地预审发现：先停在真实无更新诊断

本地已封存的原A1-best、scale1训练缓存显示，common_top7_train辅助项由1.049647变为12.584868（11.989616倍），single_positive由1.247669变为13.013430（10.430195倍），超过前面预登记的10倍尺度门。没有因此修改门值或gamma。leaf-z的坐标梯度约增193–316倍，它不是共享RoBERTa参数梯度，也不是Adam更新倍数。

当前交付先在Ubuntu核查真实原始文件，并完成fresh与原A1-best两个时点的全部无更新探针，再以退出码3/DIAGNOSTIC_SUCCESS.json报告SCALE_REVIEW_REQUIRED。它表示审计诊断完成、正式训练暂停，不是训练失败，不写pilot SUCCESS。普通异常仍以其他非零码报告。

诊断只读验收使用launch.sh inspect ROOT；正式pilot若确实完成才使用verify。停止门触发时不执行两个smoke optimizer更新、正式训练或Test，也不自动重试。已经准备的正式训练代码不会绕过该门。本轮A1-G的F1仍未知，不能填入推测数值。
