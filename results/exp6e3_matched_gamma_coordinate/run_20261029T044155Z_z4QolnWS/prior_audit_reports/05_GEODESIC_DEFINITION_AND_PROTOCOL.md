# A1-G定义及预登记协议

距离实现：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6e_geodesic_alignment/gmath.py:9；新辅助项：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6e_geodesic_alignment/gmath.py:29；仅新模型实例安装：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6e_geodesic_alignment/gmodel.py:4。
保留旧BCE与同一个Soft-target CE Tensor，仅把−mean_positive(zᵀe)替换为+mean_positive d_c(z,e)。
d_c(x,y)=acosh(1+2c||x−y||²/[(1−c||x||²)(1−c||y||²)])/sqrt(c)，c=1。
实现等价式2/sqrt(c)·asinh(sqrt(c)||x−y||/sqrt[(1−c||x||²)(1−c||y||²)])，恒等式cosh(2asinh t)=1+2t²。
球坐标输入仍FP32，距离局部FP64运算保持autograd，再转换为FP32参与正标签平均/辅助batch平均。没有epsilon、边界投影、距离归一化、平方、温度或detach。相同点返回0并使用vector_norm的零次梯度约定，不声称未平方距离在重合点可微。
L_G = BCE + .1·mean_i(C_i + mean_{j∈P_i} d_1(z_i,e_j))。固定标签不接收梯度，新距离须向z及共享encoder回传。
原A1 scale1、seed42、27类、30587/3834/3821、maxlen50、batch16/eval64、4epochs/7648步/warmup765、AdamW5e−5/原WD分组、clip1、FP32/eager、Validation Macro-F1 earliest tie、阈值.30。无新Test、额外seed、欧氏或28类训练。
先与原A1初始化哈希/初始train-eval logits/z/decoder输出/模型参数组核对，CPU重放并核对A1四轮完整ID顺序；A0只核对其真实保存的四个首batch，绝不要求不存在的完整顺序文件。
训练前在fresh及A1 best的相同train-only10cohorts比较旧/新loss与梯度。训练前预登记严重停止门：新total梯度norm或aux绝对尺度超过同forward旧值10倍，或者非有限、球外、错误梯度路径，则停止并报告，不修改权重。训练后的旧坐标目标可能趋近零，epoch探针保留比例但不再以该比例单独停机；非有限/球外/错误梯度仍停止。
即使通过门也完整报告尺度差异和实际clip频率；单独权重敏感性只能在另行定义的实验中进行，不能混入本次单变量对照。
