# Exp6D — 03_LOSS_GRADIENT_ANALYSIS

PARTIAL：已审阅源码、协议和保存的训练样本数值；真实冻结模型参数梯度检查待 Ubuntu 执行。不得据此宣称审计已全部通过。

真实参数梯度待Ubuntu运行；未把输出缓存、合成tensor梯度或旧A1普通BCE梯度冒充候选F/N的模块梯度。



候选分类项应只更新共享encoder/classifier；HNN/aux decoder来自aux的梯度保持同一frozen forward的值。权重detach并不阻断BCE经encoder的正常路径。每个module的connected tensor数量、范数、加权分类对普通BCE的cos、对aux的cos和合成norm均记录。不能从“scalar loss放大10倍”直接推出“Adam参数更新放大10倍”；自适应二阶矩、全局clip会共同改变效果。

probe clip比例只是当前checkpoint/eval小样本的反事实诊断，不是训练期间实际clip频率；train dropout与参数轨迹会变。真实训练norm、clip、球半径与aux/BCE梯度比必须将来另报。F与N1在同一batch分类梯度方向相同（仅共同正标量），但它们与aux叠加的方向和全局clip后的相对贡献不同。

普通单标签CE与mean-27 BCE的单位不同，另有原aux点积sum→mean；不将alpha.9/gamma.1解释成固定9:1监督强度。当前F会强烈减小aux/分类的实际比例，N1控制共同倍数而保留batch内部相对distance信号。
