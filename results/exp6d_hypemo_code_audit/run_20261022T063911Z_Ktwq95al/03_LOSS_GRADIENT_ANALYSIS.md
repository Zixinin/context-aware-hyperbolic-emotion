# Exp6D — 03_LOSS_GRADIENT_ANALYSIS

完整只读审计；没有训练或新 Test 推理。

冻结checkpoint、eval模式，torch.autograd.grad逐项计算；没有optimizer或backward累积，没有修改.grad。详见MODULE_GRADIENTS.csv、GRADIENT_PATHS.csv、CLIPPING_IMPLICATIONS.csv。

|模型|候选|encoder分类梯度/BCE中位数|head分类梯度/BCE中位数|总norm中位数|probe中会触发clip1比例|
|---|---|---|---|---|---|
|A1|F|10.317124|10.320186|9.995099|1.000|
|A1|N|1.003819|1.001078|1.296613|0.900|
|A1S|F|10.094692|10.082032|8.901099|1.000|
|A1S|N|1.010571|1.010138|1.163832|0.500|

N1严重放大停止门：任意A1-S probe总norm超过同probe baseline的10倍，先建议更简单只读检查，不推荐训练。此门在Ubuntu前写定，只用于阻止推进，并非调loss系数。当前触发：False。


候选分类项应只更新共享encoder/classifier；HNN/aux decoder来自aux的梯度保持同一frozen forward的值。权重detach并不阻断BCE经encoder的正常路径。每个module的connected tensor数量、范数、加权分类对普通BCE的cos、对aux的cos和合成norm均记录。不能从“scalar loss放大10倍”直接推出“Adam参数更新放大10倍”；自适应二阶矩、全局clip会共同改变效果。

probe clip比例只是当前checkpoint/eval小样本的反事实诊断，不是训练期间实际clip频率；train dropout与参数轨迹会变。真实训练norm、clip、球半径与aux/BCE梯度比必须将来另报。F与N1在同一batch分类梯度方向相同（仅共同正标量），但它们与aux叠加的方向和全局clip后的相对贡献不同。

普通单标签CE与mean-27 BCE的单位不同，另有原aux点积sum→mean；不将alpha.9/gamma.1解释成固定9:1监督强度。当前F会强烈减小aux/分类的实际比例，N1控制共同倍数而保留batch内部相对distance信号。
