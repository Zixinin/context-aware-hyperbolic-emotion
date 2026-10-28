# Exp6C-1S-Train 实现审计与预注册方案

本轮由用户指定执行：s=0.03、seed42、四轮，Validation-only pilot。Mac编辑/CPU合成测试，用户复制命令到Ubuntu D3020-A3；Codex不连接SSH、不代跑远端训练。对照的是用户提供的训练prompt，第二份附件中的情绪支持文字不作为额外实验授权或已验证的结果证据。

## 唯一模型干预与调用链

`a1model.A1Model.forward -> 冻结Roberta27.forward -> encoder.last_hidden_state[:,0,:] -> hyper.encode(h) -> TextBaseModel.encode -> HNN.encode -> proj(expmap0(proj_tan0(x,c),c),c) -> HypLinear -> HypAct -> LinearDecoder`。

`scaled_model.install_scale`仅在新构造模型的`hyper.encode`上安装一次wrapper，令`original(0.03*h)`。A1Model.forward本身原样导入，因此普通分类分支接收的h不变。wrapper不增加参数、buffer、state_dict key或随机draw，不修改HypEmo源码，不改变expmap/proj/曲率。重复安装直接报错。每次重载都从原snapshot构造相同模型，再安装同一wrapper，再严格载入state_dict。

仍用27×100固定原始Parrott标签向量；不缩放标签、不解冻标签。c=1、HNN num_layers=2、dim100、dropout0.1和decoder初始化均由冻结A1的build函数创建。

## 损失公式、归约与梯度

设B为batch大小，K=27，q_ik=y_ik/(sum_j y_ij)，z_i为文本球坐标，e_k为固定标签向量，u_i为原tangent decoder logits：

`BCE = mean_(i,k) BCEWithLogits(logit_ik, y_ik)`；

`aux = mean_i [ -sum_k q_ik log softmax(u_i)_k - sum_k q_ik (z_i · e_k) ]`；

`L = BCE + 0.1 * aux`。

直接使用冻结`a1model.loss_components`，不改正标签均值、batch均值、BCE元素均值，也不做距离加权、class weighting、归一化或aux截断。BCE→共享encoder/分类头；aux→共享encoder/HNN/decoder；aux不连接分类头，固定e无梯度。缩放的链式导数自动进入辅助梯度，0.1保持不变。

## 全部匹配协议

roberta-base同snapshot/tokenizer，27个标签顺序不变，30587/3834/3821原划分，neutral-only排除且混合neutral保留其他情绪；maxlen50、seed42、batch16、FP32/eager、四轮7648steps、AdamW5e-5/WD0.01/原betas与epsilon、线性scheduler/warmup765、clip1.0、固定sigmoid>=0.30。优化器分组和每步归一化/反传/clip/step顺序直接使用冻结A1代码。每轮完整train ID顺序与A1原记录逐项比较。

每轮完整Validation按最高Macro-F1选轮，严格`>`，同分取早。完整3834条Validation重载，IDs/gold/logits/阈值预测/指标一致。**本轮不执行新Test**，覆盖此前讨论中“一次最终Test”的旧建议；当前训练prompt对此有明确新规定。历史Test文件可以作为旧seal的字节哈希输入，但不计算/比较新的Test结果，数据加载后即移除live data字典中的Test记录。

## Smoke 与初始化

1. 环境、A0/A1/Exp6C-1D/1S封存和全旧文件清单核验。沿用原Python/包/4090，不安装或升级。
2. 复用原A1初始参数审计：fresh shared tensor逐项对应A0历史lr=0 smoke权重；shared/HNN初始hash及初始train logits hash对应冻结A1。wrapper前后参数、分类train/eval logits与初始forward RNG消耗相同。
3. 只读载入A1 best，重放s=1和s=.03既有train探针：s=1/scale03的z、decoder和分类logits逐项对应已保存缓存；Loss/各模块梯度对应旧诊断，rtol1e-6、loss atol1e-8、梯度atol1e-9。该模型随后释放，绝不作为正式初始化。
4. Fresh scaled模型检查single/multi-positive/mixed-neutral三组target、原Formula A值及梯度、BCE、模块连接和固定标签；固定train探针记录几何、梯度比和cosine。
5. 独立fresh模型做两个smoke optimizer步及64条Validation保存/重载，验证整个数值通路。Formal再次独立seed42构造，不续训smoke模型。

smoke失败停止。通过后按此次用户授权自动进行唯一正式seed42；不再询问重复许可。

## 每轮几何监测不会改变训练RNG

固定沿用1S的10组train探针，共160个cohort位置、145个独立ID；初始、epoch1–4、best reload都检查。普通forward hooks会被原HNN直接调用`.forward`绕过，因此复用已封存的manifold方法记录器。模型自身已缩放，记录器固定s=1 identity，不再乘第二次0.03。

分别保存缩放前h norm、实际expmap输入norm、expmap输出norm、首次真实proj clip、HNN后续任一text proj clip、最终半径范围/均值/std/IQR及>0.99比例。bias独立记录。每组用autograd.grad计算全部27类BCE与0.1aux对共享/分层/HNN/decoder/分类头的梯度；不调用optimizer、不积累.grad。

监测进入eval，并在finally恢复Python、NumPy、CPU/CUDA RNG及每个module.training状态；前后模型state hash相同且.grad=None。总表按ID首次出现去重，分cohort表保留重叠。训练中重新饱和只记录，不改尺度或gamma。

不声称已经逐步核验A1与A1-S的历史dropout mask；能证实的范围是初始化及本轮诊断的RNG状态。

## 结果与停止

从A0/A1封存Validation数组重算旧F1，与原metrics和prompt四位小数核对；新模型输出完整micro/macro P/R/F1、weighted-F1、固定七Rare-F1、Macro AP及27类support/TP/FP/FN/P/R/F1/AP/预测正例数/gold-positive平均概率。AP仅对有正例标签定义，正式Validation应27类全部定义；缺支持不伪造AP。

无论结果如何，生成六份指定审计/总结/比较/几何/Rare/验证报告，按真实Macro、Rare、AP和饱和轨迹给出结论。性能改善不等于双曲几何独立价值；额外参数、监督、梯度强度仍需匹配Euclidean Control隔离。不同seed、其他尺度、gamma、新Loss、新层级和新Test全部禁止自动启动。

启动器复用历史锁的只读descriptor；一旦此实验任何run已有FORMAL_TRAINING_CLAIM，拒绝第二次formal attempt。开始/成功/失败ntfy均启用，通知失败不覆盖训练退出码。smoke上限60分钟，formal上限8小时；超时保留失败，不重试。开始与结束全量核对旧文件，源文件及旧checkpoint不写入。

审核包省略大`.pt`，完整模型仍保存在Ubuntu；REVIEW_EXPORT明确列出省略文件和包内字节hash。完成标记只在smoke、四轮、Validation重载、CPU复核和导出全部成功后打印。
