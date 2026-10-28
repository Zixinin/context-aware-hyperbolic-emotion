# Exp6E-2 / A1-G-C：校准系数后的测地对齐seed42 pilot

用户在审阅Exp6E-1校准后要求下一步训练命令与代码。本包执行环境/封存检查、真实fresh梯度replay、两步丢弃smoke，然后自动进入一次完整四轮训练。没有再次等待人工许可的阶段。失败即保留日志并停止，不重试。

## 唯一冻结设置

L = mean_{B×27}(BCEWithLogits) + gamma_G mean_i(C_i + G_i)。

C_i = −Σ_j q_ij log softmax(u_i)_j；q_ij=y_ij/Σ_j y_ij；G_i=Σ_j q_ij d_1(z_i,e_j)。所有正标签平均再完整batch平均。距离复用Exp6E已验证的未平方Poincare测地距离/asinh等价式（FP64局部距离、FP32模型和归约）。

gamma_G = **0.02091538317504222**，double hex `0x1.56ad79ae165d2p-6`。从Exp6E-1成功封存的完整精度统计及原始fresh模型梯度复验，训练全程不修改或重估。

这是“测地目标 + 整个aux梯度尺度匹配”的联合设置，命名A1-G-C。相对A1同时改变对齐目标与aux系数，不能称为严格只替换内积。CE的内部公式/目标保持，实际系数也由.1变为gamma_G。单独CE与距离梯度本轮重新记录。

## 其余控制

原raw pretrained RoBERTa snapshot/tokenizer独立初始化，绝不从旧A1/A1-G权重继续训练；27非neutral情绪顺序、固定27×100 Parrott先验、c=1、原HNN与分类头、s=1均保持。

train30587/validation3834/test3821原canonical IDs与分割；maxlen50、seed42、batch16/eval64、4epochs/7648步、warmup765、原AdamW5e-5/WD分组/linear scheduler、clip1、FP32/eager、阈值.30、Validation Macro-F1 strict greater-than（平局早轮）。正式训练四轮完整ID顺序与原A1核对。

Test仅限原数据协议/哈希保护，模型不接收Test样本。验证包含两步丢弃smoke的64条、各轮完整3834条、best checkpoint完整reload；不根据Test选择模型/系数。

## 开始训练前

检查已存在环境/硬件，不安装/升级任何依赖。复验旧审计、校准seal、γ、固定标签、runtime、snapshot及历史保护清单。

新构造的模型必须与原A1的共享/HNN初始化哈希和历史初始分类logits完全匹配；系数安装及投影观察器不得消耗RNG、增加参数/buffer或改变train/eval的logits/z/decoder Tensor。

使用同10组train-only fresh探针重测四模块梯度，并与原始完整精度统计按γ/.1缩放replay（norm rtol3e-5/atol3e-8，cos rtol3e-5/atol3e-6）。相同标签和已知轻微负cos组全部保留。校准匹配不代表方向正确。

技术停止条件：哈希/初始化/顺序/replay不一致，非有限值，球外点，错误梯度路径；fresh探针中加权aux标量或总梯度范数超过原目标10倍也停止。没有修改原Exp6E门或暂停记录；这是新系数的新实验，原.1标量门不被删除或重写。训练后旧目标可接近零，因此其比例只作诊断，技术有限性/球内/梯度路径检查继续有效。

## 真实训练与诊断

每一个训练batch记录loss/BCE/CE/距离/weighted分量、全局未裁剪norm与lr；观察原proj方法的输入/输出，记录首次与后续任一text投影clip、最终半径及>.99数目。观察器不改变原Tensor和RNG；初始化parity有逐元素验收。按整轮30587个positions统计实际clip和半径，区别于小样本eval探针。

fresh、每轮末、best_reload使用固定train-only160位置/145独立IDs、eval/autograd探针，记录BCE、旧.1aux、新γaux、γCE、γdistance的模块norm、cos、合成梯度和几何；恢复Python/NumPy/CPU/CUDA RNG、模式、参数/buffer和空.grad。不调用optimizer，不按探针调γ。

s=1的已知初始截断保持并实测，不声称校准修复该问题。s=.03的组合是未来独立实验，本轮不做。

## 结果与停止

按Validation Macro-F1选best，完整重载核对logits/IDs/gold/预测/指标。输出A0/A1/A1-G-C同seed42同27类Validation的Micro/Macro/Weighted/Rare-F1、P/R、Macro AP、27逐类与固定Rare7；保留所有下降结果。记录实际训练clip与边界轨迹。

单seed、重复探索过的Validation属于探索性结果。即使改善也不证明负曲率独立价值；仍需matched Euclidean Control。本轮没有新Test、其他seed、γ搜索、欧氏训练或28类重跑。

结束后发训练完成通知；错误发停止通知。通知投递失败不会篡改任务退出码。成功必须四轮7648步、reload和保存产物核验通过；只有校准/smoke结束不发正式完成通知。
