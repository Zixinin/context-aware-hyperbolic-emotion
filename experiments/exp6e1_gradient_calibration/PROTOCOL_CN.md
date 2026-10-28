# Exp6E-1：fresh-initialization、train-only无更新梯度尺度校准

授权范围是校准一个系数并回报，尚未授权正式训练。本目录没有训练入口。保留原Exp6E及A0/A1所有代码、checkpoint、结果与暂停门。

## 数据与测量来源

唯一输入运行：`results/exp6e_geodesic_alignment/run_20261027T115121Z_clE7F1kT`。
只使用其`probes/fresh_initial/MODULE_GRADIENTS.csv`的完整精度真实梯度统计；不使用original_A1_best选择gamma。该CSV由真实fresh RoBERTa的torch.autograd.grad产生，含共享encoder、classifier、HNN、decoder的BCE与两种辅助项范数、cosine和连接数量。两种辅助梯度均已乘0.1。

先复验整份旧诊断artifact seal、原Exp6E/五套冻结代码、初始参数哈希/无更新证明、已审核终端六位小数参照及实际train.tsv SHA。确认固定10×16组全属30587条canonical train IDs；160位置/145独立样本，不删方向不好的组。旧文件只读；不重新forward/backward，不需要GPU，不导入torch，不打开Validation/Test样本或预测数组，不安装依赖。

## 唯一预定规则

每组b定义：a_b=∇encoder mean(CE−dot)，g_b=∇encoder mean(CE+distance)，BCE梯度为h_b。

逐组等范数系数：r_b=0.1 ||a_b||/||g_b||。

全局系数：gamma_G=exp(median_b(log r_b))。10组等cohort权重，偶数中位数取两个中央log平均。使用log中位数保留乘法对称性、降低单个组对比例估计的影响；不声称它是全训练集最优权重。

CSV的两个aux_norm已经乘0.1，故代码使用r_b=0.1×saved_old_norm/saved_new_norm。不添加epsilon；范数为0、非有限或哈希/路径异常即停止。没有grid、上/下界截断、gamma取整、按Validation/Test或方向择优。保留double精度实际常数及hex值。

同一batch BCE不变，匹配aux范数也匹配aux/BCE范数比。报告所有组的剩余匹配误差与范围；不声称每组精确匹配，不声称未来训练全程匹配。

## 方向和训练贡献

同时报告cos(h_b,a_b)、cos(h_b,g_b)、cos(a_b,g_b)；正gamma不能改善或改变方向。不删负cos组，不把cohort整体梯度说成某个单标签梯度。

利用已保存的范数及内积重构各模块及全部模块的||h+gamma g||；这是相同fresh前向的代数重加权。全局clip1比例只能写作这些固定eval探针的反事实标记，不能称为训练裁剪频率或Adam实际更新幅度。

本轮没有额外forward/backward，所以不会新生成CE/距离分量的单独梯度；原统计只保存合成aux，不能宣称两个分量各自匹配或方向各自正确。

## 后续实验边界

冻结的是整个新辅助项系数：L=BCE+gamma_G mean(CE+positive distance)。辅助CE的内部公式、q与归约不变，但其实际系数从0.1改变为gamma_G。若未来要固定CE系数0.1仅匹配距离项，应单独定义实验，本轮不混用。

建议后续命名A1-G-C（测地目标与整个aux的梯度尺度匹配），不称严格只换内积的单变量消融。s=1、同预训练初值、原27类及seed42协议保持；s=1已知初始径向截断仍未解决。s=.03组合、Euclidean control、多seed与权重搜索均不属于本轮。

成功只代表一个固定校准值已由真实训练样本梯度统计计算及核验。FROZEN_GAMMA_PROPOSAL.json不授权训练；先审阅方向、匹配范围和边界风险，再决定是否批准一次四轮seed42 Validation pilot。没有正结果预设。
