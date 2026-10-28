# Exp6D-1 实现审计：A1-S + N1，seed42，Validation-only

本文件是实现与验收合同，不代表GPU smoke或训练已经完成。真实完成以新run的SUCCESS、退出码、3834条重载和冻结复核为准。

唯一模型改动位于 `nmodel.loss_components/install_n1`：在新构造的冻结A1-S同结构实例上保留其原forward，只替换loss聚合；没有新增nn.Module、参数、buffer、state key、随机抽样、dropout、head或缩放。原辅助loss Tensor直接保留。

公式：P_i为全部真实非neutral正情绪，d_i为这些Poincaré geodesic的均值。w_i=stopgrad(d_i/mean_current_batch(d))；ell_i=mean_27 BCEWithLogits；L=mean_batch(w_i ell_i)+.1 L_aux。无alpha=.9，无distance clipping、epsilon、temperature、learned scaling或rare weighting。距离计算采用已完成Exp6D审计的FP64，再将归一化权重cast FP32。正标签掩码与27列按名称对齐固定buffer，向量不缩放、不训练。球外、非有限值、空正标签、无效均距直接停止，不自动修复。

`ncommon.check_sources/check_old_seals`：按字节核对A0/A1/A1-S、Exp6D及官方HypEmo/Formula A来源；Exp6D实际run固定为run_20261022T063911Z_Ktwq95al，必须完成且推荐A1-S/N1。A1-S固定run_20261022T035554Z_cZrAxW97。不调用旧实验会拒绝新目录的全局保护检查；使用本轮新保护清单，排除本轮两个新目录，保护其他实验源码、旧结果、数据、预训练snapshot、Exp6A/B与官方源码。runtime_cache、pycache等运行缓存明确不纳入研究输入清单。

`nrun.initialization`：复用A1的初值验收，核对A1-S封存的shared/head/HNN/decoder初值与初始logits哈希；安装N1前后train/eval logits、z、decoder输出、aux以及RNG消耗严格一致。正式训练重新seed42、重新从同raw snapshot构造；不从A1-S best、smoke或任何训练权重续训。

`nprobe.run_probe`：固定10组160位置/145独立train IDs；s=.03由冻结scaled_model恰好应用一次，ScalingTrace取s=1，仅观察。分别autograd.grad测ordinary BCE、N1 BCE、.1aux、combined参数梯度，检查相加关系与模块连接。共享encoder/classifier可获分类梯度；HNN/decoder只获aux梯度；标签无grad。no-update探针恢复所有Python/NumPy/CPU/CUDA RNG与module modes。首次clip、后续任一text projclip、最终radius与>0.99分开记录。分类梯度对ordinary BCE的范数比达到10或非有限即停止，不更换公式。此停止条件并非可调训练系数。

smoke先在A1-S旧best的新加载实例上重放已完成Exp6D的10个cohort、四模块N1/BCE/aux/combined梯度，值与范数按预声明容差核对；该实例没有optimizer，不用于正式初始化。随后fresh模型检查固定探针与single/multi、B1、B11，执行两次丢弃的smoke optimizer更新并对64条Validation重载。smoke失败不进入正式训练。

数值容差：参考loss/weight atol1e-7、rtol1e-6；解析BCE导数atol1e-8、rtol1e-5；旧Exp6D模块梯度norm replay atol1e-7、rtol1e-5；完整Validation重载logits要求逐元素exact；普通flat-mean BCE与先27列mean再batch mean允许FP32归约舍入atol1e-7、rtol1e-6。后者不改变辅助loss或偷偷改变分类系数。

`nmodel.train_step/nrun.train`：沿用A1-S的(loss*B).backward→grad/B→clip1→原AdamW.step→原linear scheduler.step→zero_grad顺序。固定seed42/4epochs/7648step/765warmup/5e-5/WD分组/FP32/eager/maxlen50/batch16/eval64。检查每轮全部30587个训练ID顺序与A1-S一致。最后一批为11条，用实际11条均距归一化，禁止drop_last或固定除16。

每batch记录distance/weight/Loss/梯度裁剪，逐轮保存全部30587个训练位置的原始distance、weight、BCE、positive_count。epoch分位数从这些样本合并计算，single/multi分别报告；不平均各batch的p95。N1 Validation weighted loss按当前eval batch64（最后58条）归一化，因batch依赖仅作诊断；checkpoint依据完整Validation Macro-F1，不依据weighted loss。

`nrun.predict_validation`只接受固定smoke64/完整3834条Validation。live数据从load_dataset拿到manifest后立即删除Test记录；本轮无Test预测分支。阈值沿用固定sigmoid>=.30，四轮strict greater-than，平局取早。完整Validation重载IDs/gold/logits/预测/aggregate与27类P/R/F1/AP/TP/FP/FN全部复算。A0/A1/A1-S历史Validation数组重新读取，不能照抄prompt小数。

结果封存前必须通过nrun.verify，任何失败保留日志和FAILURE，不打印最终成功标记、不自动重试。仅一个正式seed42 claim；不启动Raw F/其他seed/新s或gamma/新Test。新结果只有探索性Validation证据，不能证明几何独立优势。
