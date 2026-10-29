# Exp6C-1D：执行预览（已按用户请求授权，只读诊断）

仅写入 Ubuntu `results/exp6c1d_hyperbolic_diagnostics/run_<timestamp>_<unique>/`。

1. 校验 A0/A1 各自 SUCCESS seal、源码/配置/checkpoint 和历史保护清单中的旧文件。旧清单不允许新增文件的全局扫描不适用于新授权的诊断目录；不修改旧清单，只校验所有历史条目及两个原run的精确封存。
2. 审计完整 A1 loss/训练代码/配置/训练日志/选轮/验证报告；重算构造输入的辅助公式及对z/u梯度，检查smoke实际证据。若映射、公式或梯度断开则停止。
3. 读取 A0/A1 已保存的 validation predictions，验收全部3834 IDs/gold/27类/固定0.30；全部27类和七个预定义Rare输出support、预测正数、P/R/F1、TP/FP/FN、正样本平均概率、AP及配对转移。所有有序标签对的描述性漏检/误报计数完整导出。不选阈值、不运行新推理。
4. 读取原作者树/配置/训练标签代码和发布向量。还原TSV child→ancestor与reverse=True；区分cover边/闭包边，计算float64 c1球距离、范数、同父/异父分布和Rare最近邻。原标签训练脚本仅作为文本读取，不导入/执行。
5. 读取A1 best，model.eval()/FP32、不使用autocast。使用现有seed42设置构建模型，不创建新seed实验。train-only固定10组，每组<=16条：单正、多正、训练最高频7类、每个预定义Rare标签。按SHA256(Exp6C1D-v1:canonical ID)排序取样；允许组间重复，所有IDs/标签和规则归档。
6. 对同一forward分别autograd.grad(BCE)和autograd.grad(0.1*aux)，在共享encoder/embedding/early0–3/middle4–7/final8–11/逐12层/classifier/HNN/decoder统计norm、比例、cosine。参数梯度归约FP64；不连接的模块明确标记。禁止标准optimizer.step和Tensor.backward，.grad必须一直为None，完整state与固定buffer前后hash必须一致。
7. 总预算：最多160个训练样本cohort位置（重叠后unique更少），最多10个forward和20次autograd.grad；墙钟30分钟上限，沿用原环境，无安装/下载。GPU与既有实验锁均就地只读打开。
8. 报告各阶段证据和限制；单个后续建议仅为提案。保守筛查：若至少一半定向train cohort同时aux/BCE梯度norm>1且cosine<−0.1，则提案A预注册gamma0.03；否则优先提案D配对seed复核。这是预先公开的启发式，不是显著性判据，不使用test选方向或系数。拿到全部证据后仍需人工审阅解释。
9. Test文件仅参与原seal/hash核验；不打开历史test预测数组分析、不重新计算test指标、不新增test推理。几何独立价值仍需未来容量/目标匹配的Euclidean Control，不在本周执行。
10. 成功/失败均发送原有ntfy通知。完成后CPU独立校验诊断seal及validation统计、距离矩阵，生成审核包（无checkpoint）。失败保留现场并停止；不自动重试。

这是已授权现有产物的机制诊断，不复现任何论文表格，不会输出论文复现成功的声明。
