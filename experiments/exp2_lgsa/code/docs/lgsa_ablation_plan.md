# 后续消融：仅计划，不提交

前提：完整 LGSA-GoEmotions seed42 已结束、28类评测已核验，用户同意启动后续实验。

第一组：Standard BERT reference、LGSA full、分别去除 Lg / Lpc / Ln / Ls；一次仅改一个因素，保留其余训练/划分/评测。
第二条路线（之后）：在原 BERT+BCE 的独立副本中逐项加入 LGSA 几何项，明确 label embeddings 取自 classifier rows 还是另设原型，先分析梯度与监督来源。

若候选 coarse hierarchy 本身有问题，先提出一个预注册的 hierarchy 对照，不把所有 losses 和 hierarchy 组合同时搜索。
完整LGSA对BERT的差异包括prompt、GAT、MLM等，不能只凭full model提升认定某个几何loss有效。

现在唯一下一步：获取 HPC 环境与目录的只读状态，准备独立环境及官方 WoS smoke。
不运行消融，不实现 Hyperbolic。
