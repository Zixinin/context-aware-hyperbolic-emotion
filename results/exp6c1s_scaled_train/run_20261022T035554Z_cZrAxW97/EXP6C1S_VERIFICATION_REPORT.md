# Exp6C1S pilot verification

PASS：旧源文件/模型/结果前后哈希；相同raw snapshot/tokenizer/27标签与canonical划分；共享及HNN/decoder初始化；缩放前后分类logits与RNG；原Formula A值及梯度；train-only固定probe及无更新/RNG恢复；原AdamW/scheduler/7648步；四轮Validation最优且平局取早；完整3834条重载；保存数组指标/AP重算；新Test=0。
运行前后保护检查和数值结果均保留。完整远端verify检查checkpoint字节；供下载审核包可省略大checkpoint，由REVIEW_EXPORT.json明确列出省略项，不能把省略项描述为已本地重载验证。
