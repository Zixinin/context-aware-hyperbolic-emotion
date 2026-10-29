# Exp6D — 02_PARROTT_USAGE_AUDIT

完整只读审计；没有训练或新 Test 推理。

论文§3.3/§4.5：情绪层级用于预训练Poincaré标签表示。源码`train_label_embedding.py`读取label_tree.tsv，`Relations(reverse=True)`处理关系后训练并导出标签向量。论文描述RiemannianAdam，而公开预训练脚本默认opt=rsgd、seed7、2000epochs、dim100、batch16：`source_review/upstream/train_label_embedding.py:53`。发布bin具体训练命令/完整训练日志未建立；不得假定它必然来自当前默认参数。

下游加载`label_tree/go_emotion.bin`里的27×100固定向量，SHA256 `2695b98e160e79c62533d6bee55fdbcfdbbcba87cbcde8ab6db2734dc16ab20a`。FYP通过名字匹配27维，buffer保存固定向量，不创造neutral向量，不对标签重新投影或缩放：`source_review/exp6c1_roberta27_hypaux/a1lib.py:172`。

原版下游用这些向量计算true-label距离和ball dot；A1/A1-S仅用于正标签平均ball dot（decoder是另一个独立可训练27输出层），**不读取显式parent-child边来计算新loss**。树结构的信息间接包含于预训练向量中；aux NLL的decoder权重不等于固定标签矩阵。

label_tree.tsv及发布bin分别哈希核对；固定标签范数约.9927–.9989，靠边界本身不等于错误。层级方向/拼写应遵守冻结文件，未授权修树。原诊断记录存在nervousment节点，但不能擅自把它替换为nervousness或新训练向量。
