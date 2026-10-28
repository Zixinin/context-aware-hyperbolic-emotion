# 1 标签向量与来源审计

来源：`/home/zixinfeng/Documents/HypEmo-Reproduction/HypEmo/label_tree/go_emotion.bin`；SHA256：`2695b98e160e79c62533d6bee55fdbcfdbbcba87cbcde8ab6db2734dc16ab20a`；固定upstream commit：`97ec6af6b19b831f69cb35a82dc65ea1395fa157`。
实际矩阵27×100，标签顺序详见LABEL_VECTOR_AUDIT.csv，neutral不在其中；逐名字核对发布字典、Exp6B矩阵与A1保存矩阵，完全相同。
范数范围0.9927296810–0.9989144615；c=1表示曲率−1，数学有效半径1，超界数0。文本FP32投影保护阈值.996不是数学边界；固定标签超过.996不等于超出球，不重新投影它们。
固定buffer位置：/home/zixinfeng/Documents/context-aware-hyperbolic-emotion/experiments/exp6c1_roberta27_hypaux/a1model.py:41；不在optimizer参数组，不接收梯度。模型构造时核对encoder/decoder/每个超曲层使用相同c=1。
预训练入口读取label_tree.tsv的去root传递关系，get_dicts(...False)提供标签名；没有在所审入口发现读取train/validation/test文本或实例标签的路径。
**来源限制：发布bin原始生成命令、训练日志及taxonomy构造全过程未齐全，不能宣称完整证明历史无泄漏。当前没有发现实例级泄漏证据；如Ubuntu输入/源码核对发现实例划分依赖则停止。**
树同时含正确nervousness及额外nervousment；本轮不合并节点、不修树、不重新预训练，未知其历史数值影响。
所有真实路径、哈希和逐行位置在LABEL_AUDIT.json、SOURCE_LINE_INDEX.json；没有以Proposal替代来源证据。
