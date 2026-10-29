# 本地已核实的原作者层级结构（向量距离待Ubuntu）

7个缓存原作者文件SHA与A1冻结的UPSTREAM_REFERENCE完全一致，记录在source_review；当前网络请求超时，不宣称本轮fresh fetch成功。

原label_tree.tsv的列为child、ancestor。train_label_embedding.py以Relations(reverse=True)读入，因此审计先反转为parent→child。图共34节点、75条闭包边；传递约简后33条cover边，42条冗余祖先边；没有缺失闭包边。含root最深4层：root→semi-positive/semi-negative→positive/negative→部分正式情绪；ambiguous分支较浅。

直接父节点（全部27类在JSON中列出）：

- negative：anger、annoyance、disgust、fear、grief、remorse、sadness。
- positive：excitement、joy、love、optimism、pride。
- semi-positive：admiration、amusement、approval、caring、desire、gratitude、relief。
- semi-negative：disappointment、disapproval、embarrassment、nervousness；另有非正式nervousment节点。
- ambiguous：confusion、curiosity、realization、surprise。

原trainer移除root关系后，PoincareModel(train_data=transitive_relations)实际使用传递闭包，而非只用直接父子边。tree_relations用于可视化。

树里nervousment和nervousness同时存在：额外节点会进入声明的标签预训练关系；导出只遍历config正式label2idx，因此nervousment不作为正式第28类输出。无法仅凭发布源码/向量量化它对预训练数值的影响，不能擅自修树重训。

A1固定27标签向量，不会在A1中被常见标签挤走。当前未获得本地固定向量数组，27×27实际距离、范数与七Rare最近邻需读取Ubuntu现有.npy；不能用树关系代替实际向量距离。
