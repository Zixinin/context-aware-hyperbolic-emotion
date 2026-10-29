# Exp6D — 01_ORIGINAL_VS_EXP6C_COMPARISON

完整只读审计；没有训练或新 Test 推理。

官方commit：`97ec6af6b19b831f69cb35a82dc65ea1395fa157`。论文：[ACL2023 HypEmo](https://aclanthology.org/2023.acl-long.613/)，DOI `10.18653/v1/2023.acl-long.613`；全文本地阅读并留存哈希，非凭摘要推断。

|维度|Original HypEmo|A0|A1|A1-S|分类|来源|
|---|---|---|---|---|---|---|
|Task|单标签 softmax CE|27维独立多标签 BCE|与 A0 相同|与 A0 相同|Required by Multi-label task|`source_review/upstream/hypbert.py:115`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Labels|27非neutral；每样本恰一标签|同名同顺序27维；每样本>=1情绪|与 A0 相同|与 A0 相同|Required by Multi-label task|`source_review/upstream/config.py:12`; `source_review/exp6c1_roberta27_hypaux/a1lib.py:172`|
|Training data|23485单标签非neutral|30587；保留多情绪和混合neutral样本|与 A0 相同|与 A0 相同|Required by Multi-label task|论文§4.1；DATASET_IDENTITY/冻结dataset_manifest.json|
|Validation/Test data|2956/2984|3834/3821|与 A0 相同|与 A0 相同|Required by Multi-label task|论文§4.1；冻结dataset_manifest.json|
|Backbone|roberta-base；Exp6A固定snapshot/旧运行时|相同模型系列与固定raw snapshot|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/exp6c0_roberta27_baseline/common.py:18`|
|Tokenizer|AutoTokenizer；Exp6A离线固定RoBERTa|固定A0 snapshot fast tokenizer|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/util_functions.py:27`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Max sequence length|200|50|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/util_functions.py:27`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|CLS / Pooling|first token→linear768→dropout→ReLU→dropout|raw first token|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/hypbert.py:65`; `source_review/upstream/hypbert.py:102`; `source_review/exp6c0_roberta27_baseline/model27.py:45`|
|Classification head|pooler_fc+ReLU+共享dropout掩码+Linear768→27|27列独立dropout掩码；weight[27,768]+bias[27]|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/hypbert.py:65`; `source_review/exp6c0_roberta27_baseline/model27.py:45`|
|Hyperbolic projection|HNN接受第二次dropout后的pooled表示|无|同上游HNN；输入改为raw first token|同A1，唯一输入改为.03h|Potentially important unexplained deviation|`source_review/upstream/hypbert.py:102`; `source_review/exp6c1_roberta27_hypaux/a1model.py:54`; `source_review/exp6c1s_scaled_train/scaled_model.py:15`|
|Hyperbolic dimension|100|无|100|100|Inherited from HypEmo|`source_review/exp6c1_roberta27_hypaux/a1lib.py:172`|
|Curvature|固定c=1|无|固定c=1|固定c=1|Inherited from HypEmo|`source_review/exp6c1_roberta27_hypaux/a1lib.py:172`|
|Label embedding source|发布label_tree/go_emotion.bin，27×100|无|同一bin按名称对齐|同A1|Inherited from HypEmo|`source_review/exp6c1_roberta27_hypaux/a1lib.py:172`|
|Label trainability|固定向量，重新构造为tensor不在optimizer中|无|register_buffer固定|同A1|Inherited from HypEmo|`source_review/upstream/models/base_models.py:132`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Distance-weighted loss|0.9×detach(d_true)×CE_i，后batch均值|无|无|无|Introduced for FYP protocol consistency|`source_review/upstream/hypbert.py:115`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Auxiliary loss|batch-mean NLL − batch-SUM ball dot|无|batch-mean positive-mean CE-dot|同A1|Potentially important unexplained deviation|`source_review/upstream/models/base_models.py:132`; `source_review/upstream/models/base_models.py:140`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Positive-label aggregation|单一true-label|独立binary targets|q=y/&#124;P&#124;同时作用于aux CE与dot|同A1|Required by Multi-label task|`source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Loss reduction|CE_i先乘distance后mean；aux两项归约不同|B×27元素mean|BCE元素mean；aux正标签mean后batch mean|同A1|Required by Multi-label task|`source_review/upstream/hypbert.py:115`; `source_review/upstream/models/base_models.py:132`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Loss coefficients|alpha=.9/gamma=.1；并非实际损失比例9:1|BCE系数1|BCE系数1；aux系数.1|同A1|Introduced for FYP protocol consistency|`source_review/upstream/config.py:12`; `source_review/exp6c1_roberta27_hypaux/a1model.py:15`|
|Optimizers|Adam(main)+custom RiemannianAdam(HNN/decoder)|单一AdamW，bias/LayerNorm免WD|同A0规则，包含新参数|同A1|Introduced for FYP protocol consistency|`source_review/upstream/hypemo.py:54`; `source_review/upstream/optimizers/radam.py:99`; `source_review/exp6c0_roberta27_baseline/model27.py:80`|
|Schedulers|仅双曲branch StepLR10×.5，每BATCH调用|全参数linear warmup/decay；765/7648|同A0|同A0|Introduced for FYP protocol consistency|`source_review/upstream/hypemo.py:89`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Batch size|发布config/Exp6A=100；构造默认16；论文效率表16|16；eval64|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/config.py:12`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Epochs|20；约4700更新|4；7648更新|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/train.py:45`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Learning rates|main1e-5；branch.01随后每10batch减半|所有参数5e-5|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/hypemo.py:54`; `source_review/upstream/hypemo.py:89`; `source_review/exp6c0_roberta27_baseline/model27.py:80`|
|Gradient clipping|无显式clip|全参数global norm1|与 A0 相同|与 A0 相同|Introduced for FYP protocol consistency|`source_review/upstream/hypemo.py:89`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Dropout|backbone.1；pooler前后两次.1；HypLinear weight.1；decoder线性输出.1|backbone.1；分类27列独立.1|同A0+上游branch dropout|同A1|Introduced for FYP protocol consistency|`source_review/upstream/layers/hyp_layers.py:72`; `source_review/upstream/hypbert.py:65`; `source_review/exp6c0_roberta27_baseline/model27.py:45`|
|Evaluation mode|no_grad但没有eval()，dropout活跃|eval()+no_grad|同A0|同A0|Implementation correction|`source_review/upstream/hypemo.py:107`|
|Checkpoint selection|每轮valid+test；同时记录valid loss/F1及best test；原entry不存checkpoint|固定.30下validation Macro-F1；earliest tie|同A0|同A0；本pilot无新Test|Implementation correction|`source_review/upstream/train.py:45`; `source_review/exp6c1s_scaled_train/srun.py:224`|
|Evaluation metrics|论文true-support Weighted-F1；公开代码f1(pred,true)用pred-support|true,pred标准Micro/Macro/Weighted/Rare P/R/F1|同A0|同A0并增加Macro AP|Implementation correction|`source_review/upstream/hypemo.py:107`|
|Prediction threshold|argmax单标签，无binary阈值|sigmoid>=.30|与 A0 相同|与 A0 相同|Required by Multi-label task|`source_review/upstream/hypemo.py:107`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Seed / repetitions|config1234；Exp6A预定1234/42/123/2026/3407|仅42|仅42|仅42|Introduced for FYP protocol consistency|`source_review/upstream/config.py:12`; `source_review/exp6c0_roberta27_baseline/common.py:18`|
|Initialization / RNG|原版先构造branch后main；单标签head|历史FYP seed42初始化|额外branch初始化fork_rng，shared初值及初始logits与A0严格匹配|同A1；缩放无参数/RNG消耗|Introduced for FYP protocol consistency|`source_review/exp6c1_roberta27_hypaux/a1model.py:68`; `source_review/exp6c1s_scaled_train/scaled_model.py:15`|

分类可有次要解释：CSV note逐项说明。重要的“未解释偏离”是相对原版的语义差异，不等于当前代码有bug。
没有发现需要自动修复的A0/A1/A1-S缺陷；原版已知评估问题另见协议报告。若Ubuntu源/配置/数值不符，检查直接失败，推荐保持待定。

Head实际顺序是**Linear→Dropout→ReLU→Dropout→Classifier**，不能照示意图误写成先ReLU后唯一dropout。原版HNN也接收第二次dropout后的表示。FYP的head及HNN输入均不同，因此不能把原论文与Exp6C的差值归因于单一几何因素。A0/A1/A1-S内部head保持匹配。
