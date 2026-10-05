# Experiment 1 与 Experiment 2 的初始化和阶段边界

2026-09-23用户明确确认；以下约束优先于此前可能含糊的“复用模型”表述。

| 阶段 | 数据 | 初始化 | 目的 |
|---|---|---|---|
| Experiment 1（已完成） | GoEmotions | raw google-bert/bert-base-cased | Standard BERT；固定threshold=0.30的test Micro-F1 58.72%、Macro-F1 48.62% |
| Experiment 2准备：frozen smoke | WoS固定小子集、完整taxonomy | raw google-bert/bert-base-uncased | 原任务工程验证，不是全量benchmark复现 |
| Experiment 2准备：patched smoke | 与frozen完全相同 | 与frozen相同raw uncased revision | 仅修parent-child map方向，独立输出并验收HGS几何 |
| Experiment 2主实验（尚未实施） | GoEmotions官方划分 | raw google-bert/bert-base-cased | 独立训练LGSA-GoEmotions，与Experiment 1对比 |

## 原始模型资产与训练checkpoint不可混同

- 可复用的Experiment 1资产是Hugging Face原始预训练snapshot，repo `google-bert/bert-base-cased`，revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`。后续使用前核对baseline manifest的原始模型哈希。
- 严禁用Experiment 1微调checkpoint初始化Experiment 2；也不继承Experiment 1的optimizer/scheduler状态。
- WoS smoke产生的训练checkpoint只用于该smoke保存/重载验证，不作为LGSA-GoEmotions主实验初始化。
- 最终GoEmotions主实验计划使用raw cased。若源码兼容性确实阻断，必须在训练前报告偏离，不能静默换模型。

## 当前授权与门槛

仅推进WoS真实资产准备、Slurm GPU检查、frozen工程smoke、parent-map最小补丁与patched smoke；不实现GoEmotions adapter，不启动WoS全量复现或GoEmotions正式训练。
`original_lgsa`严格冻结于`4ea3dfb39dcee82654f443811471dd29be32e87e`。最小补丁只进入版本控制的工作副本，在docs/code_changes.md写精确diff。
两次smoke使用同数据ID、seed、模型revision和执行设施，结果独立。工程通过不证明原版HGS几何正确。
为节省排队，可在一个Slurm allocation内依次执行GPU probe和独立smoke阶段，各阶段失败即停止，各自记录日志、产物与状态；依然逐阶段验收。

## backbone证据的准确范围

uncased是本次用户明确指定的WoS smoke配置。HPT参考入口明确默认`bert-base-uncased`，冻结LGSA入口本身只有`/your model path/`占位符。因此不将参考入口默认值表述为已独立核实的LGSA论文精确配置。
2026-09-23用户已确认uncased revision `86b5e0934494bd15c9632b12f734a8a67f723594` 五个文件hash全部通过，config/tokenizer本地加载通过，`WOS_SMOKE_MODEL_ASSETS_READY`。GPU模型加载未运行。

来源：[HPT入口](https://github.com/wzh9969/HPT/blob/main/train.py#L29)、冻结LGSA本地train.py、用户本轮指示及HPC输出。
