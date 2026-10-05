# 冻结 upstream 的差异与 smoke 前置项

适用 commit：`4ea3dfb39dcee82654f443811471dd29be32e87e`。原仓库保持干净；下表是分析和后续修复方案，尚未修改其模型。

| ID | 证据 | 影响 | 后续处理与验收 |
|---|---|---|---|
| U01 | train.py:111 `(parent,child)`；prompt.py:217 按 `(child,parent)` 读取；toy AST 已复现 | HGS parent map 错误，两层树细类未做预期局部正交化 | 在受版本控制的工作副本纠正 map；记录为 bug fix；测实际 hierarchy 的所有边及每家庭 residual cosine。 |
| U02 | prompt.py:33 `from_pretrained(..., True, 0)`；225 注册 ideal Parameter；244 输出用 raw_weight | prototype table 冻结，ideal Parameter 未连通，输入/输出 label weight 并不相同 | 保留 upstream 行为作为审计对象。先用真实 forward 检查各参数 grad；是否改为论文描述的可训练共享权重属于方法选择，不能静默修改。 |
| U03 | prompt.py:128–135 | GS 返回 rank 行，可能少于标签；全零会 stack 报错，零 parent 无保护 | 初始化前检查 norm、rank、最大家庭大小；退化时明确失败，不默默丢标签。修复算法必须记录。 |
| U04 | train.py:266 再次计算 GAT；graph.py:197 dropout | HGA 与 forward embedding 来自不同 dropout realization | 当前行为记录；真实 smoke 导出两份诊断，不能宣称一次共享的确定性 prototype。 |
| U05 | loss.py:127 与131 | brother_res 日志仅最后一个家庭，total 用家庭平均 | 将诊断修为实际平均并测试；属于日志修复，保持 objective 不变。 |
| U06 | train.py:291 引用已注释 align_loss | 开 swanlab 时会 NameError | smoke 不启用 swanlab；后续修日志，不能“恢复”一个新对齐损失。模块顶层仍 import swanlab，需要安装。 |
| U07 | preprocess_wos.py:164 输出 wos.taxnomy；prompt.py:117 读取 wos.taxonomy | 官方数据预处理后模型仍可能找不到文件 | 在工作数据副本生成正确文件名，并记录 alias；不改 frozen source。 |
| U08 | train.py:39 默认 `/your model path/`；89 只 mkdir 子目录 | README 示例不足以直接运行 | 显式传本地 BERT path；先建 checkpoints；模型资产记录 revision/hash。 |
| U09 | train.py:225–234 | scheduler 写死官方数据集样本数量 | GoEmotions 入口改为 len(train_loader)，不使用 nyt fallback。 |
| U10 | train.py:141–181 | 已存在 prompt cache 会跳过处理，新 hierarchy 可能读旧 cache | 工作目录以 source+dataset+hierarchy+tokenizer+max_length 哈希区分 cache。 |
| U11 | train.py:338–362 | smoke 若 macro/micro 均为0，best 文件不会创建，最后自动 test 会缺 checkpoint | smoke 使用独立受控 harness 强制保存/重载 last；不靠 F1>0 作为工程成功条件。 |
| U12 | graph.py:216 `num_labels-3` | 某些可选 graph 分支具有硬编码；GAT 路径也须核对节点数量 | 第一版仅保留 upstream 默认 GAT；验证实际 `[C+D,H]` 和 edge_index 范围。 |
| U13 | 全 repo 无 requirements lock；README 写 torch==2.12 | 不能直接套 baseline env 或猜是2.1.2 | 使用独立候选环境，记录版本变更，pip check + GPU GAT backward 后生成真实 lock。 |

## 官方 smoke 数据选择

优先 **WebOfScience 原任务的一个小子集**，保留完整 WoS taxonomy 和类别集合，再从原训练/验证划分固定选样。原因是两层结构容易审查且数据获取路径较公开；不是断言它比另两者全量样本数更小。

[LGSA README](https://github.com/weber-xuu/LGSA) 指向 [HPT](https://github.com/wzh9969/HPT) 预处理；HPT 指向 [HDLTex](https://github.com/kk7nc/HDLTex) 获取 WoS，并要求 `data/WebOfScience/Meta-data/Data.txt`（由原始 Excel 保存）后运行 `preprocess_wos.py`、`data_wos.py`。
当前冻结 LGSA 仓库没有 WoS 原始数据，也没有生成后的 value_dict.pt / slot.pt / split JSONL；NYT 的 id 列表不等于原文。先验证真实资产，不能拿随机 toy 数据冒充 official smoke。

## 原版与修复版的标注

- frozen-upstream：记录原入口能否启动及真实错误；失败结果保留，不覆盖。
- upstream-with-documented-fixes：每项补丁进入工作分支、保留 diff；成功也仅称“带已记录修复的官方任务工程验证”。
- GoEmotions adaptation：在官方任务 smoke 通过且 hierarchy/interface 方案审阅后再实现。

## 下一次官方 smoke 必须给出的证据

1. 输入数据来源/哈希/样本 ID、taxonomy/连续 ID/单父/无环检查。
2. 原 loader 输出 shape，BERT、GAT 的构造与 CUDA device。
3. initial、HGS、GAT 后三种 label 表，实际 parent map 和 residual diagnostics。
4. MLM、classification、四项 HGA 和 total 全部有限；至少两次 optimizer update（第一步 alpha=0 不能独自证明 HGA 参与训练）。
5. 非零有效梯度及参数更新；冻结项/无梯度项显式记录。
6. checkpoint 保存并重载；validation metrics 和机器可读报告。
7. Slurm 主作业与 .batch 的 COMPLETED、ExitCode 0:0，真实 GPU/进度日志。

当前：CPU toy probes PASS；官方 smoke NOT RUN。两个状态不能互换。
