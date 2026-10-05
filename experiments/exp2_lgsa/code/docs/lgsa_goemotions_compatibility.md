# GoEmotions 接口方案（2026-09-23已实现，HPC验收待执行）

沿用 FYP 实验1的 28 labels、官方 splits、原文本、seed42。模型使用已隔离的parent-map工作分支；adapter位于独立Git仓库goemotions_adaptation，GPU执行仍受2985243的工程/几何gate约束。

| LGSA 假设 | GoEmotions 对应 | 必要改动 | 科研风险 |
|---|---|---|---|
| 显式单父层级，多层都可监督 | 原始28类没有每样本独立标注的粗类 | v0 从官方 sentiment grouping 派生4粗类；单独文件配置 | coarse 是 fine 的派生监督，不是真实额外人工标注 |
| `slot.pt` / `value_dict.pt` + JSONL | 官方 text TAB comma-separated labels TAB ID | 模块化转换，保留 ID/text/多标签/划分 | 不得运行 WoS 的随机 split 或 lowercasing |
| 多标签包含祖先 | GoEmotions 只有 fine IDs | `Y_all = Y_fine ∪ parent(Y_fine)`，每个 coarse 去重 | 同一个样本可同时有多个 coarse，不能 argmax |
| 类别数 C 包括粗/细类 | 28原始 +4人工粗类 | 原始0..27不动，粗类28..31；ROOT逻辑节点不计入32类 | 评价若直接用 C=32 会虚增分数 |
| 层级 prompt targets `[B,D*C]` | 两层 D=2 | 粗层仅4个位置0/1、其余-100；细层仅28个位置0/1 | masking/reshape 需逐样本比对避免错位 |
| GAT 含 depth prompt 节点 | 32标签+2depth prompt | GAT 节点34，另一个 MASK embedding 不入图；验证所有 index | 不要把 ROOT、depth prompt 当 ground truth |
| 一个 child 一个 parent | sentiment 每细类一父 | neutral 设独立 synthetic `Neutral` 父节点 | 单 child 分支不会发生 sibling GS/Ls；统计要单列 |
| 标签名称输入 embedding 均值 | 官方细标签名 + coarse 名 | 使用同 tokenizer、记录 special tokens 行为 | Neutral/neutral 词形相近可能造成近共线、零残差 |
| MLM + multilabel categorical CE | baseline 是 BERT classifier + BCE | 保留 LGSA 原 loss 进行完整模型实验 | 改善不能归因于 geometry 单一因素：prompt、GAT、MLM、loss、长度都变了 |
| `logit > 0` 的多标签预测 | baseline sigmoid≥0.30 | 取 fine depth 的28个 logits，sigmoid 后固定0.30/val选阈值；同时存原生logit>0诊断 | sigmoid 是分数变换，未保证概率校准；不是 BCE 训练的可比概率 |
| official trainer 的512 tokens / Adam / scheduler | baseline max_length50、AdamW、4epochs | 所有差异记录配置；首轮固定正文50 tokens加4提示位置、4epochs；具体配置见adapter README | 不强行同参，也不能隐藏预算差异 |
| macro/micro 各选一个 checkpoint | baseline val macro@0.30 选主checkpoint | 主比较预先固定 val macro@0.30；micro-best仅次要诊断 | 不可看 test 后挑 checkpoint |
| 依赖目录名与旧cache | 新 GoEmotions path | 独立 adapter/runner 和 cache fingerprint | 修改 taxonomy 后复用缓存会造成静默错误 |

## 原接口规划（现有实现对应见goemotions_adaptation/README.md）

- `data_goemotions.py`：读原 TSV，校验 `43410/5426/5427`，IDs 与 baseline 对齐；multi-hot 和 ancestor expansion；返回全层标签与独立原始28维标签。
- `hierarchy_goemotions.py`：读 JSON，校验完整覆盖、唯一父、无环；生成 `value_dict`、`slot`、`value2slot`、`depth2label`、`path_list`。
- `config_goemotions.py`：输入/输出、BERT revision、token length、seed、loss weights、threshold、hierarchy hash；禁止隐式 defaults 落入 nyt。
- 独立 runner：调用受版本控制的 LGSA 工作副本；不重写成普通 BERT+BCE；记录 upstream patches。

## 示例（说明数据合同，不运行）

`fine=[joy(17), nervousness(19)]` → 保留17、19，增加 Positive(28)、Negative(29)。
粗层目标可同时两项为1。评价仍只取原始 `[17,19]`，不能把28、29加入 Micro/Macro。

## 固定公平评价合同

- 主结果：28类，fixed0.30，官方 test5427；主 checkpoint 用 validation macro@0.30 选。
- 次结果：validation 全局 threshold candidates 0.05..0.90，步长0.05；val Macro-F1 最大，平局取较小阈值；test 不参与选择。
- Rare：完整 train 频次升序、ID平局升序，取7类。已从 baseline 归档核验：grief77、pride111、relief153、nervousness164、embarrassment303、remorse545、fear596。
- 输出 Micro/Macro P/R/F1，28类 P/R/F1/support；rare/nonrare arithmetic mean F1。
- 难类 grief、relief、realization、nervousness、disappointment、remorse：加 prediction_count/TP/FP/FN。
- 比较表单位统一百分数；delta 明确为百分点。baseline 使用本工程 `references/baseline` 的已核验小文件。
- single-seed 只能作为首轮观察；不把 PCA、F1 变化直接解释为 hierarchy 的因果效应。

## 顺序门槛

用户于2026-09-23明确采用sentiment-derived v0，并授权在patched WoS排队期间实现adapter及下游Slurm任务。
已准备自动链条：patched WoS成功与几何验收 → GoEmotions smoke → 硬验收 → 从raw cased重新初始化full。CPU toy检查不替代真实GPU smoke。
