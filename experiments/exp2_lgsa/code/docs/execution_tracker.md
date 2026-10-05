# FYP 第二个实验：不跳步的执行与验收表

截至2026-10-05（依据用户返回的HPC证据更新）。本地准备和真实HPC运行分别记账。PASS只能指明对应证据范围；未运行记 NOT RUN，不伪装成 FAIL。

## 主线门槛

| 阶段 | 实际任务 | 当前状态 | 进入下一步的条件 |
|---|---|---|---|
| 0 | 新建独立工程、核实来源、冻结源码、保存baseline依据 | 本地完成 | source hash、独立分支、baseline小文件hash已核验 |
| 1 | 论文+源码理解；subspace/HGS/HGA/GAT/四loss | 文档与CPU小张量审计完成 | 阅读核心两份文档，知道upstream差异；不误称pipeline smoke |
| 2a | HPC现状、独立环境、资产准备 | HPC CPU环境、uncased模型哈希与WoS预处理/smoke资产均PASS；frozen源码干净 | 不再重建环境；GPU验证在Slurm内 |
| 2b | frozen upstream官方WoS数据pipeline smoke | PASS；HPC Job2984968及.batch COMPLETED/0:0；64batch、GPU、参数更新和保存重载通过；真实HGS反向map问题确认 | 工程PASS不等于几何正确；实际map_matches=false、fine_vectors_changed=false |
| 2c | parent-map最小修复版同配置smoke | PASS：Job2985243主任务及.batch COMPLETED/0:0；64步、HGS几何和重载通过（用户终端证据） | 仅工作分支修一处map方向；复用2984968同一run函数、资产与环境；通过预GAT几何验收与工程smoke |
| 3 | hierarchy输入接口和适配可行性 | 静态分析完成 | frozen及parent-map patched两次独立smoke通过后审阅兼容方案与候选mapping |
| 4 | GoEmotions hierarchy v0 | 2026-09-23用户已采用sentiment-derived hierarchy v0；54,263条官方记录的派生标签合同本地通过 | 确认来源与中性分支；loader执行、覆盖/无环/唯一父等校验 |
| 5 | 模块化GoEmotions adapter | 已实现；本地11项测试及HPC tiny/真实GoEmotions GPU流程通过（用户终端证据） | 上游问题处理有diff，保留原ID和splits，多标签/祖先/深度mask逐例测试 |
| 6 | 1000train/500val/1epoch GoEmotions smoke | PASS：Job2985311，1000/500样本、63步、HGS与checkpoint重载通过 | 原任务smoke+adapter前置通过；CUDA训练、checkpoint重载、metrics与diagnostics齐全 |
| 7 | HPC seed42 full run | PASS：Job2985311，4epochs/10856步，best_epoch=4，5427条test；完整产物哈希复核待返回 | smoke完成证据逐项审查；用户终端按步骤提交；run目录带job ID |
| 8 | 与标准BERT比较 | 整体指标已比对：fixed .30 Micro59.46/Macro53.29/Rare52.05；逐类分析待返回 | 主/次threshold、28类指标、rare/难类/差值及几何诊断完成 |
| 9 | 下一步一个建议及消融计划 | 仅写计划 | full结果解释后，用户明确同意才启动消融；Hyperbolic不在此实验 |

顺序更新：用户于2026-09-23明确授权在patched smoke排队期间实现adapter和自动衔接；训练仍由patched几何gate与GoEmotions smoke gate共同控制，不跳过GPU验收。

## 用户材料0–29逐项对应，防止遗漏

| 编号 | 要求 | 交付/位置与状态 |
|---|---|---|
| 0 | baseline隔离 | 原工程仅读取；references/baseline固定参照 |
| 1 | RQ1–4/Euclidean范围 | README、code_map、subspace、compatibility；RQ4等待实测 |
| 2 | 官方source/commit/branch | original_lgsa_version.md、source_version.json、source hash |
| 3 | 15项源码映射 | lgsa_code_map.md 已完成 |
| 4 | 8个subspace问题 | lgsa_subspace_explanation.md 已完成 |
| 5 | 独立env/依赖/lock | HPC安装、CPU导入/GAT backward/入口检查通过，已生成安装版本记录；GPU待执行 |
| 6 | HPC/Slurm限定 | 用户选择终端完成环境安装和轻量CPU检查；GPU验证与训练经Slurm |
| 7 | 邮件通知 | BEGIN/END/FAIL，submit_checked用用户提供LGSA_EMAIL |
| 8 | 标准Slurm配置 | lgsa_environment.sbatch；后续训练文件逐阶段生成 |
| 9 | 原任务工程smoke | upstream_findings列完整验收；WoS资产与官方smoke待做 |
| 10 | 官方GoEmotions splits | 当前本地确认43410/5426/5427；28 labels |
| 11 | HTC兼容性 | lgsa_goemotions_compatibility.md 已完成 |
| 12 | hierarchy来源及完整mapping | JSON已获用户采用；命名sentiment-derived v0；Neutral工程补充分开标注 |
| 13 | 保留multi-label/28类评测 | compatibility合同明确；实现与执行待做 |
| 14 | 模块化adapter及改动记录 | 独立adapter已实现并版本控制；GPU执行待两级smoke门槛 |
| 15 | GoEmotions smoke1000/500/1 | 已实现；HPC待执行，输出smoke/JOBID |
| 16 | full seed42 | 已准备依赖作业；同allocation内smoke通过后从raw重新开始，输出seed42/JOBID |
| 17 | 全部指标/比较表/per_label_delta | baseline指标已复制核验；比较CSV须LGSA结果就绪后生成，不填假数字 |
| 18 | fixed0.30/val全局threshold | compatibility锁定验证集选择规则，不用test调参 |
| 19 | 六个难类及TP/FP/FN | baseline逐类F1可读；计数与LGSA比较待评测实现 |
| 20 | 几何diagnostics与PCA/heatmap | runner已实现initial/HGS/learned张量和诊断图；只从真实运行张量生成 |
| 21 | 暂无Hyperbolic | 未实现 |
| 22 | 消融只计划 | lgsa_ablation_plan.md；不启动 |
| 23 | 唯一日志/checkpoint | Slurm%j；runner已实现，目录隔离按JOBID |
| 24 | 全部实验metadata | 下表字段合同；真实运行后写，不提前编造GPU/job ID |
| 25 | 工程布局 | 独立根已创建；阶段未到的模块和输出不会放空实现冒充完成 |
| 26 | HPC精确操作顺序 | README；每次只给当前一步并检查返回 |
| 27 | 邮箱提醒/拒绝placeholder | README示例+submit_checked校验；尚未提交任何任务 |
| 28 | LEVEL1–8 | L1完成，L2解释/分析完成；L3–8运行标准均未达成 |
| 29 | 最终A–M报告 | full与comparison后生成；当前为阶段报告，禁止声称实验全完成 |

## 真正每次实验的metadata合同

`experiment_name, git_commit, original_LGSA_commit, adaptation_commit, Slurm job ID, hostname, GPU, Python version, PyTorch version, CUDA runtime, torch-geometric version, Transformers version, dataset version/hash, hierarchy version/hash, seed, batch size, learning rate, epochs, loss weights, threshold protocol, start time, end time`。
补充 checkpoint selection、数据样本ID、tokenizer/model revision、run/output/cache path、补丁清单和软件lock hash。
结果写为 `experiment_metadata.json`；开始时记录start与running状态，结束时补end/exit/status，失败也保存。

## 终验固定检查

`squeue`只证明排队/运行；`sacct`检查主作业及.batch（数组要逐子任务），`COMPLETED / 0:0`、GPU/进度日志、无未解决异常、所需产物和校验和全部一致才通过。
邮件是否送达需实际收件确认，依赖学校Slurm mail配置。
