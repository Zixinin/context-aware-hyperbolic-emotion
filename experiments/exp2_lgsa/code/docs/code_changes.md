# 本阶段改动记录

| 范围 | 原行为 | 本阶段行为 | 理由 |
|---|---|---|---|
| 原 Standard BERT 工程 | 已有源代码、归档、结果 | 仅读取；没有写入、覆盖 checkpoint 或改环境 | 保留实验1 |
| original_lgsa | GitHub main | clone 后 detached 于4ea3dfb，源码未修改 | 冻结作者代码 |
| 本工程外层 | 不存在 | 新建 goemotions-adaptation 分支、文档、审计脚本、候选环境配置 | 为实验2建立独立记录 |
| CPU geometry audit | 无 | 抽取 upstream 原始 AST 方法，用随机小tensor验证 | 为源码差异提供可复查证据；非BERT训练 |
| baseline reference capture | 指标在大归档内 | 仅复制8个小文件并对照归档SUCCESS哈希核验 | 避免依赖手抄数字 |
| GoEmotions adapter / LGSA 模型补丁 | 未实现 | adapter仍未实现；2026-09-23原版smoke通过后本地working分支只修parent-map一行 | 先原版验证，再最小补丁验证，再审适配方案 |

后续每个模型改动必须增加：文件、函数、原行为、新行为、原因、相关test与patch commit。
U01 等问题清单不等于已修复。

## 2026-09-23 U01已在本地独立工作分支应用（HPC待执行）

位置：working_lgsa/models/prompt.py，Prompt.init_embedding。分支lgsa-goemotions-parentmap，本地commit ba429c8984f22e32e9ec11641394fe5f1beebdc3。HPC commit由提交脚本创建并记录，不能用本地hash冒充HPC已应用。

```diff
-        value2slot = {child: parent for child, parent in self.path_list
+        value2slot = {child: parent for parent, child in self.path_list
```

代码文件只有上述一行变化；另一新增文件为该工作仓库的docs/code_changes.md。其余tracked源码逐文件hash与frozen一致。依据为Job2984968工程通过但真实map_matches=false/fine_vectors_changed=false。
scripts/prepare_parentmap_patch.py创建独立git clone（不使用会写original_lgsa/.git的worktree），以明确自动化身份提交，不改全局git身份；拒绝未知代码差异。
scripts/run_patched_smoke.py复用原frozen run()，核对原harness SHA、资产及pip freeze一致，并增加独立预GAT几何验收。模型源码无其他补丁。原harness仍返回工程结果中的geometry_correctness_claimed=false；patched通过另以hgs_geometry_passed=true明确限定预GAT HGS，不声称GAT后仍正交。
本地测试见tests/test_parentmap_patch.py；GPU paired smoke待HPC结果。

## 2026-09-23 frozen工程smoke执行设施（不改模型源码）

- scripts/run_frozen_smoke.py直接导入冻结Prompt、HierarchicalGofLoss、evaluate、seed_torch、scheduler函数，原始data_map_function以AST原节点编译，不重写其映射规则。
- 相比完整train.py：独立受限循环，仅运行256train/128dev一个epoch，batch=4；不读取test，始终保存并重载last，不依赖非零F1产生best文件；HGA alpha以明确batch索引计算，消除tqdm刷新频率造成的p_bar.n滞后。原WoS LR warmup7518steps保留。上述执行差异写入每次metadata；不称原train.py CLI逐行完整复现或完整benchmark。
- 始终在输出目录复制smoke数据，缓存/日志/checkpoint按JobID隔离；original_lgsa仅导入读取，禁写pyc；运行前后检查commit、git status与tracked文件SHA256。
- 记录实际反向parent map、真实初始化/HGS张量和几何诊断，不修改parent map，不以工程PASS宣称几何正确；记录first forward GAT与第二次HGA GAT不同dropout输出。
- 使用原四项HGA目标；额外报告brother_res真实family均值，保留上游最后family日志值；不改任何损失权重、冻结策略或参数连通性。
- 记录实际梯度和backbone参数变化；保存训练后GAT张量、validation预测/指标、checkpoint SHA、reload logits对齐、依赖版本和GPU证据。
- 不创建adapter或应用模型补丁。patched阶段须采用同样执行设施并记录唯一模型diff。

## 2026-09-23 WoS数据准备设施

新增scripts/prepare_wos_assets.py；不修改frozen的两个预处理脚本，通过cwd将输出定位到assets/wos-v2/full/WebOfScience。使用标准库读取已固定SHA256的官方XLSX，无需新增HPC依赖；导出的表格与openpyxl独立读取逐行一致。沿用原脚本清洗和划分，保留全部141标签。
原脚本生成wos.taxnomy，而模型要求wos.taxonomy：只在工作数据目录生成字节一致的文件别名，并记录manifest。此项是数据路径兼容处理，不是模型补丁。
smoke仅固定256训练/128验证样本和源Excel行号；不使用test、不创建GoEmotions adapter、不开展WoS全量训练。真实GPU smoke仍待执行。

## 2026-09-23 隔离安装工具与运行时导入检查

- 用户日志确认全部依赖安装完成且 `pip check` 通过，但 `torch_geometric → torch._dynamo → triton → setuptools` 导入链在 distutils 断言处失败。
- 安装脚本顶部导入 `pip._vendor.packaging`；Python 3.10 下先导入 pip、再导入 setuptools 的同类故障有上游记录：https://github.com/pypa/setuptools/issues/3044 。这是当前诊断，干净进程检查结果仍待 HPC 验证。
- `environment/repair_validation_process.py` 将原有检查代码移至 `environment/check_cpu.py`，安装脚本通过独立 Python 进程调用。检查进程不导入 pip；`pip check/freeze` 仅用 subprocess 执行。
- 保留原来的依赖导入、CPU GAT 前后向、LGSA 类导入、train.py --help、frozen commit/clean 状态和版本报告检查；单独执行 checker 不下载或安装包。
- 修复不改变依赖版本或模型源码，不修改 `original_lgsa`。本地验证仅涉及修复脚本、语法与既有下载解析测试，不代表 HPC 运行时检查成功。

## 2026-09-22 安装脚本的本地 wheel 路径解析修复

- 文件：`environment/install_all_cpu.py`，`main()` 的依赖清单读取循环。
- 用户在 tokenizers 下载并校验完成后遇到 `InvalidRequirement`：uv 输出保留了 Torch wheel 的绝对路径，不能直接传入 `packaging.Requirement`。
- 修复：解析前检查绝对路径必须等于已验证 SHA256 的 Torch wheel，将其转换成 `torch @ file://...` 格式后沿用原校验逻辑。其他本地路径仍拒绝。
- 保留已有版本清单、下载文件和断点状态，不重新解析依赖，不改 `original_lgsa` 或模型代码。
- 回归检查覆盖裸路径、命名 file URL、前后普通依赖及意外本地路径；HPC 续跑结果待用户返回。

## 2026-09-22 用户确认的最小补丁边界（尚未应用）

只在版本控制的工作副本 `models/prompt.py` / `Prompt.init_embedding` 修复parent-child map方向，原 `original_lgsa/` 不改。
预定精确补丁如下；当前记录为计划，不能算已执行：

```diff
-        value2slot = {child: parent for child, parent in self.path_list
+        value2slot = {child: parent for parent, child in self.path_list
                       if parent != -1 and child < self.num_labels and parent < self.num_labels}
```

原因：train.py提供(parent,child)，HGS消费child→parent。其余filter、权重冻结、GAT、分类/MLM/HGA损失和threshold代码均不随这项修复改变。
先完成未修改版本工程smoke，再应用/提交此一行补丁并运行独立smoke；数据、配置、seed相同，输出/JobID分离。
若另有必须修复的启动问题，另列补丁与证据，不能混入“只修parent map”标签。


## 2026-09-23 GoEmotions adapter and dependency preparation

User adopted sentiment-derived v0 and authorized parallel preparation while
Job2985243 is pending. goemotions_adaptation is a separate Git repository/branch;
working_lgsa stays at its tested minimal patch commit. See
goemotions_adaptation/docs/code_changes.md and README.md for the exact protocol.
Local tests: 11 CPU synthetic/unit checks plus all54,263 official rows' label/ID
contracts. Real raw-cased GPU smoke/full have NOT RUN.

Downstream uses afterok:2985243 and hard upstream/GoEmotions artifact gates;
smoke and full use separate processes in one allocation. Full reinitializes
raw cased weights and checks initial state fingerprints. Checkpoints every200
full steps support same-run recovery. No ablations or hyperbolic code.
