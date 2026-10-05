# HPC逐步执行记录

以下事实来自用户粘贴的HPC终端输出，不是本地执行推断。

## 已确认

- 登录节点 `xpszlogin1`，Conda22.9.0。
- HPC clone完整22个文件，checkout `4ea3dfb39dcee82654f443811471dd29be32e87e`；status为 `HEAD (no branch)`，无列出的修改。
- `cuda/11.8.0-gcc-8.5.0-d7ndetl`加载成功；nvcc11.8.89。
- sinfo显示gpu4090为up，GRES包含rtx4090；这不说明此刻有空闲GPU，也未验证本用户account/QOS提交权限。
- 该次squeue显示1个MVFA RUNNING、3个Dependency PENDING；不修改或取消这些任务。
- 用户提供的test明确显示 `LGSA_ENV_NOT_CREATED`。

## 最初的GPU环境创建方案（已被下文终端方案替代）

准备 `slurm/lgsa_create_env.sbatch`：在分配的GPU节点创建Python3.10+pip最小环境，记录GPU、版本、pip check及Conda包列表。
脚本本地仅做shell语法检查；尚未收到此任务的提交ID或运行结果。
邮箱由用户在HPC本地输入，通过sbatch `--mail-user`传入；启用BEGIN/END/FAIL。

最小环境通过后才分阶段安装Torch及LGSA依赖。`LGSA_MINIMAL_ENV_PASS`不是完整环境或LGSA smoke通过。
若遇QOSMaxSubmitJobPerUserLimit，记录提交失败、等待容量，不自行取消其他任务。

## 后续提交与调度偏好更新

用户已返回环境创建任务ID `2984874`，只确认提交成功，尚未收到运行/完成证据。
用户指出4090排队成本高，希望先在不占GPU的资源上完成环境准备，再将已准备好的GPU阶段集中执行。
最小Conda环境创建本身不需要GPU；先前脚本为同时检查GPU而申请4090，不能把这种脚本设计当成学校强制规定。
两份已读参考文档未明确授权或禁止登录节点包安装；不据此默认在登录节点进行长时间Conda求解/编译。
下一步先查看2984874状态和可用分区，优先考虑有权限且排队合适的CPU资源。未确认旧任务状态前，不在另处并发创建同一环境，不取消任务。
后续可将已准备好的GPU验证阶段放入同一allocation顺序执行，逐阶段独立日志/结果且失败即停；不得跨越尚未完成的适配方案审阅门槛，亦不因减少排队而省略frozen和patched的独立结果。

## 当前采用方案：用户明确要求在终端先准备环境

最新用户输出确认2984874仍PENDING(Priority)，Elapsed为0；此时ExitCode0:0不是运行成功。
可见CPU分区为cpu6348、cpu8358、cpudebug、cpuhuge1t，但未验证用户访问权限。
用户明确选择在当前终端完成环境准备，然后再提交实验Slurm任务，因此不再申请GPU来执行Conda/Python包安装。
先仅取消处于PENDING的2984874，并检查最终状态；若已开始或状态不明确，停止后续环境创建，避免并发写同一prefix。
取消完成后，先创建独立Python3.10+pip，随后分阶段安装预编译PyTorch/PyG及其他依赖、准备资产。实际GPU功能验证仍在Slurm分配内进行。
本地未代操作集群，取消/创建结果均待用户返回。

## 最新结果：最小环境成功

用户提供Conda transaction完成输出，Python3.10.21、pip26.2.1路径在lgsa_goemo，pip check无冲突。
最小环境已完成，不再重建。该条返回没有包含2984874的取消状态文本，不将取消单独标记为已核验。
下一步仅下载/校验Torch2.5.1+cu118的cp310 Linux wheel，采用持久.part续传；之后才进入依赖安装。

## PyTorch主wheel已完成，准备一次性安装

用户返回799MiB下载100%、SHA256 OK、TORCH_WHEEL_DOWNLOAD_VERIFIED。该文件在HPC已完成，不再下载。
用户要求一次性安装全部环境包；已准备terminal入口，将版本解析、依赖续传下载、离线安装、CPU检查与版本记录合并。真实安装与GPU验证状态仍待执行证据。

## 安装中断：本地 Torch wheel 路径解析

用户返回 tokenizers-0.19.1 wheel 下载校验成功，随后依赖清单中的 Torch 绝对路径导致 `packaging.Requirement` 抛出 `InvalidRequirement`。这是安装脚本的解析缺陷，不是 Torch 文件校验失败。已在本地修正路径规范化并准备原地修复/续跑命令；不删除缓存，不修改 frozen upstream。完整安装和 CPU_READY 尚未通过，等待 HPC 返回。

## 2026-09-23 包安装成功，运行时检查未通过

用户在 xpszlogin1 上观察到 pip 长时间处于 D 状态，两次读写计数仍有增长。随后日志确认 `Successfully installed` 和 `No broken requirements found.`，不再将包安装标记为进行中。
后续导入 torch_geometric 经 Triton 触发 setuptools/distutils 断言，未到 CPU GAT 测试、原版完整 smoke 或 GPU 验证。准备将检查移到不导入 pip 的独立进程，复用已安装环境；结果待用户返回。

## 2026-09-23 HPC CPU环境检查通过

用户返回 `VALIDATION_PROCESS_FIX_APPLIED`、`No broken requirements found.`、官方 `train.py --help` 和最终 `LGSA_ENV_CPU_READY`。
版本：Python 3.10.21；Torch 2.5.1+cu118 / runtime 11.8；PyG 2.6.1；Transformers 4.44.2；datasets 2.21.0；huggingface_hub 0.26.5。
`cpu_gat_backward_passed=true`；`cuda_available_here=false`；`gpu_smoke_passed=false`。源码commit为4ea3dfb39dcee82654f443811471dd29be32e87e，checker前后clean断言通过。
依赖指纹：bda8d58557a8641533a8ea1021a63e9796a4100cc8f3d2cb70ab0728590a95b8。
本阶段通过，不再安装或重建环境。下一步只读核验HPC现有模型/数据资产，准备官方WoS原任务smoke所需输入；尚未运行LGSA完整前后向/GPU训练，也未进入GoEmotions正式实验。

## HPC模型资产清单与smoke backbone选择

用户确认实验1 `google-bert/bert-base-cased` revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e` 缓存存在，config 570B、vocab 213450B、safetensors 435755784B；本次仅文件存在/大小检查，未完成内容哈希复核。无pytorch_model.bin，不因此重复下载同一权重。该缓存留待GoEmotions适配阶段核验复用，不改写实验1。
2026-09-23在线核对 https://raw.githubusercontent.com/wzh9969/HPT/main/train.py 第29行（1-based），默认`--arch=bert-base-uncased`；冻结LGSA只留模型路径占位符。为使WoS工程smoke沿用其参考入口的backbone，准备独立uncased snapshot，不将cased静默代入原任务配置；这不是已证明的LGSA论文精确backbone配置。
准备脚本environment/prepare_smoke_model.py：先固定revision，持久Hub缓存续传，检查每个文件的Hub blob hash并记录SHA256，只离线加载config/tokenizer。GPU模型加载留到Slurm。尚未收到HPC下载完成结果。
WoS原始数据来源链已核验：HPT → HDLTex → Mendeley DOI 10.17632/9rw3vkcfy4.2；本轮尚未下载数据。

## uncased模型通过与下一步WoS原始资产

用户返回uncased revision `86b5e0934494bd15c9632b12f734a8a67f723594`，config.json/model.safetensors/tokenizer.json/tokenizer_config.json/vocab.txt均HASH_OK，最终WOS_SMOKE_MODEL_ASSETS_READY。tokenizer清理空格默认值FutureWarning不使本步失败。
用户再次明确raw cased与Experiment 1微调checkpoint的边界，已写入docs/experiment_boundaries.md；当前不实现GoEmotions adapter。
官方Mendeley V2文件接口已返回WebOfScience.zip，60222421字节，SHA256=b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a；ReadMe.txt，2587字节，SHA256=6da4ab1e624c4ef61c46600d727770f873ff4db3c15541fb481e7af8a715c0ec。HPC下载尚待返回，不能提前称数据READY。

## HPC原始WoS下载通过，预处理脚本本地真实数据验证通过

用户返回两项文件SHA256 OK、五项压缩包条目存在且大小匹配、WOS_RAW_DOWNLOAD_VERIFIED。HPC原始下载完成；HPC预处理未执行。
本地用相同SHA256的官方压缩包验证scripts/prepare_wos_assets.py：标准库XLSX转换46985数据行，全部46986行（含表头）与独立openpyxl读出的值完全一致，无单元格制表符/换行替换。
未修改的preprocess_wos.py和data_wos.py在独立临时目录执行成功。确认train/dev/test=30070/7518/9397、141标签、7父类/134叶类、单父两层无环、taxonomy与slot.pt一致、每条split记录与源Excel行映射一致、split ID无交集。
固定seed42选256train/128dev，完整taxonomy复制到smoke目录，记录source Excel行号；test不参与smoke。仅数据准备与检查，没有本地训练或GPU运行。原版源码仍干净。
HPC下一步执行同脚本；只有返回WOS_PREPROCESS_AND_SMOKE_ASSETS_READY才记HPC资产通过。

## HPC WoS预处理与smoke资产已通过，准备提交frozen作业

用户返回XLSX_CONVERTED 46985且所有列whitespace_changed_cells=0，两项UPSTREAM_PREPROCESS_PASS，train/dev/test=30070/7518/9397、141标签、7父类/134叶类、smoke256train/128dev、seed42、test_used_for_smoke=false，最终WOS_PREPROCESS_AND_SMOKE_ASSETS_READY。
新增scripts/run_frozen_smoke.py与slurm/lgsa_frozen_smoke.sbatch，先GPU GAT probe，再用未改动的上游Prompt/loss/evaluate与AST提取原data_map_function运行一个epoch。64次optimizer.step（原LR warmup第一步lr=0照实记录），验证128样本、最后checkpoint保存/严格重载/预测一致性检查。
资源：gpu4090、hengyanliu_fyp、qos4gpus、1GPU/4CPU/20000M/1小时。直接使用已有env，不安装依赖、不联网下载。邮箱使用用户明确提供的学校邮箱。
本地helper测试2项通过（原数据映射前缀/深度mask/512截断；几何残差诊断和退化拒绝）、Python/Bash语法通过。没有在Mac运行BERT smoke；GPU完整运行尚未发生，等待用户提交JobID及运行日志。

## Frozen Job 2984968：工程PASS，真实HGS map缺陷复现

用户提供sacct主任务/.batch/.extern均COMPLETED/0:0、GPU_PROBE_PASS（RTX4090、Torch2.5.1+cu118、CUDA11.8、PyG2.6.1）、64条训练记录和FROZEN_WOS_SMOKE_PASS。
backbone最大参数变化1.8552877008914948e-05；checkpoint重载logits一致；checkpoint存在、1341349472字节。验证Micro-F1=0.21775544388609716、Macro-F1=0.02554515045361315，只用于小样本工程检查，不能与GoEmotions baseline比较。
geometry: map_matches=false、fine_vectors_changed=false、max_norm_change=5.960464477539063e-08。确认原始反向map问题在真实WoS初始化中存在。
首step lr=0、alpha=0；第二step alpha=1/6且lr非零；最后lr=8.379888268156425e-07、alpha=1。符合保留原长warmup的smoke协议，不是充分训练后的benchmark。stderr只有tokenizer/torch.load FutureWarning与新增multiclass_bias初始化提示，无未解决异常。
证据摘要存references/frozen_smoke_2984968_user_evidence.json（标注用户提供，不冒称直接读取HPC）。

## Parent-map最小补丁与paired smoke准备

本地创建独立working_lgsa git仓库，branch=lgsa-goemotions-parentmap，commit=ba429c8984f22e32e9ec11641394fe5f1beebdc3。仅models/prompt.py一行方向修复加docs/code_changes.md；original_lgsa保持干净。HPC将按同脚本创建独立commit，实际hash由用户返回。
patched runner调用原run_frozen_smoke.run函数，并核对该文件SHA256与2984968记录一致；只替换工作源码入口及源码完整性验证器，不重写训练函数。相同rawuncased、256/128样本、seed42、batch4、1epoch、损失/调度和预测规则。
新增预GAT HGS验收：map正确、细类改变、top/sibling residual最大绝对cos<=1e-4、残差满行秩、norm/projection/residual norm保持（rtol1e-5/atol1e-6）、raw初始向量和原版逐元素一致。预设阈值，不按结果临时放宽。
本地3项测试通过（精确单行diff、真实AST几何方法/验收失败路径、源码检查器分派），无本地BERT训练。脚本准备好，HPC尚未提交patched作业；GoEmotions adapter仍未实现。


## 2026-09-23 patched提交与GoEmotions自动链条准备

用户返回：working_lgsa补丁209f83762dae1b354ba40e68b6952482ecf96208；
PATCHED_SMOKE_PREFLIGHT_PASS；Job2985243，PENDING(Priority)。尚未收到GPU完成证据。

用户明确采用sentiment-derived hierarchy v0，并授权排队期间实现adapter。
本地独立goemotions-adaptation仓库commit aad08704aed6f16fb8411df5edd9c24def48ea9f 已准备；HPC安装会记录自身commit。
11项CPU测试通过；全部54,263条官方记录的ID/多标签/祖先映射合同通过。
本地缺少PyG且Transformers不同于HPC，未声称完成真实模型集成；提交前
必须在HPC已验证的环境运行tiny random CPU集成测试，然后由Slurm验证raw cased模型。

slurm/paste_submit_goemotions_chain.sh为自包含复制命令，包哈希和shell语法通过。
下游尚未提交，GPU smoke/full均NOT RUN。


## 2026-10-05 收到完成证据（运行于2026-09-23）

用户提供2985243及2985311的主任务、batch、extern均COMPLETED/0:0。
Patched WoS用时1分15秒；工程/HGS/checkpoint reload通过。top cosine
6.9221e-7、sibling max8.4316e-7，小于预设1e-4。
GoEmotions链条用时17分39秒；smoke63步通过；full4epochs/10856步，
train43410、validation5426、test5427，best_epoch4，checkpoint重载通过。
GoEmotions HGS map_matches/fine_vectors_changed/norm_projection_preservation均true。

Test固定0.30：Micro-F1 59.457167、Macro-F1 53.285457、Rare mean F1
52.054360、Nonrare mean F1 53.695823（百分数）。验证集选择的阈值也为0.30。
证据归档在references/goemotions_2985311_user_evidence.json和user_terminal.txt。
尚未从HPC拉取全量产物；待核验哈希、raw初始化指纹、逐类差异与学习后几何。
单seed/多模块优化差异，不能宣称几何的单独因果效果或统计显著。
