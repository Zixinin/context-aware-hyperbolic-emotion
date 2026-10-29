# 2026-10-29 最新用户授权：无外接盘，先训练，随后取回 Mac

本版本只调整备份编排。用户明确要求现在开始跑本轮实验，完成后取回Mac；原prompt的外接盘强制门由这项新要求替代。未做外接盘备份，旧外接盘完整性未经当前验证，Mac副本在实际下载和SHA校验前一律PENDING。不是声称已经完成迁移，也不声称旧Exp6已上GitHub。

新的独立源码目录：experiments/exp6e3_mac_transfer_v1_1。旧Exp6E-3源码/安装包和所有已有实验不改写。结果仍写独立的Exp6E-3 run目录；两个版本共用重复训练保护与GPU锁，不自动重试已有PLAN。

科学部分与原交付完全一致：只降低coordinate auxiliary整体系数至0.02091538317504222，s=1，seed42，从相同预训练权重独立初始化，smoke后四轮7648步，固定.30，Validation Macro-F1选轮，完整3834验证重载，不新增Test或seed，不改gamma。mrun/mmodel/mprobe/mreport逐字节与原交付相同；只修改mcommon备份门和mprepare的备份状态字段。

训练完成后封存run.log/exit code，单独生成包含本轮完整run、源代码及8个冻结依赖源码目录的下载tar.gz、逐文件SHA清单、归档SHA以及Mac安全提取/目标校验程序。包括best checkpoint及optimizer/RNG状态、NPZ/NPY、CSV、报告、配置、所有运行日志；仅排除可重建runtime_cache/Python字节码。该包是本轮实验产物，并非旧项目全量备份，也不是原始预训练模型与完整数据集的迁移包。

下载包保存在repo外，导出失败不会重训或撤销科学完成状态。Mac从既有ubuntu4090拉取，Ubuntu不会主动SSH到Mac。只有Mac实际读取目标全部文件并验SHA后才打印MAC_LOCAL_BACKUP_VERIFIED_PASS。代码及小型结果以后再整理GitHub，本版本不commit或push。外接盘状态始终NOT_ATTEMPTED_USER_AUTHORIZED_DEFER。

以下是原科学协议和原始备份要求的历史记录；备份编排以上面的最新授权为准：

# Exp6E-3：A1-C-Gamma单种子对照与强制迁移备份

本轮只训练一个seed42模型，且在外接盘预备份完成之后训练。没有新Test、其他seed、重新校准、s=.03、N1、其他Loss、改层级或Euclidean训练。禁止自动git commit/push、安装依赖、删除旧备份、改系统配置或定时任务。

## 科学变量

原A1：L=BCE+.1 mean_i(C_i−D_i)。本轮：L=BCE+0.02091538317504222 mean_i(C_i−D_i)。

q_ij=y_ij/sum_j y_ij；C_i=−sum_j q_ij log softmax(u_i)_j；D_i=sum_j q_ij z_iᵀe_j。全部正标签平均，再完整batch平均，BCE按B×27平均。只改变整个aux系数，CE内部/coordinate dot及其符号/归约不变。mmodel.py直接包装原A1，无测地对齐。

保持RoBERTa/tokenizer/27顺序/固定27×100标签先验/c=1/HNN/切空间decoder/分类头/s=1；同30587/3834/3821分割、seed42、batch16/eval64、maxlen50、四轮7648步、warmup765、AdamW5e-5/原WD/linear scheduler、clip1、FP32/eager、阈值.30与Validation Macro-F1选轮（平局早轮）。从同raw pretrained权重独立初始化，不续训旧checkpoint。

原A1和本轮共享/HNN初始参数与初始分类logits核对；train/eval logits/z/decoder/BCE/CE/dot/未加权aux/positive count/q一致。新weighted aux的参数梯度向量应为旧.1 aux梯度向量×gamma_G/.1。分类BCE路径不变；固定标签无梯度；真实fresh40模块/样本组与旧完整精度统计replay。不能宣称更新后随机轨迹和权重相同。

先做两步丢弃smoke及64条Validation重载，再fresh重置正式训练四轮。记录每batch Loss/CE/负dot/weighted分量、未裁剪norm、真实proj clip和半径。每轮同train-only160位置/145 IDs做无更新eval梯度/几何探针，恢复mode/RNG/state/.grad；记录BCE–aux方向与CE/negative-dot分量，不按探针改gamma。非有限、球外、错误路径或parity失败停止保留证据；已知s=1截断如实记录，不修设置。

完成四轮按Macro选best并完整3834条Validation reload。对照A0/A1/A1-C-Gamma/A1-G-C，分别报告降低gamma效应、同gamma目标比较、对A0的差异；全部27类F1/AP及Rare7、上升/下降数、grief/relief的Macro贡献都保留。同gamma不等于同梯度强度；单seed探索性Validation不能证明显著、泛化或双曲优于欧氏。

best_checkpoint/training_state.pt保存选中轮optimizer/scheduler/CPU+CUDA/Python/NumPy RNG及DataLoader generator；不另存最终轮权重/状态，不提供自动续训，保存边界写入TRAINING_STATE_SCOPE.json。

## 强制备份与迁移

指定用户目录：`/media/zixinfeng/外接硬盘/fzx`。必须通过findmnt确认独立于Ubuntu根盘的/dev设备、rw挂载、设备身份、可写状态、容量；目录存在不算挂载。挂载或空间失败则不开始训练，也不猜替换设备。

专用备份根：`/media/zixinfeng/外接硬盘/fzx/FYP_context-aware-hyperbolic-emotion_verified`。首次创建项目所有权marker；存在且marker匹配才复用，否则停止，避免覆盖他人项目。

备份整个repository（包括.git、checkpoint、NPZ/NPY、CSV、日志、配置、标签映射和数据），另备份原HypEmo代码/先验、HypEmo-Exp6B和既有huggingface模型来源。只排除Python虚拟环境、__pycache__和.pyc；不使用Git ignore，不按二进制扩展名过滤。列出所有符号链接并物化为真实文件；未审阅根之外的链接、环、坏链接或特殊文件停止报告，不默默跳过。Git工作树外部元数据按.git指针另纳入。

rsync增量，绝不--delete；覆盖变化文件前用--backup-dir保留旧版。先全源哈希/容量计划（增量+8GiB余量），再复制、独立读取每个目标文件验证size/SHA、核对源无变化和挂载身份、数量/逻辑总字节。rsync任何非零立即停止。完整manifest、路径映射、symlink覆盖、排除列表、rsync日志及verification保存于repository之外的`/home/zixinfeng/Documents/FYP_migration_records`，并复制到目标records目录，避免自复制。

预备份包含本轮源码、旧A1-G-C、Exp6D-2和所有现有结果；prepare/smoke/train均再核对挂载和关键副本。训练后停止源run.log写入，保存科学exit_code=0，再对静止源做完整后备份。实验完成和备份完成是独立状态；后备份失败时旧科学SUCCESS不撤销，但pipeline非零且不能宣布备份成功。

Mac不能被假设可由Ubuntu主动SSH连接。MAC_BACKUP_TRANSFER_INSTRUCTIONS.md提供Mac通过ubuntu4090拉取payload/manifest并在目标逐文件SHA验收的命令；之前一律MAC_LOCAL_BACKUP=PENDING_USER_TRANSFER。外接盘PASS绝不冒充Mac PASS。

backup.sh供之后手动重复全量范围增量备份，不训练、不删历史、不设后台任务。一次pre备份不能代替包含新checkpoint的post备份。

## GitHub范围

用户计划之后保存到Zixinin/context-aware-hyperbolic-emotion；当前任务不自动push。github_prepare.py只在仓库外生成源码/小型聚合结果待审包、大小和哈希清单，并筛查显式密钥，不修改Git index/branch/commit/.gitignore。现有.gitignore排除checkpoint及NPZ，所以代码托管不能替代完整迁移备份。公开上传前须核对远端新head及清单，不用git add .或强推。

## 成功条件

科学验收后才打印EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE；目标盘真实校验后才打印EXP6E3_EXTERNAL_BACKUP_VERIFIED_PASS，标明pre/post。Mac只有在其本机目标校验后才PASS。最终状态同时列科学指标、reload/历史保护、外接盘、Mac pending及真实源/目标路径。
