# FYP 第二个实验：LGSA-GoEmotions Adaptation

**当前完成：独立工程和来源冻结、论文/源码理解、接口方案、候选层级、CPU组件审计。尚未在HPC安装环境或运行LGSA。**

实验1 = Standard BERT GoEmotions Baseline；实验2 = 本工程。若论文采用从0编号，可另标 E0/E1，但本项目对外称“第一个/第二个实验”，避免混淆。
目标是理解并验证 LGSA 的 Euclidean hierarchy-aware geometry 在 GoEmotions 上是否有效。

## 先读这三份

1. [subspace的精确定义](docs/lgsa_subspace_explanation.md)：一句话、公式、反例和代码差异。
2. [15项论文到代码映射](docs/lgsa_code_map.md)：调用链、文件/函数/行号/shape。
3. [GoEmotions接口方案](docs/lgsa_goemotions_compatibility.md)：祖先标签、多标签、28类评价与公平性。

其余入口：[完整执行和验收表](docs/execution_tracker.md)、[上游问题](docs/upstream_findings.md)、[候选hierarchy](docs/goemotions_hierarchy_v0.md)、[版本](docs/original_lgsa_version.md)、[环境说明](environment/environment_notes.md)。

## 不跳步的执行顺序

`冻结来源 → 理解HGS/HGA → HPC独立环境 → frozen upstream原任务smoke → 仅修parent map的独立版本smoke → 审阅hierarchy/接口 → adapter → GoEmotions smoke → seed42 full → 公平比较 → 一个后续建议`。

2026-09-22用户进一步确认：[原版与最小补丁的独立验证规则](docs/frozen_and_patched_smoke_protocol.md)。原版smoke只说明工程可运行；正式适配以有记录、通过几何验收的parent-map最小修复版为起点。

本轮依据用户材料末尾“先做到Phase0–3，再审方案”的要求：没有启动full training，也没有提前实现adapter。官方smoke尚待HPC资产与环境，不把当前源码分析称为Phase0–3全部完成。

## 已核验的baseline

来自已有归档 `goemotions_bert_seed42_job2968116_20260921_141855.tar.gz`，仅读取原工程。8个小文件复制到 `references/baseline/` 并逐项匹配归档 `SUCCESS.json` 内SHA256。

| Model | Micro-F1 | Macro-F1 | Rare-F1 | Non-Rare-F1 |
|---|---:|---:|---:|---:|
| Standard BERT，seed42，fixed0.30 | 58.72 | 48.62 | 36.00 | 52.83 |
| LGSA-GoEmotions | 尚未运行 | 尚未运行 | 尚未运行 | 尚未运行 |

单位为%。原始精确值和per-label结果见reference文件。归档记录job2968116；本轮没有重新查询该job的实时sacct。

## 当前下一步：HPC只读现状

由你在学校HPC登录终端执行并把输出贴回；我不代上传或提交。当前只进行这一检查，不要求你一次执行后续全部步骤：

```bash
hostname
conda env list
ls -ld /gpfs/work/aac/zixinfeng23/fyp /gpfs/work/aac/zixinfeng23/.conda/envs
```

后续才根据返回逐步给命令：创建独立HPC目录、clone同一commit、部署本阶段小文件、确认module/Slurm资源、独立env setup、官方WoS资产与smoke。
本地已创建的文件**不会自动出现在HPC**；上传/粘贴步骤必须明确完成后才可使用下面的脚本路径。

## 候选环境步骤（待现状检查后执行）

目标根：`/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions`。
环境：`/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo`。
源commit：`4ea3dfb39dcee82654f443811471dd29be32e87e`。

1. 在HPC根中保留干净 `original_lgsa` clone，checkout上述commit；工作副本另建分支，原源码不改。
2. 复制/粘贴已审阅脚本，先 `bash -n`，核对学校module/资源；设置自己的真实邮箱。
3. 用 `bash slurm/submit_checked.sh environment` 通过sbatch创建/验证独立环境，绝不 `bash xxx.sbatch` 当作训练方式。
4. 检查任务，再看GPU probe与pip lock文件。官方LGSA smoke是另一项任务，环境通过不等于模型通过。

所有安装使用绝对环境Python的 `-m pip`。不覆盖baseline环境。

## 邮件

标准写法如下（仅示例，不可照着占位符提交）：

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<YOUR_XJTLU_EMAIL>
```

本项目采用更安全的提交参数方式：设置真实 `LGSA_EMAIL`，`submit_checked.sh`校验后传 `sbatch --mail-user="$LGSA_EMAIL"`；sbatch文件自身不存邮箱占位符。
BEGIN对应真正开跑；END/FAIL报告结束/失败。邮件依赖学校Slurm mail服务，最终成功仍看sacct、日志和产物。

## 每次任务状态检查

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,JobName%24,State,Elapsed,ExitCode
```

检查本次JOBID对应的.out/.err中 `Traceback|CUDA out of memory|Killed|Segmentation fault|Exception`，并逐项查看产物。不是所有stderr内容都等于失败；warning要结合实际结果判断。
后续官方smoke、GoEmotions smoke、full脚本会在各自前置门槛达到时创建，避免误提交。

## 可重做的本地审计

只在本工程运行 `python3 tests/audit_upstream_geometry.py`：小CPU张量，不下载BERT、不运行训练loop。
其结果在 `docs/upstream_geometry_audit.json`；冻结源码不改。`scripts/capture_baseline_reference.py`仅从已有归档核验/复制小文件。

## 最终交付（运行后）

按执行表生成 `experiment_metadata.json`、环境lock、唯一JOBID日志/输出、全量指标、两种threshold、per-label deltas、六难类计数、几何diagnostics与PCA/heatmap。
最终报告包含要求A–M，选择一个后续实验建议；消融与Hyperbolic仅在用户后续明确授权后启动。
