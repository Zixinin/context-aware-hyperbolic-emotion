# 用户确认：Frozen upstream 与 parent-map patched 分开验证

确认日期：2026-09-22。这里记录本轮用户直接指示；附带HPC文档仅提供环境/调度参考。

## 最初收到的HPC证据（2026-09-22历史状态）

- 登录节点：`xpszlogin1`，当前shell激活base。
- `fypemotion` 已在Conda环境列表，`lgsa_goemo` 未出现在列表（尚不据此断言磁盘上绝不存在同名目录）。
- `/gpfs/work/aac/zixinfeng23/fyp` 与 `/gpfs/work/aac/zixinfeng23/.conda/envs` 存在且owner为用户。
- 尚无本实验GPU分配、安装或smoke运行证据。

## 两份HPC文档的适用范围

输入：`HPC_代码复现环境配置完整_SOP.pdf`、`HPC_Slurm_使用.pdf`，位于本地 `rgy 项目😼/HPC使用说明/`。
采用通用流程：独立环境、最小Conda再分阶段pip、绝对Python、清除污染变量、CUDA模块、实际GPU验证、sbatch提交、sacct和日志/产物验收。
文档中的hyperclip环境、Python3.8、torch1.13.1/cu117、Detectron2、RTX3090编译参数8.6和IQEclip训练命令是别的项目实例，不复制到LGSA。
“不要重建hyperclip”针对已经建立的HyperCLIP环境，不禁止为依赖不同的LGSA新建lgsa_goemo。
用户已指定正式训练通过Slurm。后续用户明确选择终端准备环境：截至2026-09-23，包安装和轻量CPU检查已完成；GPU验证与训练仍通过Slurm执行。先前统一分配GPU创建环境的方案已被替代。

## 按顺序验收

| 顺序 | 源码 | 任务目的 | 运行隔离 |
|---|---|---|---|
| 1 | `original_lgsa`固定4ea3dfb39dcee82654f443811471dd29be32e87e；0改动 | 原任务真实loader、model、forward、backward、checkpoint、validation工程测试 | `outputs/lgsa_official_smoke/frozen/JOBID/` |
| 2 | 工作分支从同commit起，仅改parent-map解包方向 | 同样工程验证，加HGS几何正确性验收 | `outputs/lgsa_official_smoke/patched_parentmap/JOBID/` |
| 3 | 第二步的已记录patch commit上开发 | GoEmotions模块化适配 | `outputs/lgsa_goemotions/smoke/JOBID/` |

frozen目录既不写模型补丁，也不写数据cache/checkpoint/__pycache__；运行资产和输出置于独立run目录，设置PYTHONDONTWRITEBYTECODE。若入口相对路径要求在运行目录布置源码，可生成该commit的字节一致执行快照，记录所有文件hash，不能暗中修改快照源码。

两次smoke使用相同官方数据subset与完整taxonomy、模型资产、seed、参数与harness。不得为了让patched看起来更好同时改loss、GAT、可训练权重、阈值或数据。
除输入/输出路径隔离等共同执行设施外，如原版存在启动阻断，保留失败结果并逐项定位；不能修改original后仍称frozen PASS。

## 几何验收的准确对象

- 检查真实edge列表与child→parent字典逐条匹配。
- 从真实label初始向量计算expected projection和residual，验证**GAT之前**的HGS顶层cosine、兄弟residual cosine和范数保持；确定数值tolerance并记录dtype。
- 全零、近零与rank deficient输入显式标记，不能把零向量的cosine输出当成功。
- GAT之后只记录正交偏离与四项HGA，不要求普通GAT严格保持初始化的零cosine。
- frozen原始映射的几何偏差照实保存；其工程PASS不等于几何正确。
- 两个JobID分别验收主任务/.batch的COMPLETED/0:0、训练进度和产物。

## Threshold维持双设置，不在此阶段改代码

Setting A：fixed sigmoid≥0.30，保证与实验1协议连续。
Setting B：validation-selected global threshold，验证集选择规则保持已记录的Macro-F1准则和候选网格，不用test调参。
BCE与LGSA multi-label categorical CE的logit calibration未必一致；两种设置都报告，不能只用0.30断定模型优劣。

## 现在下一步

2026-09-23已完成HPC源码冻结、独立环境安装与CPU检查，收到 `LGSA_ENV_CPU_READY`。下一步核验可复用的模型缓存并准备官方WoS数据，随后按上述门槛进行frozen与patched两次独立smoke；还未开始GoEmotions正式训练。
