# GoEmotions 自动衔接准备

用户于 2026-09-23 要求当前 patched WoS smoke 成功后自动继续后续实验，减少人工等待。

当前上游 Job 为 `2985243`；用户返回状态为 `PENDING (Priority)`，不是运行通过。
HPC 已提交的最小补丁 commit 为 `209f83762dae1b354ba40e68b6952482ecf96208`。

计划用 Slurm `afterok:2985243` 依赖一个 GoEmotions 作业。依赖满足仅代表允许调度；仍需等待 GPU 资源，不能保证立即运行，也不预留 GPU。

后续作业内按顺序执行：

1. `scripts/check_patched_smoke_success.py` 只读核验指定上游作业的 SUCCESS、HGS 几何验收、64 步更新、checkpoint 重载及哈希、源码 commit 和哈希。
2. 从原始 cased BERT 开始 GoEmotions 1000 train / 500 validation / 1 epoch smoke。
3. 自动验收 GoEmotions smoke；失败立即以非零退出码结束，不能启动 full run。
4. 用新进程重新加载原始 cased BERT，开始 seed42 full run。不得加载 baseline、WoS 或 GoEmotions smoke 的训练 checkpoint。

步骤 2–4 共用一次 GPU 分配，输出目录独立。每个阶段分别保存 metadata 和成功证据。正式训练的断点恢复仅允许恢复同一 full run 的 checkpoint。

实现状态：用户已采用sentiment-derived hierarchy v0并授权实现；adapter、smoke/full、独立硬验收与Slurm脚本已准备。11项本地CPU测试通过。HPC tiny-random预检查、GPU smoke和full尚未执行，尚未提交下游作业。

Slurm 官方说明：https://slurm.schedmd.com/sbatch.html#OPT_dependency
`afterok` 要求上游退出码为零。上游失败时依赖任务不会运行；上游已结束太久时可能超出 MinJobAge 而无法新建依赖，此时须重新核验会计记录和完整产物后再决定提交方式，不能改用 afterany 绕过成功条件。
