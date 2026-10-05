# 一次排队：smoke 成功后在原 GPU 上执行完整 baseline

环境与 assets 已由 setup 作业完成后，新增启动器：

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/goemotions
mkdir -p logs
sbatch slurm/smoke_then_baseline_seed42.sbatch
```

此启动器申请一次 12 小时的单 GPU 资源。先用原定 1000/500 样本完成 1 epoch smoke；
检查该进程成功退出、输出哈希、checkpoint 保存与重载、相同 Job ID、源码、数据/model manifest、包版本和协议一致后，
在同一节点、同一作业内自动启动完整 4 epoch baseline。任一阶段失败都会使父作业失败，不继续下一阶段。

两阶段使用独立 Python 进程。完整 baseline 重新从原始预训练 BERT 初始化、重新设置 seed 42，
不延续 smoke 权重、优化器或随机数状态。它仍使用完整官方训练/验证/测试划分和原实验超参数。

同一 Slurm 作业在阶段交接时仍为 RUNNING，因此不能在作业内要求父作业 sacct=COMPLETED。
该启动器使用 smoke 进程退出码和完整 artifact 验证作为阶段交接条件；整个任务完成后才检查 sacct。
原来的分开提交入口和其 sacct 检查均未修改。

输出目录与原协议相同：

- smoke：outputs/smoke/seed42/jobJOBID/
- full：outputs/bert_baseline/seed42/
- 两阶段共同日志：logs/slurm-bert42-JOBID.out / .err

日志中依次出现 SMOKE_STAGE_START、SMOKE_STAGE_PASSED、FULL_BASELINE_START、SMOKE_AND_FULL_BASELINE_PASS。
不要另外提交 smoke_test.sbatch 或 baseline_bert_seed42.sbatch，避免与这个组合任务重复。
如已有 seed42 正式结果目录，会在 smoke 之前停止，不覆盖结果。

结束后：

```bash
sacct -j JOBID --format=JobID,JobName%22,State,Elapsed,ExitCode
source hpc/activate_env.sh
"$PYTHON" -m scripts.verify_run outputs/bert_baseline/seed42 --check-slurm
```

组合任务的 smoke 没有单独的 Slurm 作业或 smoke 日志文件，不能用独立 smoke 的 --check-slurm 命令验收它；
检查组合任务的共同日志、父作业退出状态和最终 baseline artifact 即可。

本次新增启动器完成 shell 语法与 7 项模拟阶段交接测试；模拟不运行模型。
首次真实 GPU 训练是否通过，仍以 HPC 的实际输出为准。
