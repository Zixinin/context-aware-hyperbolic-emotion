# Exp6C-0：RoBERTa 27-label 多标签 BCE baseline

仅 seed42 / 4 epochs；不自动启动 A1/A2 或额外 seed。Ubuntu 由用户执行。先环境检查，再 GPU smoke，验收 smoke 后才正式训练。

文件：common.py（固定配置/哈希保护）、data27.py（canonical 数据过滤）、model27.py（固定分类路径）、metrics27.py（完整预测与指标）、run.py（smoke/train/verify）、inspect_ubuntu.py（环境盘点）、launch.sh（通知/日志/锁/退出码）、tests.py（CPU 合成回归测试）。协议细节见 IMPLEMENTATION_AUDIT.md；原 prompt 见 USER_EXPERIMENT_PROMPT.md。

## 阶段1：环境和数据盘点

在 Ubuntu 的 VS Code Remote SSH Terminal 中操作。代码包由 Mac 的 scp 命令上传；具体校验/解压命令见随包交付的 UBUNTU_COMMANDS.md。这里假设包已安全解压到 FYP 的独立 experiments/exp6c0_roberta27_baseline 目录。

```bash
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
python3 experiments/exp6c0_roberta27_baseline/inspect_ubuntu.py
curl --fail --silent --show-error --connect-timeout 5 --max-time 15 --data-binary 'Exp6C-0 notification delivery test; no training started' https://ntfy.sh/zixin-fyp-4090-1119
```

把完整输出发给 Codex。需核对：D3020-A3、4090、GPU空闲、现有环境版本、唯一/指定的本地RoBERTa snapshot、canonical数据通过、远程AGENTS.md。通知测试须实际送达。此阶段不训练、不安装、不下载模型。

## 阶段2：GPU smoke（盘点验收后）

默认复用 FYP .venv-bert。若盘点证据表明应使用另一现有兼容环境，由明确的 EXP6C0_PYTHON 指定；不要直接改环境。

```bash
tmux new -s exp6c0_smoke
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
bash experiments/exp6c0_roberta27_baseline/launch.sh smoke
```

如果有多个本地 snapshot，使用盘点验收后确认的实际路径：`launch.sh smoke --snapshot /完整路径/40位commit`。不要自行猜路径或 revision。smoke 只用真实训练样本和验证样本，无 test 推理。

复制末尾 `RUN_ROOT=...`、`NEXT_AFTER_REVIEW:...`，并发送 `smoke/SMOKE_REPORT.json` 与退出码。出现任何错误停止，不运行正式训练。smoke会检查真实RoBERTa加载、27维targets、多标签保留、CUDA、BCE、encoder/head梯度、优化器、调度器和重载。

## 阶段3：正式训练（smoke验收后）

smoke成功后控制台会直接打印含真实run路径的完整命令：`NEXT_AFTER_REVIEW: bash .../launch.sh train .../run_...`。在同一 tmux 中复制该条 bash 命令即可；也可另建 tmux 会话后运行。不要复制占位路径。

开始后会发ntfy；日志在该run/train.log。离开保持运行：Ctrl+B，松开，再按D。返回：`tmux attach -t exp6c0_smoke`。

正式训练从原始预训练权重和seed42重新初始化，不沿用smoke权重。固定4epochs、7648steps、765warmup（以Ubuntu校验后的实际过滤样本动态计算）、阈值0.30。选轮仅用验证集，最佳权重全验证集重载核对后进行一次正式Test。

## 阶段4：验收与回传

训练成功必须同时看到最终成功标记、`train_exit_code.txt=0`、`SUCCESS.json`，并有完整产物。在原run路径执行 `bash .../launch.sh verify /实际run路径` 可重复CPU验收；不会重新推理或训练。先记录run路径后，复制真实路径到下面两个命令中：

```bash
cat /实际run路径/EXP6C0_SUMMARY.md
cat /实际run路径/VERIFICATION_REPORT.md
```

请回传summary、verification、environment、checkpoint_selection、checkpoint_reload_verification、training_log、test_metrics、rare_label_metrics、test_per_label_metrics、SUCCESS和退出码。完整NPZ预测、canonical IDs和checkpoint留在Ubuntu，不需要把大型checkpoint发进聊天。

若需打包小型验收文件，使用交付目录中的 `collect_review.py --run-root /实际run路径`。该工具先CPU复核成功产物，再生成唯一小包；不包含checkpoint与大型缓存。Mac下载命令会由它输出。

## 保留与失败规则

每次smoke创建唯一run目录；同一run正式训练只允许启动一次，失败也不自动重跑。先发FAILURE.json与phase日志定位；不要删除claim、覆盖结果或为了指标重训。GPU忙、历史源码哈希变化、数据身份不可靠、环境变化、重载不一致均阻断训练/成功标记。日志中的通知失败不会吞掉真实实验退出码。

27-label过滤测试集与既有28-label/单标签测试协议不同，不直接计算跨协议提升。该baseline完成即停止。
