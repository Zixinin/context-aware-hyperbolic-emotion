# Exp6C-1 A1：RoBERTa27 + BCE + HypEmo-inspired auxiliary

只授权seed42/4epochs。A0冻结目录：`results/exp6c0_roberta27_baseline/run_20261020T033547Z_vQWCGEh8`。A1独立代码：`experiments/exp6c1_roberta27_hypaux`；独立结果：`results/exp6c1_roberta27_hypaux/run_<UTC>_<random>`。

训练前先读 LOSS_SPECIFICATION.md。使用与Exp6B已验收Formula A完全一致的辅助值/梯度，但排除其距离加权分类部分；实际目标只有普通mean BCE+0.1×aux。HNN/decoder/固定标签向量就地只读复用，不复制第三方源码进FYP。

用户在Ubuntu终端执行；Codex不连接远端。代码上传/安全解压/盘点命令随交付包提供。现有 `.venv-bert` 必须与A0实际记录完全一致；不安装任何依赖、驱动或CUDA，不改系统Python。

## 1. 安装与盘点

Mac运行交付目录的UPLOAD_FROM_MAC.sh，然后Ubuntu运行Exp6C1_stage1_20261020_v1.sh。这一步用标准库核对源码/既有A0成功记录与环境元数据，检查GPU、tmux和通知；不训练。把完整输出发回核对。遇到远程AGENTS.md也一并返回。

## 2. 真实GPU smoke

盘点通过后，在tmux中运行：

```bash
cd /home/zixinfeng/Documents/context-aware-hyperbolic-emotion
bash experiments/exp6c1_roberta27_hypaux/launch.sh smoke
```

内部先验证A0封存、原始数据/IDs、原snapshot、Exp6B辅助公式/向量/原作者源码身份，然后真实RoBERTa/HNN CUDA smoke。逐张量校验A0历史初始锚点及A1初始shared state；train/eval初始logits精确相同；3种真实样本case检验辅助公式/归约/梯度、球内数值和BCE/辅助尺度；执行两次优化器迭代并重载smoke checkpoint。正式训练与Test推理均为0。

输出与文件：`INITIALIZATION_PARITY.json`、`smoke/LOSS_AND_GRADIENT_CASES.json`、`smoke/SMOKE_REPORT.json`、`smoke/SMOKE_VERIFIED.json`、`smoke_exit_code.txt`。smoke必须退出0且证明完整。回传实际loss/梯度报告；不因loss比例或预期性能自动改0.1或公式。

## 3. 正式训练

smoke验收后，使用控制台打印的NEXT_AFTER_REVIEW准确命令启动。不能猜run目录或复用smoke权重。正式启动前再检查所有A0/上游/数据/模型/代码hash；fresh raw snapshot/seed42；4epochs、7648steps、765warmup、batch16、maxlen50、AdamW5e-5/WD0.01、clip1、FP32/eager、fixed0.30、validation Macro-F1选轮，精确tie取早轮。

整轮first-batch IDs还会逐轮对照历史A0。双曲分支保留原dropout，构造隔离RNG；训练前向新增分支会额外消耗随机数，后续dropout掩码轨迹不保证逐位等于A0。

开始/成功/失败通知已内置。日志在run/train.log。退出tmux保留运行：Ctrl+B松开再D；返回：`tmux attach -t exp6c1_smoke`。4轮训练后，全3834验证样本重载确认，再一次正式Test；不额外Test、不调阈值。

## 4. 验收与对比

成功标记：`EXP6C1_ROBERTA27_HYPAUX_VERIFIED`。必须同时有train退出0、SUCCESS.json以及完整结果；通知本身不是独立验收证据。

输出包括config/environment/dataset manifests、fixed vectors身份、原始源码hash、A0 baseline proof、initial parity、逐batch和逐epochBCE/aux/total/ratio/梯度/球norm、checkpoint与selection、full validation reload、完整validation/test NPZ、全部27类TP/FP/FN/support/Precision/Recall/F1、Rare-7及matched比较CSV、EXP6C1_SUMMARY.md、VERIFICATION_REPORT.md、SUCCESS.json。

`launch.sh verify /准确run路径` 只读CPU复核保存预测，不加载模型、不重跑Test。

`CUDA_VISIBLE_DEVICES='' .venv-bert/bin/python experiments/exp6c1_roberta27_hypaux/collect_review.py --run-root /准确run路径` 先验证，再生成唯一小包与Mac下载命令；不包含大checkpoint和缓存，包含A0/A1配对预测及诊断，原始A0保持只读。

比较以相同27标签与canonical IDs为基础；Micro/Macro/Weighted/Rare-F1及P/R均来自同一固定阈值。单seed只能报告本次配对差异，不能称已隔离几何本身的因果收益。

## 5. 失败与停止

任何公式、归约、来源、映射、shape、环境、数值、球界、初始共享state/logit或封存hash不符均停止并写FAILURE.json。正式run独占claim，不覆盖、不自动重试；先回传phase日志与错误原因。不要删claim、改旧A0清单或为修复失败而另起更多seed。

完成Exp6C-1后停止。不自动进入距离加权、ASL、Graph、新层级或更多seed。
