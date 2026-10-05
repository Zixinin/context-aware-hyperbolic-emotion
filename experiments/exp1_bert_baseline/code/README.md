# FYP 实验 1：GoEmotions 标准 BERT 多标签 baseline

课题：Hyperbolic Representation Learning for Multi-label Emotion Recognition。
本实验仅建立标准 BERT 参照点。没有双曲层、GNN、语义子空间、类别权重或数据增强。

**当前交付状态：代码与 HPC 提交流程已准备；真实 smoke/full 是否完成必须看学校 Slurm 日志与实验输出。不能把本地检查当成训练成功。**

## 1. 固定路径与环境

```text
PROJECT_ROOT=/gpfs/work/aac/zixinfeng23/fyp/goemotions
CONDA_ENV=/gpfs/work/aac/zixinfeng23/.conda/envs/fypemotion
PYTHON=/gpfs/work/aac/zixinfeng23/.conda/envs/fypemotion/bin/python
HF_HOME=/gpfs/work/aac/zixinfeng23/.cache/huggingface
```

所有模型命令均使用绝对环境 Python。脚本中的路径可以通过环境变量覆盖；代码内部路径相对项目根目录。
不会复用或修改 `hyperclip`。不安装 Detectron2、CLIP、open_clip、torchvision 或 torchaudio。

首次安装采用 Python 3.10、PyTorch 2.6.0+cu118、Transformers 4.51.3、Datasets 3.5.0，其他固定版本见 requirements。
学校模块为 `cuda/11.8.0-gcc-8.5.0-d7ndetl`；驱动的 `nvidia-smi CUDA Version` 与 torch 自带 runtime 是不同信息。
官方 cu118 安装说明：https://pytorch.org/get-started/previous-versions/#v260

## 2. 上传到已经创建的 fyp 文件夹

可以在学校网页文件管理器中，把交付的 `goemotions_hpc_baseline.tar.gz` 上传到：

```text
/gpfs/work/aac/zixinfeng23/fyp/
```

然后在 HPC Terminal 执行：

```bash
cd /gpfs/work/aac/zixinfeng23/fyp
tar -tzf goemotions_hpc_baseline.tar.gz | head
# 首次部署、确认没有同名代码目录时解压：
tar -xzf goemotions_hpc_baseline.tar.gz
cd /gpfs/work/aac/zixinfeng23/fyp/goemotions
sha256sum -c MANIFEST.sha256
mkdir -p logs
```

如果用 SSH/SCP，先连接学校 VPN，在本地终端切到压缩包所在目录再执行：

```bash
scp goemotions_hpc_baseline.tar.gz zixinfeng23@xpszlogin2.xjtlu.edu.cn:/gpfs/work/aac/zixinfeng23/fyp/
ssh zixinfeng23@xpszlogin2.xjtlu.edu.cn
```

无需在聊天里提供密码。所有下面的运行命令均在学校 HPC 执行。
如目录已有其他文件，先检查，不要用解压覆盖已有实验。

## 3. 第一次：环境安装、GPU 验证、缓存模型和数据

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/goemotions
mkdir -p logs
sbatch slurm/setup_assets.sbatch
```

记下输出的 SETUP_JOB_ID。用真实数字替换命令中的占位符：

```bash
squeue -u "$USER"
tail -F logs/slurm-setup-SETUP_JOB_ID.out
# Ctrl+C 结束 tail，不影响后台作业
sacct -j SETUP_JOB_ID --format=JobID,JobName%18,State,Elapsed,ExitCode
```

setup 作业在分配的 GPU 节点上：创建最小 Conda 环境、分阶段安装、`python -m pip check`、检查完整 import 链和 CUDA、执行轻量逻辑测试、缓存 BERT 与官方 GoEmotions、完成数据统计。
环境通过检查后写入完成标记；以后重复 setup 会验证环境并复用，不重新创建或重装。
若首次安装中断，重提交 setup 即可继续安装未完成部分。

环境记录为 `environment/pip-freeze.txt`、`python-version.txt`、`torch-info.txt`、`torch-info.json`。
只有 **COMPLETED / 0:0** 且日志含 `SETUP_AND_ASSETS_PASS` 才进入下一步。

默认数据加载为：

```python
datasets.load_dataset("google-research-datasets/go_emotions", "simplified", revision=resolved_commit)
```

准备时解析模型/数据 `main` 对应的具体 commit 并保存在 `assets/manifest.json`；后续实验用这一固定缓存，离线运行。
官方数据集：https://huggingface.co/datasets/google-research-datasets/go_emotions
模型：https://huggingface.co/google-bert/bert-base-cased

如果 **数据集** 的 Hugging Face 下载不可用，可以显式改用包内原版 TSV：

```bash
sbatch --export=ALL,DATA_SOURCE=official-tsv slurm/setup_assets.sbatch
```

TSV 逐文件校验 SHA-256，保留原始划分和标签顺序，来源差异写入 metadata；不会悄悄自动切换。
此选项仍需能下载 BERT 权重。若集群完全无法访问 Hugging Face，需先在有网络的机器准备官方模型快照并上传，不能凭空跳过模型下载。
若 `assets/manifest.json` 已存在，必须沿用其中来源；不要在同一实验目录随意切换。

## 4. 数据分析

setup 已自动生成 `outputs/dataset_analysis/`：

```text
label_frequencies.csv       所有 28 类的 ID、名称、三种 split 频次
dataset_statistics.json    样本数、平均标签数、单/多标签百分比、min/max
rare_labels.json            仅按完整训练集频次选最低 7 类
label_frequency.png
```

以后要再次查看：

```bash
source hpc/activate_env.sh
"$PYTHON" -m scripts.analyze_dataset
```

强制检查官方样本数 43,410 / 5,426 / 5,427、标签合法性、评论 ID 不跨 split 重复。
稀有类按训练频次升序、同频时 label_id 升序取 7 类，smoke 也使用完整训练集定义。
统计测试标签频次仅用于用户要求的数据描述，不进入模型、checkpoint 或阈值选择。

## 5. 先提交 smoke test

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/goemotions
mkdir -p logs
sbatch slurm/smoke_test.sbatch
```

记下 SMOKE_JOB_ID。smoke 使用原训练划分中的 1,000 条和原验证划分中的 500 条，seed 42 确定子集，1 epoch；不重新划分数据，不评估测试集。
检查加载、动态 padding、float32 multi-hot、CUDA 前向、BCE 数值、反传、梯度、指标、checkpoint 保存与重新加载、CSV/JSON 和图表。
输出：`outputs/smoke/seed42/jobSMOKE_JOB_ID/`；指标仅用于工程检查。

```bash
tail -F logs/slurm-smoke-SMOKE_JOB_ID.out
tail -F logs/slurm-smoke-SMOKE_JOB_ID.err
sacct -j SMOKE_JOB_ID --format=JobID,JobName%18,State,Elapsed,ExitCode
source hpc/activate_env.sh
"$PYTHON" -m scripts.verify_run outputs/smoke/seed42/jobSMOKE_JOB_ID --check-slurm
```

最后一条需在作业结束后运行。`sacct` 可能稍有更新延迟；不要在 RUNNING 时判为训练失败。

## 6. smoke 成功后提交正式 seed 42

用真实数字替换 SMOKE_JOB_ID：

```bash
bash hpc/submit_baseline.sh SMOKE_JOB_ID
```

它会核对 smoke 的 Slurm **COMPLETED / 0:0**、所有结果文件哈希、28 类指标、18 个阈值、训练轮数、checkpoint、日志严重错误，并要求源码/配置、缓存 manifest、Python 包与 smoke 一致，然后执行 `sbatch`。
正式作业开始时还会再核对一次。不能拿旧版本代码的 smoke 结果批准新代码。

等价的手动提交流程为：

```bash
source hpc/activate_env.sh
export SMOKE_RUN="$PROJECT_ROOT/outputs/smoke/seed42/jobSMOKE_JOB_ID"
"$PYTHON" -m scripts.verify_run "$SMOKE_RUN" --gate
sbatch --export=ALL slurm/baseline_bert_seed42.sbatch
```

正式输出固定在 `outputs/bert_baseline/seed42/`。已有目录时拒绝覆盖。若之前失败，检查日志后把那个具体目录改名归档，再重新提交；没有自动续训或悄悄重启。
本项目只提交一个 seed，不自动启动多种子任务。

## 7. 训练与评估协议

| 项目 | 设置 |
|---|---|
| 模型 | bert-base-cased，标准 pooler + dropout + linear head |
| 标签 | 28 维 float32 multi-hot；BCEWithLogitsLoss |
| 输入 | max_length=50，截断，动态 padding，保留原文本 |
| 训练 | 4 epochs，seed 42，单卡 FP32 |
| Batch | 每卡 16，accumulation 1，effective 16 |
| 优化 | AdamW，LR 5e-5，weight decay 0.01 |
| 其他 | warmup ratio 0.1，linear decay，clip norm 1.0 |
| 模型选择 | 每个 epoch 后按固定 0.30 的验证集 Macro-F1，平分保留较早 epoch |
| 主结果 | sigmoid(logits) >= 0.30 |
| 附加结果 | 在最佳 checkpoint 的验证预测上搜 0.05 至 0.90、步长 0.05 的单一全局阈值 |
| 阈值选择 | 验证 Macro-F1 最大；完全同分取较低阈值 |
| 测试 | 阈值选择保存后，只跑一次测试预测，并从同一份 logits 报告两个协议 |

没有 softmax、CrossEntropyLoss、BCE 前 sigmoid、每类阈值、强制 top-1 标签或早停。
Macro 指标始终对全部 28 类求平均，未预测类别使用 `zero_division=0`。
`per_label_metrics.csv` 默认为**测试集固定 0.30**、F1 从低到高；另存两种阈值和两种 split 的明细。
`rare_label_metrics.json` 分别报告 rare/non-rare 的按类别平均 F1。

FP32、关闭 TF32、eager attention、固定随机种子、确定性算法（遇到不支持的算子报错）。
AdamW 的 bias/LayerNorm 不做 weight decay；beta=(0.9,0.999)，epsilon=1e-8。
warmup steps=ceil(总 optimizer steps × 0.1)。不丢弃尾批：正式训练每轮末尾 2 条，梯度按实际样本数归一化，这一点记录在 environment.json。

只有发生 OOM 才显式修改 `config/baseline_bert.json` 的 batch 为 8、accumulation 为 2，然后重新 smoke，再提交 full。
不自动降 batch，不改变 effective batch、学习率或训练轮数。此修改及实际参数会完整记录。
以后可通过 `src.train --seed` 和新输出目录扩展其他 seed；需先完成同配置 seed 的 smoke。当前启动脚本只管理 seed 42。

## 8. 每次以后登录 HPC

```bash
cd /gpfs/work/aac/zixinfeng23/fyp/goemotions
source hpc/activate_env.sh
echo "$CONDA_PREFIX"
"$PYTHON" --version
"$PYTHON" -m pip check
squeue -u "$USER"
```

不用重建环境，也不用换学校 Jupyter kernel。Jupyter Terminal 只负责提交任务。
`scripts.check_environment` 的 GPU 检查要在分配的 GPU 节点运行，不要求登录节点 `CUDA available=True`。
绝不能使用 `bash slurm/smoke_test.sbatch` 或 `bash slurm/baseline_bert_seed42.sbatch` 启动训练。

## 9. 监控与完成验收

```bash
squeue -u "$USER"
watch -n 30 'squeue -u "$USER" -o "%.20i %.20j %.10T %.12M %.12l %R"'
tail -F logs/slurm-bert42-BASELINE_JOB_ID.out
tail -F logs/slurm-bert42-BASELINE_JOB_ID.err
sacct -j BASELINE_JOB_ID --format=JobID,JobName%18,State,Elapsed,ExitCode
grep -nE 'Traceback|CUDA out of memory|Killed|Segmentation fault|Exception' logs/slurm-*.err
"$PYTHON" -m scripts.verify_run outputs/bert_baseline/seed42 --check-slurm
```

tqdm 通常写入 stderr，`.err` 非空不等于失败。grep 没找到时退出码 1 是正常结果。
成功标准：主任务和子步骤 COMPLETED / 0:0；checkpoint 与全部指标存在且哈希通过；日志无严重错误；训练轮数和 optimizer steps 完整。
任务从 squeue 消失不能证明成功，用 sacct 查历史。Slurm batch 提交成功后可以关闭浏览器和本地电脑。

## 10. 结果与论文记录

```text
outputs/bert_baseline/seed42/
  config.json                     实际运行超参数
  environment.json                GPU / job / 版本 / 数据来源 / commit / 源码哈希 / 起止时间
  pip-freeze.txt
  training_log.csv                 每轮 mean BCE loss 和验证指标
  validation_metrics.json
  test_metrics.json               fixed_0.30 与 validation_selected 分开
  threshold_search.csv            验证集 18 个候选的六个指标
  threshold_selection.json
  per_label_metrics.csv           固定 0.30 的测试集结果
  {validation,test}_per_label_{fixed_0.30,validation_selected}.csv
  rare_label_metrics.json
  validation_predictions.npz      最佳 checkpoint 的验证 logits/targets/IDs
  test_predictions.npz            唯一一次测试预测
  dataset_analysis/
  checkpoints/best/               model.safetensors + tokenizer + selection.json
  figures/                        要求的五张图，及 validation/test 分目录图
  summary.md                      自动生成可用于汇报的实际研究摘要
  SUCCESS.json                    文件校验清单；仍需 Slurm 完成证据
```

没有 Git 时 commit 为 null，用源码逐文件 SHA-256 追踪代码。数据/model 的固定版本和哈希由 assets manifest 记录。
可选 PCA 本次未启用；不影响本实验完成标准。

## 11. 本地允许的检查

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
bash -n hpc/setup_environment.sh
bash -n slurm/smoke_test.sbatch
```

本地仅静态检查、完整 import 和小规模合成 CPU 逻辑。**不在 Mac 运行真实 BERT smoke 或 full 训练**。
所有模型前向、反向和真实评估均在 Slurm GPU 作业内进行。

## 12. 实现选择与资料边界

两份学校 PDF 用于核对模块、路径隔离和 Slurm 参数。PDF 中 HyperCLIP/Detectron2 的旧环境示例不属于本 FYP。
本实现使用普通 PyTorch 训练循环与 Hugging Face AutoTokenizer/AutoModelForSequenceClassification，便于审计 BCE、累积梯度和评估边界；不依赖 Trainer 默认行为。
为保证绝对路径，Conda 用 `create -p`，与要求的最小 `create -n` 环境方式等价。
smoke 按 Job ID 放在独立目录以避免覆盖；正式结果仍使用要求的 seed42 目录。
首次 setup 也提供 Slurm 脚本，确保 CUDA 验证发生在 GPU 节点。

参考：

- https://pytorch.org/get-started/previous-versions/#v260
- https://huggingface.co/docs/transformers/v4.51.3/model_doc/bert
- https://huggingface.co/datasets/google-research-datasets/go_emotions
- https://github.com/google-research/google-research/tree/master/goemotions
