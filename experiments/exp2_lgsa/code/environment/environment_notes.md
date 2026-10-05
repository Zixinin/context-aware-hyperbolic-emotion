# 独立候选环境：尚未在 HPC 安装或验证

固定独立路径 `/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo`。绝不向 `fypemotion` 安装依赖。

| 项 | 候选 | 证据/待验证 |
|---|---|---|
| Python | 3.10 | 新环境版本，不复用Mac Python |
| Torch | 2.5.1 CUDA11.8 wheel | [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/#v251)列有cu118组合；是兼容性选择，不是README的2.12 |
| CUDA module | cuda/11.8.0-gcc-8.5.0-d7ndetl | 用户给出的学校模块，仍需当前HPC核对 |
| PyG | 2.6.1 | 与LGSA README一致；[PyG安装文档](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)说明基础功能可使用PyTorch原生支持；真正GAT CUDA backward待运行 |
| Transformers | 4.44.2 | 保留BertOnlyMLMHead等旧接口的候选；未宣称官方指定版本 |
| Datasets | 2.21.0 | 候选，真实JSON loading/cache待验证 |
| GPU | RTX4090（目标） | 尚无本实验分配/GPU日志 |

README 的 `torch==2.12` 按原文记录，不能擅自解释成2.1.2。PyTorch2.5.1还避免了旧代码在较新版本默认 `torch.load(weights_only=True)` 下读取普通Python checkpoint对象的问题；仅加载可信且经哈希核验的资产。

`requirements-candidate.txt` 是待验证顶层依赖，不是真正 lock。swanlab 顶层 import 强制存在，即使不开日志也需安装；暂限制兼容范围，安装后记录实际版本。
旧 `setup_lgsa_goemo.sh` 将安装与GPU验证绑定，仅能在Linux Slurm GPU分配内运行。**用户已选择终端分阶段安装，因此不再提交旧的环境创建/安装GPU作业。** 当前先通过终端创建最小环境、安装预编译包并做CPU导入/依赖检查；届时保存安装快照，不将它称为GPU验证通过。实际CUDA/PyG backward检查移到实验Slurm作业开头，失败即停。

如果需要额外PyG扩展，按实际错误针对所选torch/cu118安装对应wheel，不在失败前把torch-scatter/torch-sparse全部盲装或源码编译。

当前本地仅用已有Mac torch2.6.0运行小tensor审计；这与候选HPC环境验证无关。
HPC源码clone、CUDA模块/nvcc和分区已由用户输出核验，2984874提交成功但仍排队；环境、实际GPU验证尚未完成。
