# Exp6D-2 — Paired Multi-seed Verification

用户授权补充六次独立训练。预定矩阵：A0 / A1-S / N1 × seeds123/2026；seed42分别复用三个冻结已完成run，不重训，不新增seed。每个新run从同raw RoBERTa snapshot独立初始化，绝不从旧best或smoke继续训练。只运行Validation，不做新Test、Raw F、欧氏对照、公式/缩放/gamma/阈值搜索。

主控制A1-S用于隔离N1样本权重；A0用于判断整体方案是否超过简单BCE。A0只用原BCE，A1-S只用原BCE+.1aux及s=.03，N1完全导入Exp6D-1已封存生产实现；均不更换分类头、标签向量、层级或优化器。

协议固定27类、30587/3834/3821、maxlen50、batch16/eval64、4epochs/7648step/765warmup、AdamW5e-5、WD.01且bias/LayerNorm免WD、clip1、FP32/eager、sigmoid>=.30、Validation Macro-F1 strict greater-than选轮（平局最早）。每个模型完整3834条Validation重载logits/IDs/gold/阈值预测与27类指标/AP重新核对。

唯一新的控制变量是预定seed：旧seed函数和DataLoader内部均固定42，旧源码不修改。新适配器seed_all(seed)和make_loader(...seed)统一使用对应种子。模型初始化随机数与专用训练DataLoader generator分离；generator跨四轮自然推进，不每epoch重置。对seed42，CPU重放冻结A0 DataLoader的四轮顺序；A0仅有已封存的四个首batch可直接核对，A1-S/N1有四轮完整历史顺序可核对。A0全程顺序未保存，不将代码重建冒充历史全程记录，不给A0补写文件。seed123/2026每轮完整ID序列在任何训练前写入planned_orders。

每个seed的smoke和正式训练均在设置对应PYTHONHASHSEED的新Python进程中执行。旧runtime也限定hashseed42，因此新目录单独复制其平台/包版本/GPU/环境检查，只把允许的hashseed扩展到42/123/2026；不临时伪装环境变量，也不修改旧helper。每个smoke/正式进程仍与旧A0及本轮plan核对同host/Python/packages/pip_freeze/GPU/CUDA。同一seed三模型shared encoder/classifier参数哈希、初始train/eval分类logits与optimizer分组一致；A1-S/N1的HNN/decoder初值及z/decoder初始输出一致。额外分支初始化fork RNG不改变shared初始状态。因A0没有辅助dropout，不能声称完整训练中的随机mask轨迹都相同；配对仍指预定相同seed和数据顺序。

全部六个smoke（每个模型/seed两次丢弃的optimizer更新及64条Validation exact reload）通过后，串行完成seed123 A0→A1-S→N1，再seed2026 A0→A1-S→N1。每次正式训练重新构造fresh模型。沿用相应冻结分支的train-only no-update/RNG隔离探针。A0无双曲空间，几何为not applicable；训练日志中的A0 max_ball_norm=0为不适用占位，不是球半径观察。

开始、每个单元开始/完成、全套成功或失败均ntfy通知。任何smoke、公式、数值、数据、初始化、旧文件保护或重载失败，停止且保留失败文件。只允许跳过已封存并复验成功的单元；不覆盖或自动重试已有不完整单元。不因效果不好换seed。resume只允许继续尚未开始的单元，失败单元须先人工审阅；不在失败checkpoint上续训。

全部旧实验源码/checkpoint/结果不变；修复代码在experiments/exp6d2_paired_multiseed_v1_1，原experiments/exp6d2_paired_multiseed完整保留；新结果全部在results/exp6d2_paired_multiseed。保护清单排除本轮两个新目录及运行缓存，覆盖旧A0/A1/A1-S/Exp6D/Exp6D-1/Exp6A/B等。验证同一snapshot和环境，不安装新依赖或修改CUDA/驱动，不自动git操作。

预先登记的主要比较：N1−A1-S的Validation Macro-F1；第二比较N1−A0的Macro-F1。Micro/Rare/Weighted-F1、Macro AP及逐标签P/R/F1/AP/TP/FP/FN全部报告。统计重复单位为3个训练seed，均值±样本SD(ddof=1)，另报每seed配对差、配对差均值/SD/范围与胜平负数。27个标签与3834条Validation文本不作为独立训练重复数。没有自动显著性检验、p值筛选、事后seed排除或正结果筛选。

每seed对N1−A1-S报告grief的Macro/Rare贡献与其余26类/6个Rare贡献；这是完整指标的代数分解，不删grief、不重新定义主指标。七Rare低支持与AP/固定阈值F1分别解释。三个seed只揭示预定seed下的波动；同一Validation已经多次用于探索，不当作独立最终泛化评估。

交付：九个原始run表、三seed模型均值/SD、配对差表、全27类/七Rare逐seed表、单标签配对差和grief贡献分解、中文组会总结。六个新run各自保存checkpoint、config/environment、四轮日志、训练ID顺序、Validation数组/指标、exact reload和成功封存证明。N1另保存逐训练样本权重/距离与每轮single/multi分布。

本轮完成后停止。欧氏对照是未来单独授权的容量/辅助目标/标签先验/loss尺度匹配实验。最终方法及协议冻结后才考虑正式Test，本轮不执行。

## v1.1修复记录

v1在prepare错误要求A0不存在的完整train_order文件，导致FileNotFoundError。修复只调整证据提取：A0查验封存的四个首batch，原DataLoader只读重建四轮完整顺序；A1-S/N1仍强制核对完整记录。新六个run继续记录并核对全部ID。原失败run_20261023T055040Z_mDXWyluM、原v1源码、旧A0/A1-S/N1的代码、checkpoint、结果均不改写，修复版新增RECOVERY_INPUTS_SHA256.json保护失败证据。六次训练矩阵和公式保持原授权，未开始的训练从新suite独立初始化。
