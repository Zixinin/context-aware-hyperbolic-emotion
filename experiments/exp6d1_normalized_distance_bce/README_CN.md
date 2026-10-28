# Exp6D-1：Normalized Hyperbolic Distance-weighted BCE pilot

本包只授权一个N1 seed42独立训练：smoke通过后自动4epochs，完整Validation选轮/重载，不做新Test，不自动多seed或换公式。用户手动在Ubuntu执行，Codex不连接远端。

匹配控制A1-S的run_20261022T035554Z_cZrAxW97；先复核已完成Exp6D run_20261022T063911Z_Ktwq95al。源码/封存结果/数据/snapshot必须保持原样。新增代码在experiments/exp6d1_normalized_distance_bce，所有新结果在results/exp6d1_normalized_distance_bce。

入口：`bash launch.sh`。本包smoke90分钟上限、正式训练8小时上限；非有限、映射错误、参考N1不一致、异常梯度或冻结保护失败会停止，不自动重试。smoke旧checkpoint仅用于no-update对照，正式训练从同raw snapshot和seed42重新开始。

已包含开始、smoke通过、成功和失败ntfy通知；通知网络失败打印提示并保留实验真实退出码。结束在终端直接打印中文总结、四模型Validation比较、七Rare和几何轨迹，不要求下载审核包。

成功后CPU只读检查：`bash launch.sh verify /home/zixinfeng/Documents/context-aware-hyperbolic-emotion/results/exp6d1_normalized_distance_bce/run_<actual>`。查看：`cat <run>/EXP6D1_TRAINING_SUMMARY.md`。旧结果哈希再检查可用新run的protected_before.json逐文件复核，不能调用旧分支拒绝新目录的全局inventory规则。

`LOCAL_QA.json`仅记录Mac CPU合成/保存缓存验收，不是Ubuntu GPU smoke或训练证据。真实结果以smoke/SMOKE_VERIFIED.json、train_exit_code.txt=0、SUCCESS.json、完整3834条exact reload与FROZEN_POSTCHECK.json为准。

生产公式与限制、参考容差、梯度路径、实际最后11条batch、N1与eval batch64的关系详见IMPLEMENTATION_AUDIT.md。
