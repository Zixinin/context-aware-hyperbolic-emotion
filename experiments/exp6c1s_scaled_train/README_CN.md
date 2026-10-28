# Exp6C-1S-Train：s=.03、seed42、Validation-only pilot

唯一模型变化在`scaled_model.py`，其余训练原语直接导入冻结A1。完整约束与原用户prompt见`TRAIN_PROMPT.md`，数学公式/梯度路径、RNG保护、smoke和停止规则见`IMPLEMENTATION_AUDIT.md`。

Ubuntu用户运行`bash experiments/exp6c1s_scaled_train/launch.sh`：先smoke和只读复核，通过后自动开始已授权的四轮seed42。仅完整Validation选择与重载，不运行新Test，不训练其他seed，不搜索尺度或gamma。每次run独立创建；若存在任何正式训练claim，拒绝第二次formal attempt。

完成后只读检查：`bash experiments/exp6c1s_scaled_train/launch.sh verify <精确RUN_ROOT>`。该检查禁用CUDA，不重新训练、重载模型前向或调用Test。完整checkpoint留在Ubuntu，下载审核包明确列出省略的大文件。

此包的本地测试为合成CPU模型/保存数组检查，不冒充真实RoBERTa GPU smoke或训练完成。最终结果以用户运行后生成的Validation数组、几何诊断、训练日志、完整重载证明和SUCCESS为准。
