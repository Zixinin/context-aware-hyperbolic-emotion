# Exp6D-1 结果验收

最终SUCCESS仅在nrun.verify通过后生成；不能凭本Markdown存在认定成功。校验器逐项检查：源manifest、config、4轮7648步/765warmup、最后LR0、批日志与每轮样本权重、完整训练顺序、checkpoint SHA、Validation数组与所有27类metrics/AP、重载、无更新几何探针以及冻结保护证明。
初始化来自相同raw snapshot；shared/head/HNN/decoder初值哈希与A1-S一致，安装N1前后train/eval logits、z、decoder输出及RNG消耗一致。
选择证据：{'epoch': 4, 'global_step': 7648, 'validation_macro_f1': 0.5779985784591636, 'threshold': 0.3, 'selection_split': 'validation', 'tie_break': 'earliest strict greater-than', 'checkpoint_sha256': '67f5c0ad1be7220db6e335255eaf83d2e84ce2fa9c9eb67da3e38afbdb5f8d7d'}
完整3834条Validation重载：{'num_examples': 3834, 'ids_equal': True, 'targets_equal': True, 'threshold_predictions_equal': True, 'metrics_equal': True, 'max_abs_logit_error': 0.0, 'atol': 1e-06, 'rtol': 1e-06, 'logits_exact_equal': True}
冻结文件复核：{'all_historical_files_rehashed_unchanged': True, 'protected_file_count': 4858, 'protected_manifest_sha256': '53b2289f4f736d79770df099c82fbbb9361c06a8fc08d371a29e21e96e4218f8', 'completed': '2026-10-23T04:37:33.745315+00:00'}
新Test推理=0；无Test预测文件；没有rawF训练、额外seed或参数搜索。训练与smoke退出码由shell记录，完整SUCCESS封存后还会由CPU只读校验器复查。
