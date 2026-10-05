# 运行时生成的环境记录

`pip-freeze.txt`、`python-version.txt`、`torch-info.txt` 由 HPC 首次安装脚本生成。
不放置伪造的运行记录。每次 smoke/full 作业另存自己的版本和 GPU 信息。
本地交付检查写入 `local_validation.json`，不能替代 HPC GPU 验证。
