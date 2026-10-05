# 下载与安装中断恢复

用户明确要求中断后避免全部重新来。本工程区分已完成文件缓存、同进程网络重试、跨进程字节续传、安装事务恢复。

## 当前HPC环境已经完成

用户输出确认Python3.10.21、pip26.2.1、pip check通过。不要重建此环境，也不要降级pip到旧脚本的25.0.1。
尚未收到实际CUDA/PyG验证或模型训练证据。

## 当前下一步：PyTorch主wheel

`environment/download_torch.sh`使用官方索引链接和索引给出的SHA256，下载到持久的 `downloads/torch/*.part`。
同一命令重跑时 `curl --continue-at -`从磁盘文件长度继续；成功SHA256校验后才改名为.whl。已完成文件重新校验后跳过网络。
若中断发生在全部字节已到达、改名前，重跑先检查.part的hash并直接完成改名。
任何下载失败或校验失败都不会开始安装。不要手工删除.part、下载目录或Conda环境。
如果上游不支持HTTP Range，curl会报错；届时检查后仅重下该文件，不能承诺所有服务器都允许续传。

## 通用大文件工具

`scripts/resume_download.py`提供持久.part、SHA256资产身份、互斥锁、Range响应检查、重试和验证后原子改名。
五项本地测试（模拟HTTP传输）通过：中断后第二次调用续传、完成文件不重下、无Range服务仅重下本文件、错误Range拒绝追加、hash/资产版本不符拒绝接受。
这些是本地逻辑测试，不是HPC真实网络验收。当前给用户的PyTorch命令直接用curl，以便无需先上传通用脚本。

## 后续pip/Hugging Face

`environment/download_settings.sh`提供持久pip/HF cache与超时/重试配置。已安装pip26.2.1可用resume retries，无需升级/降级。
[pip官方说明](https://pip.pypa.io/en/stable/news/#v25-2)自25.2默认开启下载重试/恢复；这不等于承诺pip进程被杀后所有临时wheel都能跨进程续传。
主wheel采用上述持久.part方式；后续大型CUDA依赖在安装前也应采用持久下载/校验方案，不直接用pip临时下载方式冒充跨进程续传。已下载完整依赖保留本地wheel缓存。
模型使用固定revision与持久HF cache，官方支持的恢复行为需结合实际hub版本核验；模型下载步骤尚未执行。

## 安装不是字节下载

Conda/pip安装中断后先检查已有环境、包版本和pip check，再重试缺失阶段；不自动删除整个环境。Conda已有完整包通常可复用缓存，但不能保证旧版Conda对每个半下载包做跨进程字节续传。
当前最小环境已经成功，恢复问题主要在接下来的大型wheel/model/data。下载完成不等于依赖安装完成，安装完成不等于GPU验证完成。

所有这些新脚本目前保存在本地项目；HPC仅执行用户在对话中实际复制的命令，不假定本地文件已自动同步。

## 一次性安装全部依赖

`environment/install_all_cpu.py` + `install_all_terminal.sh` 合并后可一次复制执行。
先校验已有Torch主wheel；使用固定uv0.8.22只解析依赖版本（uv自身wheel也续传并校验），按目标Python/平台从PyPI元数据选择兼容wheel，然后全部持久下载/校验，最后pip离线安装。
未采用pip dry-run作为大包预解析步骤：其内部可能下载完整wheel到临时目录，会破坏预期的持久续传流程。
首次版本解析成功后保存dependency-versions.txt和fingerprint，重跑复用；完成wheel校验后生成包含每包精确版本/SHA256的requirements-resolved-hashes.txt。
新增显式固定huggingface-hub0.26.5、swanlab0.6.12，以及后续诊断作图所需matplotlib3.9.2。
安装后pip check、全模块import、CPU GAT有限前后向、官方train.py --help、frozen git清洁检查全部通过才写LGSA_ENV_CPU_READY；不标记GPU smoke通过。
本地验证范围：Python/Shell语法、三项模拟下载恢复与hash拒绝测试。尚未在HPC完成依赖解析/安装，等待用户实际执行结果。
