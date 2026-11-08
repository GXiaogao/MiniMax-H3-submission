# H3 条件分支诊断：第一步构建

上一轮保存的采样点中，视频与音频分支一致，条件分支 TP8 相对 L2 约为 1.0754。此次为 H3 条件分支增加采样点，继续定位误差；尚未修改数值实现。

## 上传与启动

在 Windows 本地解压 `minimax-h3-condition-build-upload.zip`，把以下五个文件上传到平台“我的项目”的同一目录：

- `build_minimax_h3_condition_trace.sh`
- `prepare_minimax_h3_condition_trace.py`
- `prepare_minimax_h3_boundary_trace.py`
- `h3_boundary_trace.hpp`
- `build_infinilm_extension.sh`

这几个小文件属于项目启动输入；模型、wheel 和仓库继续使用个人 AFS 中已有内容。已有同名依赖可保留，但必须与本压缩包校验一致，不要换用其他版本的原始构建脚本。

设置：

- 入口：**`build_minimax_h3_condition_trace.sh`**，不要选择 Python 文件。
- 方式：批处理作业；环境沿用已验证的 `scnet-dcu`，八卡整节点。
- 程序参数：**留空**。
- 时长：**28 分钟**；脚本内部构建超时为 25 分钟。

本次只构建诊断扩展，不加载正式模型生成视频，也不要求重新上传 `first-inputs.npz`。构建耗时及能否在配额内完成以服务器日志为准。

## 产物与结果回传

成功时末尾出现 `ISOLATED_CONDITION_BUILD COMPLETE` 且退出码为 0。请返回：

1. 本次完整下载日志。
2. 个人 AFS `minimax-h3/state/infinilm-condition.success` 文件。

若失败，返回日志中首次失败阶段及完整下载日志，不要反复重跑。

扩展独立放在 `$WORK_HOME/diagnostics/h3-condition/<本次编号>/InfiniLM/`；正式扩展和上一轮 boundary 扩展保留。日志位于 `$WORK_HOME/logs/infinifabric-prepare-h3-condition-*` 与 `infinifabric-build-h3-condition-*`。

## 后续对比范围

新扩展共保留 24 个边界：原有 9 个，以及条件输入、投影、两个 refiner 子块各 6 个边界、最终归一化。两个子块分别记录 norm1、Attention 输出、Attention 残差输出、norm2、MLP 输出、MLP 残差输出。采样仅在显式启用时执行，rank0 每个边界最多 65536 个值。

获得新扩展的哈希后，再运行复用现有输入的 TP1/TP8/参考对比。构建成功只证明编译、链接与导入检查完成，不代表八卡数值正确或视频质量恢复。
