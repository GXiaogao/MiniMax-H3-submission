# 实际模型中间值诊断：第一步构建

独立 AdaLN 微测试未复现大偏差，接下来需要实际 Native 模型的中间值。受每次作业最多 28 分钟限制，本阶段分为两次作业：先构建带可选插桩的独立扩展；确认构建成功后，再运行 TP=1/TP=8 与参考的中间值采集。当前入口只构建并做扩展导入检查，不生成视频或执行中间值比较。

## 上传和启动

解压 `minimax-h3-boundary-build-upload.zip`，把四个文件放在同一平台项目根目录：

1. `build_minimax_h3_boundary_trace.sh`（入口）
2. `prepare_minimax_h3_boundary_trace.py`
3. `h3_boundary_trace.hpp`
4. `build_infinilm_extension.sh`（复用的基准构建脚本，按包内版本核对）

入口选择 `build_minimax_h3_boundary_trace.sh`，程序参数留空，使用现有 scnet-dcu 八卡批处理，时长 **28 分钟**。内部构建限时 25 分钟，实际编译是否能在预算内结束需本次实测。

沿用 AFS 已准备的 `setup_infinifabric_env.sh`、源码归档 `packages/InfiniLM-infinifabric-v3.tar.gz`、`packages/pybind11-v3.1.0.zip`、`packages/xmake-repo-cache.tar.gz` 和已安装 InfiniCore。无需重新上传模型或 wheel 到项目输入区，也不修改系统解释器。缺失文件会在日志中明确指出。

## 构建隔离

- 原扩展与 `state/infinilm-extension.success` 保留。
- 新扩展及 Python 副本保存到 `minimax-h3/diagnostics/h3-boundary/<本次构建编号>/InfiniLM/`。
- 独立标记为 `minimax-h3/state/infinilm-boundary.success`，其中 `extension=` 指明新扩展路径。
- 每次构建使用不同目录，不覆盖旧构建扩展；构建失败时不发布成功标记。
- 现有构建脚本本体不修改，补丁只应用到校验后的作业本地源码副本。日志保留插桩后的 C++、辅助头文件及源码哈希记录。

插桩默认关闭。后续独立诊断会在单个 H3 主干块、关闭图编译的条件下启用它，记录 packed hidden、时间嵌入、AdaLN 投影、Attention 输入/输出和残差后、MLP 输入/输出和残差后共 9 个边界。原计算结果继续返回。当前不要手工设置 `H3_BOUNDARY_TRACE_DIR` 或用它运行完整 50 层模型。

每个边界只保存 rank 0 至多 65,536 个按确定性整数索引采样的值，BF16 可精确转存为 FP32；另存形状、dtype、样本数和采样公式。采样不覆盖全部元素，不能用它宣称全部张量正确或有限；插桩包含同步及文件写入，不用于性能测量。

## 返回文件

首先提供本次运行页面的完整日志。日志会列出两个目录：

- 外层 `minimax-h3/logs/infinifabric-prepare-h3-boundary-<编号>/`
- 实际构建 `minimax-h3/logs/infinifabric-build-h3-boundary-<编号>/`

成功后，再从个人存储下载 `minimax-h3/state/infinilm-boundary.success`。后续根据该标记加载独立扩展，再执行中间值采集。

成功标签 `ISOLATED_BOUNDARY_BUILD COMPLETE` 仅代表诊断扩展已构建并导入，不能说明视频质量或模型数值已通过。若失败，提供实际构建目录中对应阶段日志，尤其 `xmake-build.log` 或 `h3-boundary-trace-patch.log`。
