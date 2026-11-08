# 真实权重同步装载对照

刚返回的候选构建及导入已成功。本包使用该扩展执行真实权重对照，不修改正式InfiniLM/Core库、模型权重或正式入口。无需重新构建或重新下载依赖/模型。

1. 解压 `minimax-h3-load-sync-capture-upload.zip`，将其中8个 `.py` / `.sh` 执行文件上传到项目同一目录，更新已有同名辅助文件。
2. 入口选择 `diagnose_minimax_h3_load_sync.sh`，参数留空，scnet-dcu八卡，时长28分钟。内部超时25分钟。
3. 在个人存储 `minimax-h3/logs/infinifabric-h3-load-sync-capture-<本次ID>/` 下载 **`load-sync-parity.json` 和 `run.log`**。若失败，也下载对应 `native-tp1`、`native-tp8` 或 `reference` 子目录的 `worker.log`。

必须保留以下已有材料：

- 候选扩展 `diagnostics/h3-load-sync/20261107T030125Z-2935713/InfiniLM/`、`state/infinilm-load-sync.success`。
- 对应构建证据 `logs/infinifabric-build-h3-load-sync-20261107T030127Z-2935732/`，包含CPP、头文件、trace元数据及构建哈希清单。
- 原始输入 `logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.npz` 及 `.json`。
- 修复前报告 `logs/infinifabric-h3-projection-20261107T015112Z-2822929/projection-parity.json`。
- 正式扩展、私有运行时清单、离线wheel和模型权重。

新扩展固定 SHA256 `ee448772f8e03947f6f33fdaa670fd4079bdd88df31f4e2655ce405aa0b34047`。脚本核对原生装载补丁及辅助头文件、Python副本、共享Core库文件哈希，确认源同步候选是本次主要变更，并将正确扩展哈希传入子进程。

顺序运行候选 Native TP1、Native TP8 和独立参考，复用同一真实输入；每次都重新计算。保留既有24处边界、每卡Attention及投影阶段，比较真实加载的投影权重采样和原始/重算控制结果，再比较此 **单主块前向** 的完整视频与音频输出数组。报告同时列出修复前后的八卡权重采样一致数量及投影相对L2。

`COMPARISON_COMPLETE` 仅表示所有进程、规定边界/卡号和输出材料完整，不等于达到官方质量阈值。权重至多采样65536点；完整输出数组仅覆盖单主块诊断，不是完整50块生成模型。额外投影控制仍会改变调度；参考局部计算也包含上游MHA差异。不把本次总耗时作性能成绩，候选装载期额外传输/内存开销仍需后续评估。
