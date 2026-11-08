# 八卡权重分片复制检测

本包只有两个执行文件，不装载模型、不安装依赖、不重建扩展，不修改正式推理。用于核查实际 InfiniCore 跨卡非连续分片复制的临时缓冲区和源设备时序。

1. 解压 `minimax-h3-shard-copy-probe-upload.zip`，把 `probe_minimax_h3_shard_copy.py` 和 `.sh` 上传到平台项目同一目录。
2. 入口选 `probe_minimax_h3_shard_copy.sh`，参数留空，scnet-dcu 八卡，时长 28 分钟。内部超时 25 分钟。
3. 在个人存储 `minimax-h3/logs/infinifabric-h3-shard-copy-<本次ID>/` 下载 `shard-copy.json` 和 `run.log`。

需保留刚才已完成的目录 `logs/infinifabric-h3-projection-20261107T015112Z-2822929/`，尤其 `projection-parity.json`、`native-tp1/native-prefix-model.json`、`native-tp8/native-prefix-model.json`。检测会核对原报告哈希并记录实际 staging 元数据。仍使用现有 `$WORK_HOME/.infini` 库及 InfiniCore Python 目录，不需要重新下载 wheel、模型或 FlashAttention。

每次在 GPU0 创建合成源矩阵，使用原生 InfiniCore 窄化和复制到八卡。分别测试：

- `implicit`：直接复制非连续分片，由现有库内部处理连续化。
- `retained`：调用相同的原生连续化操作，显式保留其结果到全部复制完成。
- `retained_source_sync`：在上一控制基础上，复制前等待源设备流完成。

覆盖 FP32/BF16、小矩阵与 `[5376,7168]` 实际权重形状，每种模式重复三组不同输入。所有卡提交复制后才同步和读取观察结果；显式缓冲区在同步前始终保留。Torch 只用来产生输入和读取输出，不替代被测试的 InfiniCore 运算。精确整数在这两种精度可表示，因此复制结果按完整数组逐值比较。

`PROBE_COMPLETE` 仅表示所有控制已执行并记录，须查看 `mode_summary` 和每卡 `mismatched_elements`。隐式路径可能在本轮无法复现；控制路径通过也不单独证明模型已经恢复。不把该探测耗时作为性能成绩。
