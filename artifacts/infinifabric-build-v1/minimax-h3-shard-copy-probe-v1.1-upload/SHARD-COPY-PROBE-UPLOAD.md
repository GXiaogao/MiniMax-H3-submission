# 八卡权重分片复制检测

2026-11-07 v1.1：此前探测在结果读取的原生 `copy_` 调用段错误，具体原生原因尚未确定。本版去除读取阶段额外的 Core 同卡复制，改用已有 InfiniRT 同步 D2H；工作线程保留到同步和读取结束。混合模式及三种单独模式分四个子进程，分别保存检查点与退出状态，单组崩溃后继续其他组。更新两个同名执行文件，入口不变。

本包只有两个执行文件，不装载模型、不安装依赖、不重建扩展，不修改正式推理。用于核查实际 InfiniCore 跨卡非连续分片复制的临时缓冲区和源设备时序。

1. 解压 `minimax-h3-shard-copy-probe-v1.1-upload.zip`，把 `probe_minimax_h3_shard_copy.py` 和 `.sh` 更新到平台项目同一目录。
2. 入口选 `probe_minimax_h3_shard_copy.sh`，参数留空，scnet-dcu 八卡，时长 28 分钟。内部超时 25 分钟。
3. 在个人存储 `minimax-h3/logs/infinifabric-h3-shard-copy-<本次ID>/` 下载 `shard-copy.json` 和 `run.log`。

需保留刚才已完成的目录 `logs/infinifabric-h3-projection-20261107T015112Z-2822929/`，尤其 `projection-parity.json`、`native-tp1/native-prefix-model.json`、`native-tp8/native-prefix-model.json`。检测会核对原报告哈希并记录实际 staging 元数据。仍使用现有 `$WORK_HOME/.infini` 库及 InfiniCore Python 目录，不需要重新下载 wheel、模型或 FlashAttention。

每次在 GPU0 创建合成源矩阵，使用原生 InfiniCore 窄化和复制到八卡。分别测试：

- `implicit`：直接复制非连续分片，由现有库内部处理连续化。
- `retained`：调用相同的原生连续化操作，显式保留其结果到全部复制完成。
- `retained_source_sync`：在上一控制基础上，复制前等待源设备流完成。

覆盖 FP32/BF16、小矩阵与 `[5376,7168]` 实际权重形状，每种模式重复三组不同输入。所有卡提交复制后才同步和读取观察结果；显式缓冲区及线程保持到读取结束。Torch 只用来产生输入；InfiniRT 同步 D2H 读取已完成的结果，不替代被测试的 InfiniCore 运算。精确整数在这两种精度可表示，因此复制结果按完整数组逐值比较。

父进程报告的 `groups` 分别为 `mixed`、`implicit`、`retained`、`retained_source_sync`。mixed 组覆盖 36 个测试，另外三组各 12 个测试，每个测试均含八卡。每组最长五分钟，总外层超时 25 分钟；每 30 秒打印心跳。每次读回前保存模式、形状、精度、epoch 和卡号，即使原生崩溃也能保留最近检查点。脚本禁止生成大体积 core dump 文件。

`PROBE_COMPLETE` 仅表示所有组和规定测试均完整执行。`INCOMPLETE_CHILD_FAILURE` 表示至少一组崩溃、超时或缺少结果，此时父进程退出码仍非零，但会保存 `shard-copy.json`。请照常下载它和 `run.log`；需要进一步定位时再取相应子目录的 `worker.log` 和 `shard-copy.json`。

数值结果查看 `groups.<组名>.checkpoint.mode_summary` 及各测试每卡 `mismatched_elements`。隐式路径可能在本轮无法复现；改变观察和生命周期也可能改变故障表现，控制通过不能单独证明根因或模型已恢复。不把该探测耗时作为性能成绩。
