# AdaLN 分片投影与 allgather 独立测试

本轮针对首块 TP=8 比 TP=1 偏差更大的现象，检查 AdaLN 的投影、按输出列分片和聚合排列。使用原始输入中的时间步和原始 checkpoint 首块权重。正式 Native 模型、权重和配置文件不变。本轮不是视频生成、完整模型相似度或性能测量。

## 上传启动

解压 `minimax-h3-adaln-probe-upload.zip`，将以下六个文件放入同一个平台项目根目录：

1. `diagnose_minimax_h3_adaln_probe.sh`（新入口）
2. `diagnose_minimax_h3_adaln_probe.py`
3. `adaln_probe_shim.cpp`
4. `diagnose_minimax_h3_transformer_parity.py`
5. `run_minimax_h3_fl2va_native_smoke.py`
6. `install_infinilm_private_runtime.py`

前三个为本轮新增；其余相同版本可复用。入口选择 `diagnose_minimax_h3_adaln_probe.sh`，参数留空，现有 scnet-dcu 八卡批处理，28 分钟。内部超时 25 分钟。不需要上传模型或 wheel 到项目输入区。

保留个人 AFS 原输入目录：

```text
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/
```

需要其中的 `first-inputs.json`、`first-inputs.npz`。共享模型路径仍为 `/public/home/aclsylqidf/shared/MiniMax-H3`。

## 编译与测试内容

使用现有 g++/c++，只编译一个小型 C++ 共享库；不重建 InfiniLM/InfiniCore，不安装系统依赖。编译包含现有 InfiniCore 头文件，并链接个人 `.infini/lib` 的现有四个库。缺少编译器、头文件或符号会停止并保留 `compile.log`。

1. 读取六个所需 checkpoint 张量，以参考计算重建时间嵌入与激活，转为 BF16，作为两边共同投影输入。
2. 用八卡卡号／时间行标记检查 allgather 的聚合顺序，标记数值可由 BF16 精确表示。
3. 运行单卡完整 AdaLN 投影。
4. 运行八卡各自分片投影，再执行与归档模型相同的 transpose→allgather→transpose。记录每卡局部结果及聚合后误差。

核心测试计算调用现有 InfiniCore linear 与 allgather；PyTorch 仅产生诊断参考输入/输出和承载 GPU 缓冲区。手动切片重放不经过实际模型的 Parameter 加载器，也不记录真实模型的内部中间值；若结果接近，只能说明此独立投影/聚合测试接近，不能排除加载、原生 SiLU、图执行或其他模块问题。若排列正确但局部投影已有误差，应先检查 GEMM/精度；若局部正确但聚合不同，再检查聚合路径。两者都接近时，继续实际模型的加载/中间边界检查，不直接更改通信实现。

## 下载结果

目录：`minimax-h3/logs/infinifabric-adaln-probe-<时间>-<进程号>/`。

返回 `adaln-probe.json` 和 `run.log`。编译失败时补充 `compile.log`，其余失败补充错误前的完整日志。保留 `adaln-inputs.npz/.json`、`adaln-reference.npz/.json` 及 `projection-tp1`、`projection-tp8` 的输出文件。

标签 `ADALN_COMPONENT COMPLETE` 和 `COMPARISON_COMPLETE` 仅表示比较已采集；模型质量仍为 UNASSESSED，不设置官方阈值。没有新 MP4。本地未具备海光编译/运行环境，C++ 的实际编译和运行待本轮服务器验证。

已知限制：Torch ABI=1 预检不能单独证明已有 InfiniCore 的 ABI；若不兼容，以编译/装载日志为准。某个 rank 在 collective 前异常，其他 rank 可能等待到外层超时，失败时不保证逐个销毁 communicator；这是一次性诊断进程，日志和失败状态仍保留，不会把部分运行记为比较完成。
