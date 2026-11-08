# H3 公共前后处理与首个主干块对照

这是独立组件诊断，不生成视频，不代表比赛工作量、正确性或性能通过。MiniMax-H3 正式 Native 配置保持不变；不会改写权重、服务器源码或模型 config.json，也无需重新编译。

## 上传运行

解压 `minimax-h3-prefix-parity-upload.zip`，把七个脚本放入同一项目根目录，已有辅助脚本按包内版本核对：

- `diagnose_minimax_h3_prefix_parity.sh`（此次入口）
- `diagnose_minimax_h3_prefix_parity.py`
- `diagnose_minimax_h3_transformer_parity.py`
- `run_minimax_h3_fl2va_native_smoke.py`
- `prepare_minimax_h3_vae_dtype_probe.py`
- `install_infinilm_private_runtime.py`
- `prepare_minimax_h3_hygon_runtime.py`

选择入口 `diagnose_minimax_h3_prefix_parity.sh`，参数留空，scnet-dcu 八卡批处理，28 分钟。内部 25 分钟超时。不要上传模型或旧 NPZ 到项目输入区。

脚本直接读取个人 AFS 下既有目录：

```text
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/
```

该目录的 `first-inputs.json` 和 `first-inputs.npz` 必须保留。脚本校验约定的 NPZ 哈希、原始模型配置、Native 扩展以及入口源码哈希；漂移会停止并记录原因。无需重新采集 Qwen 输出。

## 本轮范围

按顺序执行 prefix-0 Native、prefix-0 参考、prefix-1 Native、prefix-1 参考；每个进程退出后再运行下一个。

- **prefix-0**：保留输入投影、两个 token-refiner 块、时间嵌入、行布局、最终归一化和输出头；不执行 50 个 H3 主干块。
- **prefix-1**：在相同公共模块之间加入第一个 H3 主干块。

仅诊断对象在内存中的主干块数设为 0/1，使用原权重中对应模块。Native 过滤未实例化模块的权重，参考端只允许明确被排除的主干块键出现在 unexpected_keys，原始报告保存在 `reference-prefix-loading.json`。这不是把裁剪模型用作正式推理，也不替换核心计算。

prefix-0 若已明显不同，应检查公共前后模块；若 prefix-0 接近而 prefix-1 明显不同，则优先定位首块及其交互。两者都接近则继续后续块。这些是定位线索，不是首个错误层的直接证明；输出头仍可能放大中间误差。两者均不能验证 Qwen、完整生成或官方相似度。

## 返回结果

从 `minimax-h3/logs/infinifabric-prefix-parity-<时间>-<进程号>/` 下载：

1. `prefix-parity.json`
2. `run.log`

失败时补充对应 `prefix-0/` 或 `prefix-1/` 中的 `native.log`、`reference.log`；保留两组元数据与输出 NPZ。成功采集标签为 `PREFIX_DIAGNOSTIC COMPLETE`，JSON 为 `COMPARISON_COMPLETE`；这只表示取得比较结果，没有自动通过阈值。本地测试不代替服务器 GPU 实测，实际时长仍需此次运行确认。
