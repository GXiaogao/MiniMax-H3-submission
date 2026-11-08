# H3 首块 Native 单卡与八卡对照

本轮只新增一次 **1 个 H3 主干块、TP=1 的 Native 计算**，与上次保存的 TP=8 Native 和参考输出分别比较。输入投影、token refiner、时间嵌入及输出头继续保留；不是正式完整模型或比赛工作量，不能用作速度/质量验收。正式八卡 Native 路径和磁盘权重/config.json 不变。

## 上传和启动

解压 `minimax-h3-tp-parity-upload.zip`，将七个脚本上传到同一平台项目根目录：

1. `diagnose_minimax_h3_tp_parity.sh`（新入口）
2. `diagnose_minimax_h3_prefix_parity.py`（更新至 1.2）
3. `diagnose_minimax_h3_transformer_parity.py`
4. `run_minimax_h3_fl2va_native_smoke.py`
5. `prepare_minimax_h3_vae_dtype_probe.py`
6. `install_infinilm_private_runtime.py`
7. `prepare_minimax_h3_hygon_runtime.py`

前两项为此次新增或修改，其余已存在的相同版本可复用。不要把模型、历史输出 NPZ 或安装包上传到项目输入区。

- 入口：`diagnose_minimax_h3_tp_parity.sh`
- 程序参数：留空。
- 使用方式：批处理作业；现有 scnet-dcu 八卡节点，申请方式沿用上次。
- 时长：28 分钟，脚本内部 25 分钟超时。实际时间以服务器本轮执行为准。

平台仍分配八卡，但本次诊断模型使用 cuda:0；无需寻找单卡申请选项。不会重新运行 Qwen、参考模型或完整视频生成，不需要重新编译。

## 必须保留的 AFS 文件

原始输入目录：

```text
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/
```

保留 `first-inputs.json`、`first-inputs.npz`。

上次 0/1 层成功对照目录：

```text
minimax-h3/logs/infinifabric-prefix-parity-20261105T132107Z-693509/
```

保留根目录 `prefix-parity.json`，以及 `prefix-1/` 中：

- `first-inputs.json`、`first-inputs.npz`
- `first-native-outputs.json`、`first-native-outputs.npz`
- `first-reference-outputs.json`、`first-reference-outputs.npz`

脚本在环境准备前核对原始输入、已完成的四个 worker、基线输入与扩展哈希，并验证每份 NPZ 的字节哈希、形状、dtype 和有限性；缺少文件或数据损坏会明确失败。报告记录复用来源和文件校验值。

## 返回结果与解释

从 `minimax-h3/logs/infinifabric-tp-parity-<时间>-<进程号>/` 下载 `tp-parity.json` 和 `run.log`。失败时补充 `tp-1/native.log`，保留该目录的模型元数据与输出 NPZ。

成功采集标签：`TP_DIAGNOSTIC COMPLETE`，结果 `COMPARISON_COMPLETE`。三组指标为：

- `tp1_vs_reference`：单卡相对参考的误差。
- `tp8_vs_reference`：复算上次八卡相对参考的误差。
- `tp1_vs_tp8`：单卡相对上次八卡的误差，以八卡输出范数为相对 L2 的分母。

没有自动通过阈值。单卡显著更接近参考时，优先检查分片与通信；单卡与八卡都偏离参考时，优先检查共用首块计算。历史八卡与本轮单卡不在同一次进程执行，这些仍是定位线索，不能直接证明某项通信或算子错误；必要时须做同环境重复性与内部边界验证。本轮没有新 MP4，性能和官方相似度仍未验证。
