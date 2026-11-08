# 实际首块中间值采集和比较

独立诊断扩展已经在服务器构建并导入成功：构建编号 `20261106T015551Z-1300344`，扩展 SHA256 `ae7c7b1fab287717d050abde9f59c7f58c463c20c913688ed2dfc7869b8debe4`。本轮直接使用该产物，无需再次编译。

## 上传与启动

解压 `minimax-h3-boundary-capture-upload.zip`，将八个脚本放入同一平台项目根目录：

1. `diagnose_minimax_h3_boundaries.sh`（新入口）
2. `diagnose_minimax_h3_boundaries.py`（新采集器）
3. `diagnose_minimax_h3_prefix_parity.py`
4. `diagnose_minimax_h3_transformer_parity.py`
5. `run_minimax_h3_fl2va_native_smoke.py`
6. `prepare_minimax_h3_vae_dtype_probe.py`
7. `install_infinilm_private_runtime.py`
8. `prepare_minimax_h3_hygon_runtime.py`

前两项为新增，其他已有相同版本可复用。入口选择 **`diagnose_minimax_h3_boundaries.sh`**，参数留空，现有 scnet-dcu 八卡批处理，28 分钟；内部 25 分钟超时。

保持个人 AFS 中的 `state/infinilm-boundary.success`、它指向的 `diagnostics/h3-boundary/20261106T015551Z-1300344/` 整个目录，以及原始输入 `logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.json` 和 `first-inputs.npz`。共享 checkpoint 和已准备的离线包继续复用，不上传至项目输入区。

## 本轮执行

1. 验证输入、模型配置、原扩展、诊断扩展的身份及哈希；逐文件核对诊断 Python 副本，检查构建日志的复制权限警告是否影响文件内容。
2. 在独立进程依次运行 **TP=1 Native、TP=8 Native、参考模型**。每边仅执行一个 H3 主干块，其余公共输入/输出模块保留；三个进程不会同时占用显存。
3. Native 使用诊断目录的扩展，核对实际导入路径和文件哈希，关闭图编译，启用 9 个 rank-0 边界记录；正式扩展继续保留。
4. 参考模型使用同一输入和原始权重，仅对独立诊断对象保留首块，在 cuda:0 运行。通过只读钩子记录对应中间值。
5. 比较 packed hidden、时间嵌入、AdaLN 原始投影、Attention 输入/输出/残差后、MLP 输入/输出/残差后。

每个张量最多比较 65,536 个确定性整数索引样本，BF16 值以 FP32 精确保留。报告记录完整形状、dtype、采样值误差和校验值；有限性检查也只覆盖保存样本。这不等于完整张量逐元素通过。没有预设官方阈值，不自动把最早的非零舍入误差认定为故障。

本轮是中间边界定位诊断，不是完整 50 层模型或比赛工作量，不生成新视频，不验证 Qwen、完整质量或性能。同步/磁盘采集会影响时序，数值结论还需结合输出变化和复现验证。

## 返回文件

从 `minimax-h3/logs/infinifabric-h3-boundaries-<时间>-<进程号>/` 下载：

- `boundary-parity.json`
- `run.log`

失败时，补充 `native-tp1/worker.log`、`native-tp8/worker.log` 或 `reference/worker.log` 中对应的文件。保留三组目录下 `trace/` 中的 `.f32` 和 `.json` 文件，方便继续定位。

正常标签 `BOUNDARY_DIAGNOSTIC COMPLETE` 及 JSON `COMPARISON_COMPLETE` 仅表示三边边界已采集并取得比较结果，质量仍为 UNASSESSED。实际采集效果和运行时长须由本次服务器执行确认。
