# H3 Transformer 单次前向对照

本阶段定位 Native 生成彩色网格的问题，不生成成片，不报告性能或官方正确性通过。正式路径继续使用 Native。参考 Diffusers H3 只在独立进程中对同一输入进行诊断计算。

## 上传与启动

解压 `minimax-h3-transformer-parity-upload.zip`，将其中六个脚本放入同一个平台项目的根目录：

1. `diagnose_minimax_h3_transformer_parity.sh`（新增入口）
2. `diagnose_minimax_h3_transformer_parity.py`（新增诊断）
3. `prepare_minimax_h3_vae_dtype_probe.py`
4. `run_minimax_h3_fl2va_native_smoke.py`
5. `install_infinilm_private_runtime.py`
6. `prepare_minimax_h3_hygon_runtime.py`

后四个为已用过的辅助脚本；若项目内已有相同版本，无须重复创建。包内附逐文件 SHA256。不要往项目输入区重新上传模型、wheel 或先前生成的视频/NPZ。

- 入口：`diagnose_minimax_h3_transformer_parity.sh`
- 程序参数：留空。
- 使用方式：批处理作业；运行环境：已有 scnet-dcu；整节点八卡。
- 运行时长：28 分钟。脚本内部 25 分钟超时；实际所需时长须由服务器实测，不能保证参考权重加载/计算在预算内完成。
- 依赖沿用 AFS 下已准备的安装状态与共享权重，不执行在线下载或系统 pip 安装。
- 个人首帧仍为 `$WORK_HOME/inputs/fl2va/Golde33443.jpg`。

## 会执行什么

1. 校验已有环境状态与 Native 扩展，准备作业本地依赖和只保留 VAE FP32 的入口副本。
2. 运行原 Native FL2VA 的初始化、Qwen 编码和第一次 H3 Transformer 前向。源配置维持 8 卡、124 帧、10 步参数、seed 42、576×384；保存原输入和 Native 输出后有意结束这个独立组件诊断。不会继续调度或生成视频。
3. 等 Native 子进程退出、释放显存后，在新进程加载相同 H3 权重的 Diffusers 参考实现，重放完全相同的张量。保存/恢复保持索引 int64、位置 float64 等 dtype；BF16 以可精确表示它的 FP32 存储，再恢复 BF16。
4. 输出视频/音频预测张量的全元素 MAE、RMSE、最大误差、相对 L2 和余弦相似度。没有预设官方通过阈值。

Qwen 编码在两边保持固定，因此此诊断不验证 Qwen 自身正确性。如果 H3 输出差异明显，下一步定位 H3 内部；若两者接近，还须继续对照 Qwen、调度/布局和解码输入。单次前向接近不能证明整条生成链路正确。

## 取回结果

日志目录：`$WORK_HOME/logs/infinifabric-transformer-parity-<时间>-<进程号>/`

首先提供 `transformer-parity.json` 和 `run.log`。若失败，补充 `native.log` 或 `reference.log` 中存在的文件；若需要逐元素定位，保留 `first-inputs.npz/.json`、`first-native-outputs.npz/.json`、`first-reference-outputs.npz/.json`。

正常结束标签：`PARITY_DIAGNOSTIC COMPLETE`，JSON 为 `COMPARISON_COMPLETE`。这只表示取得了对照数据，`quality_status` 仍为 `UNASSESSED`。本阶段没有新 MP4，属于预期行为。
