# 第一个 token refiner Attention：每卡采样对比

已核对本次完整日志与 `infinilm-attention.success`：构建、链接、扩展导入检查和发布完成。版本 `1.0-isolated-h3-attention-trace`，构建编号 `20261106T083515Z-1660705`，扩展SHA256：

```
f1726e0267f6a1c08cfb8b2e39abe4739022717ac75278dbbd048aa9c9cb0e4b
```

日志中的 `cp: preserving permissions ... Bad file descriptor` 是复制权限元数据警告。导入检查已通过；本轮仍逐项校验诊断 Python 副本，不据此直接假定所有复制内容正确。

## 上传与启动

本地解压 `minimax-h3-attention-capture-upload.zip`，将八个运行文件上传到平台同一项目目录：

- `diagnose_minimax_h3_attention.sh`
- `diagnose_minimax_h3_attention.py`
- `diagnose_minimax_h3_prefix_parity.py`
- `diagnose_minimax_h3_transformer_parity.py`
- `run_minimax_h3_fl2va_native_smoke.py`
- `prepare_minimax_h3_vae_dtype_probe.py`
- `install_infinilm_private_runtime.py`
- `prepare_minimax_h3_hygon_runtime.py`

已有同名依赖请核对本包校验清单，使用本包版本。无需重新编译、下载模型或上传原始输入。

启动设置：

- 入口：**`diagnose_minimax_h3_attention.sh`**。
- 方式：批处理作业；沿用已验证的 `scnet-dcu` 八卡整节点环境。
- 程序参数：**留空**。
- 时长：**28 分钟**；内部诊断总超时25分钟。

## 需要保留的现有文件

脚本自动读取个人AFS中：

```
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.npz
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.json
minimax-h3/state/infinilm-attention.success
minimax-h3/logs/infinifabric-build-h3-attention-20261106T083516Z-1660775/
minimax-h3/diagnostics/h3-attention/20261106T083515Z-1660705/
```

保留上述目录，尤其构建校验清单、采样源码及元数据。旧 condition 结果也建议保留以便对照。脚本核对官方扩展、模型配置、新扩展、构建身份与原始输入哈希后，依次执行 Native单卡、Native八卡和参考模型进程。

## 回传结果

日志目录：`$WORK_HOME/logs/infinifabric-h3-attention-<本次编号>/`。

完成后返回：

1. `attention-parity.json`
2. `run.log`，或网页下载的完整运行日志。

如果失败，额外返回失败阶段对应的 `native-tp1/worker.log`、`native-tp8/worker.log` 或 `reference/worker.log`。预检失败只需返回报告和主日志，先不要重复重跑。

正常完成显示 `ATTENTION_DIAGNOSTIC COMPLETE`，报告 `COMPARISON_COMPLETE`，退出码0。它表示采样对比完成，不表示数值误差合格。此次不生成视频。

## 对比口径

保留24个条件/主块采样边界，新增第一个refiner Attention每卡8个边界。参考Q/K/V和MHA先按每个token的56个头切成TP8每卡7头，再展平采样；归一化Q/K保留四维。单卡与八卡分别对齐参考，不能直接切分单卡已有采样点来冒充八卡分片。八卡输入和投影归约后输出另做卡间一致性比较。

所有结果仅针对保存的采样点。输出投影与归约仍作为原组合阶段观察。正式扩展、权重和评测工作量保持原样，本次一个主块配置仅用于诊断，不是正式质量或性能验收。
