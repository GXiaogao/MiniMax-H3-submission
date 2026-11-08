# H3 条件分支诊断：第二步采样对比

已收到 `infinilm-condition.success`，记录构建编号 `20261106T071847Z-74362`、版本 `1.0-isolated-h3-condition-trace`、ABI1。仅凭标记不能独立核验二进制；启动时会读取该构建的 `artifacts.sha256`，校验扩展、保存的采样源码和 Python 副本，再加载诊断扩展。

## 上传与运行

本地解压 `minimax-h3-condition-capture-upload.zip`，将八个运行文件放入平台同一项目目录：

- `diagnose_minimax_h3_conditions.sh`
- `diagnose_minimax_h3_conditions.py`
- `diagnose_minimax_h3_prefix_parity.py`
- `diagnose_minimax_h3_transformer_parity.py`
- `run_minimax_h3_fl2va_native_smoke.py`
- `prepare_minimax_h3_vae_dtype_probe.py`
- `install_infinilm_private_runtime.py`
- `prepare_minimax_h3_hygon_runtime.py`

项目中已有同名依赖时核对本包校验文件，使用本包版本。仅上传这些小文件，已有模型、仓库与 wheel 继续使用个人 AFS。

设置：入口 **`diagnose_minimax_h3_conditions.sh`**；批处理作业；沿用 `scnet-dcu` 八卡整节点；**程序参数留空，时长 28 分钟**。脚本内部总诊断超时 25 分钟。无需重新编译。

自动复用个人 AFS 中：

```
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.npz
minimax-h3/logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.json
```

这两个原始输入文件、`state/infinilm-condition.success`、对应诊断扩展与构建日志目录都需保留；本地下载的输入无需再上传。脚本读取构建目录中的 `artifacts.sha256`、`minimax_h3_transformer.cpp` 与 `.cpp.trace.json`。

单卡 Native、八卡 Native、单卡参考进程顺序执行，使用相同权重和已保存输入。参考路径仅用于诊断。此次只执行一个主 Transformer 块，不是正式评测工作量。

## 回传结果

日志目录：`$WORK_HOME/logs/infinifabric-h3-conditions-<本次编号>/`。

完成后返回该目录中的：

1. `condition-parity.json`
2. `run.log`（也可用网页“下载日志”获得完整输出）

若失败，额外返回最后一个失败 worker 子目录中的 `worker.log`。若在预检阶段失败，则只返回报告与主日志即可，不要反复重跑。

成功执行会显示 `CONDITION_DIAGNOSTIC COMPLETE`、报告 `COMPARISON_COMPLETE` 且退出码0。此状态只表示三路采样和比较完成；误差是否可接受、视频质量是否恢复需另行判断。本轮没有视频输出。

24 个采样边界覆盖条件输入与投影、两个 refiner 子块的 norm1/Attention/残差/norm2/MLP/残差、最终归一化，以及原有九个主块边界。每个张量最多保存65536点，不能据此声称全张量正确或完整生成通过。正式扩展保持原样。
