# Native 生成链路诊断

前置结论：独立 VAE 的 22 帧图片重建正常；完整 FL2VA 视频正确性仍未通过。本次记录真实 Native 生成过程，继续定位彩色噪声来源。不要重新编译或安装模型环境。

## 上传与启动

1. 在本机解压 `minimax-h3-native-chain-upload.zip`。
2. 将包内 **8 个 .py/.sh 文件** 上传到平台“我的项目 → Minimax-h3-runner”同一目录。已有同名 helper 需更新为包内版本；不要上传整个压缩包作为项目输入，也不要重新上传模型/wheel。
3. 入口文件选 `diagnose_minimax_h3_native_chain.sh`；程序参数留空；使用方式“批处理作业”；运行环境选择此前成功的 scnet-dcu；时长 28 分钟。脚本内部超时 25 分钟。
4. 正常完成后从个人 AFS 下载下面三份文件：

   - `run.log`
   - `native-chain-summary.json`
   - `native-chain.jsonl`

三份文件位于同一个新日志目录：

```text
minimax-h3/logs/infinifabric-run-minimax-h3-native-fl2va-native-chain-trace-<运行编号>/
```

同目录还会生成 `condition-normalized-001.npz` 和 `decoder-input-001.npz`，保留供后续比对；如方便，也一并下载。视频写到：

```text
minimax-h3/outputs/fl2va-native-native-chain-trace/
```

入口沿用当前 8 卡、124 帧、10 步参数、seed 42、576×384、`A dog falls asleep`，并沿用保留 VAE 必需 FP32 参数的作业副本。此处没有降低现有生成工作量。原 AFS 模型与源码不覆盖。

## 文件清单

| 文件 | 用途 |
| --- | --- |
| diagnose_minimax_h3_native_chain.sh | 本次选择的入口 |
| diagnose_minimax_h3_native_chain.py | 原生链路观察与持久日志 |
| run_minimax_h3_fl2va_native_quality_probe.sh | 复用已有准备与固定启动流程，包内为更新版 |
| prepare_minimax_h3_vae_dtype_probe.py | 仅处理本作业入口副本的两处 VAE dtype 转换 |
| run_minimax_h3_fl2va_native_smoke.py | 复用 Native 运行与媒体结构检查 |
| diagnose_minimax_h3_vae_roundtrip.py | 本次只复用参数精度检查函数，不重跑独立重建 |
| install_infinilm_private_runtime.py | 复用已校验的离线依赖暂存 |
| prepare_minimax_h3_hygon_runtime.py | 复用海光依赖暂存 |

日志每个环节会打印 `chain_trace ... START/DONE`，长期阶段仍有 heartbeat。遇到异常，先返回三份日志；不要因为出现 “STRUCTURAL_PASSED” 就认为画面正确。

`native-chain-summary.json` 的 `DIAGNOSTIC_COMPLETE` 仅表示诊断记录完成；`nonfinite_observations` 是发现的 NaN/Inf 位置；数值有限仍可能语义错误。均值/标准差等是标明的确定性抽样，全元素另行检查是否有限。`quality_status` 保持 `UNASSESSED`。诊断含额外同步与存盘，不能据本次时间评价加速。
