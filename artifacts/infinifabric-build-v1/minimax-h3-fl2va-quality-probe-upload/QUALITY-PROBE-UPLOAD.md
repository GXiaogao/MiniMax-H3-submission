# Native FL2VA 画面质量诊断：上传和运行

这是一次 VAE 精度原因排查，尚未证明画面问题已修复。

## 上传

解压 `minimax-h3-fl2va-quality-probe-upload.zip`，把下面五个脚本放到同一个平台项目的根目录。已有同名文件时使用包内版本替换；不要只上传 ZIP 后直接启动。

| 文件 | 用途 |
| --- | --- |
| run_minimax_h3_fl2va_native_quality_probe.sh | 本次新增启动入口 |
| prepare_minimax_h3_vae_dtype_probe.py | 本次新增临时入口补丁器 |
| run_minimax_h3_fl2va_native_smoke.py | 本次更新的运行 helper |
| install_infinilm_private_runtime.py | 沿用已通过的私有依赖 helper |
| prepare_minimax_h3_hygon_runtime.py | 沿用已通过的海光运行时 helper |

`SHA256SUMS.quality-probe` 用于核对包内文件。模型、wheel、首帧和已构建的 InfiniLM/InfiniCore 继续使用现有 AFS/共享存储内容，无需重新上传或编译。

## 页面设置

- 入口文件：`run_minimax_h3_fl2va_native_quality_probe.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行环境：此前成功运行的 scnet-dcu 环境
- 运行时长：28 分钟
- 卡数：平台整节点 8 卡

固定 A/B 配置为 124 帧、10 步、seed 42、576×384、`A dog falls asleep`、`$WORK_HOME/inputs/fl2va/Golde33443.jpg`。这里的 10 步用于复现当前故障，最终比赛配置须按主办方指定参数验证。

## 取回结果

视频：`$WORK_HOME/outputs/fl2va-native-quality-probe/` 中本次新生成的 MP4。

日志：`$WORK_HOME/logs/infinifabric-run-minimax-h3-native-fl2va-quality-probe-*` 中的 `run.log`、`summary.txt`、`vae-dtype-patch.json`、`minimax-h3-native-fl2va-formal.json`。若补丁阶段失败，再取回 `vae-dtype-patch.log`。

末尾出现 `STRUCTURAL_PASSED` 和 `QUALITY_PROBE RUN_COMPLETE` 仅说明作业及媒体检查完成，`quality_status=UNASSESSED` 表示仍需核对画面。请提供 **MP4 与完整 run.log**，不要只提供页面“成功”提示。

## 诊断边界

补丁只移除原示例中 VAE/Audio-VAE 的两处强制 BF16 转换。持久化官方入口和权重保持原样，源码不同或目标已存在时会停止。输出正常后还需参考相似度验证；若仍有彩色块，则继续隔离 VAE 与 Native Transformer，不能把本次假设当成已证实根因。
