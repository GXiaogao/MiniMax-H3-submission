# 下一步：定位44.643% SSIM的上游差异

解压后全部上传到原项目，覆盖同名辅助文件。图片、wheel和模型不用重传。

入口：`diagnose_minimax_h3_input_boundaries.sh`

程序参数留空，八卡，时长28分钟；内部25分钟上限。只运行局部诊断，通常应比完整生成短，但以实际日志为准。

本轮保留原图、A dog falls asleep、50点/124帧/seed42及默认768×928画布。分别重新计算Native与原始参考的Qwen和首帧VAE条件，记录：

- Qwen真实输入张量/标量选项与第50层隐藏状态。
- 首帧VAE原始像素、归一化参数和编码结果。
- 请求每次噪声抽样及CPU RNG前后状态。
- 第一次H3调用的全部输入张量、调度与模型身份。

在H3实际计算前主动停止，**不执行去噪更新、解码或生成视频**。这不是减步数的视频，不用于比赛计时；完整50主块仍按原配置装载。Native沿用已验证的候选与VAE精度修复，参考入口保持原字节，不修改原始基线来提高SSIM。所有捕获只读，不增加随机抽样。

结果在：

`$WORK_HOME/logs/infinifabric-h3-input-boundaries-<时间>-<PID>/`

先返回父目录的：

- `input-boundary-parity.json`
- `run.log`

如失败，另返回对应`native/capture.json`、`native/worker.log`或`reference/capture.json`、`reference/worker.log`。成功时各子目录保存原始NPZ/JSON，先保留，不必全部下载。以后需要特定张量时再取对应NPZ。

成功标签为`INPUT_BOUNDARY_COMPARISON_COMPLETE`，表示捕获和比较完成，不能当作误差合格/视频生成或性能通过。形状、字段或数值不同会直接记录，浮点非零误差不自动判定为故障根因。本轮不产生新视频，日志中的`intentional_stop=true`是预期诊断停止，不是执行失败。

请保留原Native和参考视频及其原作业目录、`infinifabric-h3-existing-media-20261107T090136Z-3046211`复核报告、`infinifabric-h3-reference-compare-20261107T124511Z-3527846`比较报告、候选`diagnostics/h3-load-sync/20261107T030125Z-2935713`及对应构建日志、assets模板PNG、state、模型和运行时依赖。脚本会核对这些已有证据，不接受已改变的文件。
