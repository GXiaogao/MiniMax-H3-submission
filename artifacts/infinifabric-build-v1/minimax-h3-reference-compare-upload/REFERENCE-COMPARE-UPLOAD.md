# 下一阶段：原始参考生成及视频质量比较

解压包内文件并全部上传到原运行项目，覆盖同名辅助文件。入口选择：

`run_minimax_h3_reference_compare.sh`

程序参数留空，八卡整节点，运行时长28分钟；内部25分钟上限，失败保留日志。**这次不用重跑Native，也不用重传图片。** 图片从之前持久化的个人AFS读取并校验。

脚本依次执行：原Native MP4及已返回的复核报告校验→离线运行时暂存→SSIM依赖自检→原始参考生成→参考媒体规格/抽帧检查→未改动原版比较脚本。依赖不兼容时会在参考模型加载前停止，不自动联网安装。

统一输入为模板原始PNG、A dog falls asleep、seed42、50点请求、124帧、默认768×928画布。参考侧使用原入口`--legacy`，Qwen放GPU0至3、H3放GPU4至7，不允许CPU/disk offload。该设备划分不同于Native TP8，是质量参考配置，本次不计算加速比。

保护原始基线：不改原始参考入口、权重或比较脚本，不对参考VAE应用Native已有的精度补丁。报告会列出参考各组件的精度审计及这项差异；这不是已统一辅助精度的质量验收，也不根据某个预设SSIM阈值宣布比赛通过。参考工具仅用于独立验证，不替代参赛Native核心。

输出目录：

`$WORK_HOME/logs/infinifabric-h3-reference-compare-<时间>-<PID>/`

请返回：

- `reference-comparison.json`
- `run.log`
- `official-comparison.log`（原版脚本Size、Length、Average SSIM输出）
- `reference/reference-media.json`
- `reference/generated-000.png`、031、062、093、123；或参考MP4。

参考视频在`reference/fl2va-reference-original-50steps.mp4`。如果比较阶段失败，视频和reference-media.json仍会保存，不必立即重跑生成；先返回父JSON/run.log及失败阶段worker.log。如果早期失败，还没有official-comparison.log，返回现有日志即可。

结果`QUALITY_COMPARISON_COMPLETE`只说明质量比较执行完成。SSIM是原版脚本按灰度、最大边512计算后显示的三位小数百分比；JSON不另造精确值或合格阈值。provided-video比较没有推理计时，不能拿它宣称加速。全片人工复核、正式八卡九齿调用/收益、干净重建和最终报告仍需继续完成。

请保留原Native作业目录`infinifabric-h3-load-sync-generation-20261107T082020Z-3236632`、复核目录`infinifabric-h3-existing-media-20261107T090136Z-3046211`、`assets`模板PNG、共享模型、state、运行时wheel及当前源码。不要把大型模型/wheel重新上传到项目输入区。
