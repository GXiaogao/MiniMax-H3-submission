# 本次选“复核已有视频”，不用重新生成

v1.1本次视频已被官方入口保存，辅助检查错误地要求50次调用；固定调度器的50点参数实际对应49次更新。v1.2保留50步请求，按真实调度长度验证。

请把本修复包的三个代码文件上传到原运行项目：覆盖`run_minimax_h3_load_sync_generation.py`，新增`recover_minimax_h3_generation_media.py`和`.sh`。原图、旧辅助文件、模型及wheel不用重传，不重新编译。

**这次入口选`recover_minimax_h3_generation_media.sh`，参数留空，运行时长5分钟。** 本脚本只读原MP4并抽帧，不执行新推理；平台若固定整节点，沿用当前资源设置即可。内部4分钟上限。

它只读取本次原作业：

`$WORK_HOME/logs/infinifabric-h3-load-sync-generation-20261107T082020Z-3236632/native/`

请保留其中的`native-generation.json`、`worker.log`和`fl2va-native-load-sync-50steps.mp4`。原视频现在就可下载查看，无需等待复核脚本。

新复核结果在：

`$WORK_HOME/logs/infinifabric-h3-existing-media-<时间>-<PID>/`

返回`media-recovery.json`、`run.log`，以及`generated-000.png`、031、062、093、123，或直接返回原MP4。这些抽帧没有改变视频像素。若失败，返回完整JSON和日志，不删除原作业。

`EXISTING_MEDIA_VALIDATED`表示原视频可完整解码且规格通过；不代表画质/SSIM/比赛验收或新一轮推理成功。实际画面复核与参考比较继续进行。

更新后的生成runner用于将来的新生成：原`.sh`入口和50步/124帧/seed42不变，报告版本`1.2-load-sync-template-Native-actual-schedules`，新增真实schedule字段。本次无需再跑该生成入口。
