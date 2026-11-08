# 完整 Native FL2VA 重新生成

2026-11-07 v1.2更正：50步参数对应本调度器49次实际更新，完成门控已改为读取真实schedule。已生成MP4但遇v1.1固定次数误判的本次用户，请按`RECOVER-GENERATION-MEDIA.md`运行现有媒体复核，无需重新生成。本文以下“50次视频/音频调度”描述是旧门控要求，应以实际schedule长度取代；50步参数及完整工作量不变。

2026-11-07 v1.1更新：修复进度观察器丢失forward签名、导致5个布局参数漏传的问题。已上传v1.0全部文件的用户只需覆盖`run_minimax_h3_load_sync_generation.py`，入口、图片和其余依赖不变。新JSON版本为`1.1-load-sync-template-Native-signature-preserved`。新用户使用v1.1完整包。

前提：已完成并保留 `20261107T065018Z-3136076` 的 full-forward-parity.json，以及 load-sync 候选和构建记录、共享Core库、私有离线运行时。新包不重新编译、不替换正式库；仍实际加载修复候选 ee448772...。

解压同目录文件，全部上传到运行项目，覆盖包内同名辅助文件。新增入口：

`run_minimax_h3_load_sync_generation.sh`

程序参数留空，使用8卡整节点，申请28分钟。内部25分钟限时，含准备/装载；默认完整画布的50步可能超时，不保证在限时内完成。不要通过减步数/帧数/分辨率替代模板比较。

包内 `minimax-h3-template-first-frame.png` 是报告模板嵌入的原始PNG（800×965），不是旧JPEG。脚本校验并持久保存到个人AFS `$WORK_HOME/assets/template-first-frame-<SHA256>.png`，不覆盖旧素材。无需再次手工选图或改参数。

工作量：Native Qwen与完整50主块H3，Prompt `A dog falls asleep`、50步、124帧、seed42、GPU0至7。与固定比较脚本相同，不传height/width，按实际共享模型的pipeline配置解析并记录画布；依赖默认短边768时该图得到768宽×928高，不能把旧576×384诊断片视为相同配置。日志会打印resolved_canvas。仅作业本地入口复制保留已验证的VAE FP32模块，原始入口/比较脚本/权重不改。trace和图关闭；不使用历史张量代替本次编码。

运行结果位于 `$WORK_HOME/logs/infinifabric-h3-load-sync-generation-<时间>-<PID>/`：

- `native-generation.json`、`run.log`、`summary.txt`。
- `native/native-generation.json`、`native/worker.log`：详细进度、实际画布、Native路径及完成调用次数。
- 成功时 `native/fl2va-native-load-sync-50steps.mp4`，以及 `native/generated-000.png`、031、062、093、123。

请返回父目录的JSON、run.log及MP4；若MP4过大先取五张PNG。失败或超时返回两个JSON、run.log与worker.log；不要只截最后一行。常规阶段与每次计算有START/DONE，长阶段另有30秒心跳。超时可能只写入子报告部分进度，summary记录非零退出，不会被判为通过。

`MEDIA_COMPLETE` 仅表示完整官方入口返回、设备同步、50次视频/音频调度及完整媒体解码检查完成；画质为UNASSESSED、官方SSIM未测、性能比较无效。本次是质量恢复验证，不是比赛验收或加速比。随后分作业执行同配置参考生成和未改动compare脚本。九齿正式八卡接入及收益另行验证。

不要删除服务器 `state`、`diagnostics/h3-load-sync/20261107T030125Z-2935713`、对应构建日志、full-forward报告、模型/运行时/wheel目录；脚本需要读取并核对这些文件。平台“我的项目”只放包内小文件，依赖/模型继续保留在个人AFS。
