# MiniMax-H3 FL2VA 适配阶段提交目录

本目录是2026年11月8日从实际开发目录复制的可审阅材料，保留原开发目录，不是最终验收版本。
当前 Native 完整生成768×928、124帧；原版比较 Average SSIM 44.643%。Qwen第50层输出相对L2为71.698%，仍在排查。
九齿已有gfx906单卡门控残差接入及gfx936独立JIT证据，正式八卡模型中的九齿调用及收益尚未验证；没有整体加速结论。

## 视频结果展示

当前阶段样例：相同模板首帧、Prompt `A dog falls asleep`、seed42、50步参数，输出768×928、124帧、24fps。原版比较Size/Length通过，Average SSIM **44.643%**。这是阶段输出，Qwen一致性和最终画质仍在改进，没有官方验收或端到端加速结论。

[Native生成视频 MP4](results/demo/native-fl2va-50steps.mp4) · [原始参考视频 MP4](results/demo/reference-fl2va-50steps.mp4) · [视频来源与SHA256](results/demo/media-manifest.json)

下表为原始视频第0、62、123帧，完整视频请点击上方链接查看或下载。

| 来源 | 第0帧 | 第62帧 | 第123帧 |
|---|---|---|---|
| Native | ![Native frame 0](results/media-recovery-result-20261107T090136Z-3046211/generated-000.png) | ![Native frame 62](results/media-recovery-result-20261107T090136Z-3046211/generated-062.png) | ![Native frame 123](results/media-recovery-result-20261107T090136Z-3046211/generated-123.png) |
| 参考 | ![参考 frame 0](results/reference-compare-result-20261107T124511Z-3527846/generated-000.png) | ![参考 frame 62](results/reference-compare-result-20261107T124511Z-3527846/generated-062.png) | ![参考 frame 123](results/reference-compare-result-20261107T124511Z-3527846/generated-123.png) |

Native视频来自2026年11月7日旧候选ee448772…的完整生成；不是2026年11月8日新重建库189e52b7…重新生成的结果。新库目前完成源码摘要、隔离重建及同输入H3首次前向复现。视频原件保持原始字节，不通过重新编码、挑帧替换或历史结果缓存参与正式推理。

## 当前优先事项

1. 对照视觉pooler、DeepStack、语言输入embedding和语言层中间结果，定位Qwen最早偏差；已确认两侧外部输入完全一致，误差集中于有效图像token。
2. 正确性恢复后，使用固定完整生成配置重做Native/参考和原版比较。
3. 验证正式九齿调用、受控性能及连续多用例，再更新最终报告和提交版本。

## 目录

- artifacts/：保持原运行包内部路径的源码、补丁生成器、构建/运行/诊断脚本。infinifabric-build-v1 是当前八卡材料，其他包是历史单卡证据。
- src/：九齿v3和性能包解包后的文本源码，以及 git diff 未覆盖的新增实现。
- patches/：本地工作树改动和适用基线说明。
- scripts/、tests/：原有开发辅助程序和测试；旧Pod脚本不是八卡推荐入口。
- configs/：本地版本来源。服务器无Git元数据，已记录源码包/变换/候选摘要及重建证据。
- docs/：规则整理和关键结果记录。
- results/：本地已有的原始JSON、日志和抽帧。只作验证证据，不是推理输入缓存。
- reports/：阶段Word初稿，队伍、仓库和最终commit待填写。
- requirements/：依赖位置与下载边界说明。
- server-downloads/：只在本地收取服务器材料，已被Git忽略。

## 当前入口

在 artifacts/infinifabric-build-v1 查看 LOAD-SYNC-BUILD-UPLOAD.md、FULL-FORWARD-LOAD-SYNC-UPLOAD.md、LOAD-SYNC-GENERATION-UPLOAD.md、REFERENCE-COMPARE-UPLOAD.md。
生成入口为 run_minimax_h3_load_sync_generation.sh，参考比较为 run_minimax_h3_reference_compare.sh；diagnose_* 为诊断，不是正式生成。
包说明包含旧版本历史，真实调度完成门控以当前v1.2代码及回执为准。
各作业分开提交，申请28分钟；模型和wheel放AFS，不放平台项目输入目录。
现有候选路径及回执存在绑定，尚不能宣称本目录能一键干净重建；需补服务器实际源码、候选构建来源并验证。

## Git状态

本目录作为开发快照纳入Git；提交/推送状态以实际Git历史为准，不改变仓库可见性。服务器采集、候选隔离重建及单组首次前向已完成；完整环境重建与连续多用例仍待验证。
先核对 docs/WEBSITE-DOWNLOADS.md 和 manifests，再选择提交文件。
模型、wheel、私密凭据和大型中间张量不进Git；仅明确列出的两份已校验演示视频纳入results/demo。源码包保留是为了固定现有离线构建输入。
第三方代码遵守各自原许可证，本队许可证和最终依赖许可清单待确定。

## 服务器证据采集

操作见 docs/SERVER-COLLECTION.md；上传scripts中的collect_server_submission.sh和同名.py两个文件即可。结果回传前，不把Windows本地基线当作服务器版本。提交进度见docs/SUBMISSION-PROGRESS.md。

## 装载候选源码还原

已按实际服务器trace核对11项摘要，并完成补丁应用往返检查。还原范围、操作及限制见docs/LOAD-SYNC-SOURCE-RESTORATION.md。此项不等于服务器二进制已重建或Qwen一致性已修复。
