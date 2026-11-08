# MiniMax-H3 FL2VA 适配阶段提交目录

本目录是2026年11月8日从实际开发目录复制的可审阅材料，保留原开发目录，不是最终验收版本。
当前 Native 完整生成768×928、124帧；原版比较 Average SSIM 44.643%。Qwen第50层输出相对L2为71.698%，仍在排查。
九齿已有gfx906单卡门控残差接入及gfx936独立JIT证据，正式八卡模型中的九齿调用及收益尚未验证；没有整体加速结论。

## Native 与 PyTorch 参考实现的关系

本项目适配的生成模型始终是 **MiniMax-H3 FL2VA**。Native与PyTorch表示执行模型计算的两条软件路径，不是两种不同的生成模型；Qwen3-VL是其中的条件编码模块，不是替换MiniMax-H3的视频生成模型。

**下面两段视频都由本项目的服务器作业生成。两路收到的是同一张首帧图片和文字提示，没有把一段参考视频输入给Native，也没有从网上下载视频当作生成目标。**

| 项目 | Native 适配实现 | PyTorch 参考实现 |
|---|---|---|
| 模型与权重 | 指定MiniMax-H3模型结构及权重 | 同一指定模型及权重 |
| 本次外部输入 | 模板首帧PNG与`A dog falls asleep` | 同一PNG与提示词 |
| 生成参数 | seed42、50步参数、124帧、默认画布 | 相同生成参数 |
| 核心执行路径 | Qwen条件编码与H3核心走InfiniLM／InfiniCore；新增九齿能力按实际接入范围记录 | 独立运行指定的原始PyTorch参考实现 |
| 产生的输出 | Native实现生成结果 | PyTorch参考实现生成结果 |
| 在本项目中的作用 | 验证参赛适配，并在正确性满足要求后优化 | 独立对照适配结果，不作为Native的输入 |

流程是：**同一首帧与提示词 → 分别运行两条实现 → 得到两个输出视频 → 使用原版比较脚本计算规格与SSIM。** 本页“参考实现视频”指第二条路径的输出，不是Ref2VA任务所需的输入参考视频。本次展示的是FL2VA首帧生成样例。

项目目标是先保持指定权重、输入、生成工作量及计算语义，验证Native与参考实现的一致性；然后围绕算子、显存、传输、多卡协作和调度优化。不是训练另一个模型，也不为贴近某一个视频而改权重、裁剪异常值、硬编码或缓存生成结果。合理浮点差异不要求逐像素完全相等，验收以主办方指定比较方式及要求为准。

当前比较尚有差距：Qwen外部输入一致而内部输出存在集中偏差；两路VAE精度设置也有已记录差异，尚未证明各自对最终SSIM的影响。因此参考输出是对照基准，不称为不可质疑的“标准答案”，现有结果也不宣称已达到官方质量验收。

更多来源、配置差异与验证范围见[两条路径及视频来源说明](docs/NATIVE-AND-REFERENCE.md)。

## 视频结果展示

当前阶段样例：相同模板首帧、Prompt `A dog falls asleep`、seed42、50步参数，输出768×928、124帧、24fps。原版比较Size/Length通过，Average SSIM **44.643%**。这是阶段输出，Qwen一致性和最终画质仍在改进，没有官方验收或端到端加速结论。

[Native适配实现生成结果 MP4](results/demo/native-fl2va-50steps.mp4) · [PyTorch参考实现生成结果 MP4](results/demo/reference-fl2va-50steps.mp4) · [视频来源与SHA256](results/demo/media-manifest.json)

下表为原始视频第0、62、123帧，完整视频请点击上方链接查看或下载。

| 来源 | 第0帧 | 第62帧 | 第123帧 |
|---|---|---|---|
| Native适配实现 | ![Native frame 0](results/media-recovery-result-20261107T090136Z-3046211/generated-000.png) | ![Native frame 62](results/media-recovery-result-20261107T090136Z-3046211/generated-062.png) | ![Native frame 123](results/media-recovery-result-20261107T090136Z-3046211/generated-123.png) |
| PyTorch参考实现 | ![参考 frame 0](results/reference-compare-result-20261107T124511Z-3527846/generated-000.png) | ![参考 frame 62](results/reference-compare-result-20261107T124511Z-3527846/generated-062.png) | ![参考 frame 123](results/reference-compare-result-20261107T124511Z-3527846/generated-123.png) |

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

## 从 GitHub 源码运行

请先阅读[从GitHub源码到服务器验证的完整步骤](docs/FROM-GITHUB-TO-SERVER.md)：固定版本下载 → AFS前提核对 → 从源码组装14文件上传包 → 隔离重建 → 候选身份核验 → 首次前向验证。说明明确区分已验证候选重放、新候选接入与仍待验证的空环境部署。

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

已按实际服务器trace核对11项摘要，并完成补丁应用往返检查。还原范围、操作及限制见docs/LOAD-SYNC-SOURCE-RESTORATION.md。随后已完成服务器隔离重建及单组首次前向复现，见[重建前向结果](docs/REBUILT-FORWARD-RESULT.md)；Qwen一致性仍未修复。
