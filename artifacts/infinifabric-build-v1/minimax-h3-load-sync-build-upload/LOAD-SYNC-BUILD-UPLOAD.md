# 真实权重同步装载候选构建

上轮源同步控制24/24组八卡完整数组正确，未同步模式仍有大量错误。此包构建独立 Native InfiniLM 候选，用于下一步真实权重验证；不是完整生成验收或性能优化。

1. 解压 `minimax-h3-load-sync-build-upload.zip`，将其中11个 `.py`、`.sh`、`.hpp` 执行/构建文件上传到当前平台项目同一目录，更新已有同名辅助文件。源码模型和依赖仍放个人AFS，不要整目录上传进项目输入。
2. 入口选择 `build_minimax_h3_load_sync_trace.sh`，参数留空，scnet-dcu八卡，运行时长28分钟。内部构建超时25分钟。
3. 返回完整运行日志和个人存储 `minimax-h3/state/infinilm-load-sync.success`。编译成功后再根据实际扩展哈希生成配套投影对照脚本。

候选发布到 `minimax-h3/diagnostics/h3-load-sync/<构建ID>/InfiniLM`。原正式 InfiniLM 扩展、共享 InfiniCore 库、原投影诊断构建、模型权重、生成参数不变，不用重新下载wheel或模型。构建状态标记与原来的 `infinilm-projection.success` 分开。

修改限定在原生权重加载命令：HYGON跨卡来源先连续化，等待源流，复制完整参数到目标rank并等待目标完成，保留源及目标暂存到相应操作完成；随后调用现有Core装载，在本地切片。LOAD和LOAD_BATCH都覆盖，严格键检查与批末同步保留。同设备/CPU/其他设备类型沿用原路径。

该方法在装载期间增加完整参数复制及临时内存；当前仅为验证正确性的候选，不声称提速，也未证明全部正式工作量内存足够。若后续真实权重对照恢复正确，再考虑更细粒度的通用跨流依赖修复及装载开销。未新增手写设备内核，模型计算继续走Native InfiniCore。

请继续保留原始 `first-inputs.npz/.json` 所在目录、投影诊断及分片复制测试目录和状态文件，后续将复用这些证据。
