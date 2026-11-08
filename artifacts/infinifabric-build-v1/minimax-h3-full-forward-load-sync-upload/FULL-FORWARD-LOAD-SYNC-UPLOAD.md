# 完整50主块首次前向对照

同步装载候选已让八卡投影权重采样恢复一致，投影误差明显降低。本包继续使用刚验证的候选扩展，无需再次编译、下载模型或wheel。

1. 解压 `minimax-h3-full-forward-load-sync-upload.zip`，将其中9个 `.py` / `.sh` 执行文件上传到平台项目同一目录，更新同名辅助文件。
2. 入口选择 `diagnose_minimax_h3_full_forward_load_sync.sh`，参数留空，scnet-dcu八卡，时长28分钟。内部超时25分钟，装载和前向阶段每30秒输出心跳。
3. 完成后在个人存储 `minimax-h3/logs/infinifabric-h3-full-forward-load-sync-<本次ID>/` 下载 **`full-forward-parity.json` 和 `run.log`**。失败时另取 `native/worker.log` 或 `reference/worker.log`，保留整轮目录。

保留当前候选 `diagnostics/h3-load-sync/20261107T030125Z-2935713/InfiniLM/`、构建证据 `logs/infinifabric-build-h3-load-sync-20261107T030127Z-2935732/` 及状态标记。还需保留刚返回的 `logs/infinifabric-h3-load-sync-capture-20261107T034425Z-2750046/load-sync-parity.json`，原始 `first-inputs.npz/.json` 目录 `logs/infinifabric-transformer-parity-20261105T080500Z-408969/`，以及已有模型和私有运行时依赖。

本轮在两个独立进程中顺序运行：候选Native TP8完整50主块H3、参考完整50主块H3。关闭所有本候选可选追踪与图编译，不覆盖模型层数，核对50层配置及实际参数键。参考保留既有混合精度和八卡平衡分配，核对完整权重加载信息；两者均比较全部输出数组。

这只是 **完整Transformer的一次前向**，不是完整去噪/视频生成或性能验收。输入沿用同一历史捕获，Qwen编码器正确性未由此验证。候选仍含已关闭的诊断能力，不作为干净性能基线。COMPARISON_COMPLETE 只代表材料完整，需看视频和音频误差，不预设官方阈值。

原正式扩展、Core库、权重及正式生成配置均不改变。本次若超时，请保留已经写出的元数据、输出和日志，以便后续复用，不删除重跑证据。
