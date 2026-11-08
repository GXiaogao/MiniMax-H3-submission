# 第一个 token refiner Attention：诊断构建

上一轮已定位：Attention 前单卡/八卡的归一化采样值一致，但 Attention 后八卡相对 L2 误差约75.55%。本轮继续记录 Attention 内部数据，尚未修改数学实现。

## 上传与启动

本地解压 `minimax-h3-attention-build-upload.zip`，将下面七个运行文件上传到平台同一项目目录：

- `build_minimax_h3_attention_trace.sh`
- `prepare_minimax_h3_attention_trace.py`
- `prepare_minimax_h3_condition_trace.py`
- `prepare_minimax_h3_boundary_trace.py`
- `h3_attention_trace.hpp`
- `h3_boundary_trace.hpp`
- `build_infinilm_extension.sh`

已有同名依赖可核对包内校验清单后保留，使用本包版本。特别注意上传新增的 `h3_attention_trace.hpp`。模型、wheel 与仓库继续使用个人 AFS 中的现有文件。

启动设置：

- 入口：**`build_minimax_h3_attention_trace.sh`**。
- 方式：批处理作业；沿用已验证的 `scnet-dcu` 八卡整节点环境。
- 程序参数：**留空**。
- 时长：**28 分钟**；脚本内部构建超时25分钟。

本轮只构建诊断扩展，不生成视频。新扩展发布至 `$WORK_HOME/diagnostics/h3-attention/<本次编号>/InfiniLM/`，使用独立标记 `$WORK_HOME/state/infinilm-attention.success`。正式扩展与此前 condition/boundary 扩展保留。

## 回传文件

运行成功显示 `ISOLATED_ATTENTION_BUILD COMPLETE` 且退出码0后，请返回：

1. 完整下载日志。
2. 个人 AFS 的 `minimax-h3/state/infinilm-attention.success`。

如果失败，返回完整日志与首次失败阶段，先不要重复运行。

日志位于 `$WORK_HOME/logs/infinifabric-prepare-h3-attention-*` 和 `infinifabric-build-h3-attention-*`。保留新构建日志、现有 `first-inputs.npz/.json` 和上一轮条件采样结果，以便下一步加载新扩展对比。

## 诊断范围

保留已有24个 rank0 边界；新增八个边界在每个 TP rank 分别采样：输入、原始Q/K/V、归一化Q/K、局部MHA、输出投影与归约后的结果。新文件名包含 rank，不同卡分别记录。

采样仅在显式启用且进入第一个 refiner Attention 时执行，每个张量最多65536点。新构建完成后再运行采样对比，无需重新上传原始输入。未来参考对比按每个 token 的头维度切片再采样，TP8对应每卡7个头，不能把一张卡的分片与全56头直接比较。

输出投影与归约仍是原组合接口；若误差在该组合阶段出现，后续再分开定位。构建成功不代表算子数值正确、完整视频质量恢复或性能提升。
