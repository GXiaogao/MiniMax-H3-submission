# 输出投影与归约：第一步构建诊断扩展

上一轮已定位：每卡MHA采样误差约0.6%，输出投影/归约后约79.71%。本轮增加实际权重、原接口输出、本地复算与受控归约的记录，尚未修复数学实现。

## 上传

Windows本地解压 `minimax-h3-projection-build-upload.zip`，将以下九个运行文件上传到同一平台项目目录：

- `build_minimax_h3_projection_trace.sh`
- `prepare_minimax_h3_projection_trace.py`
- `prepare_minimax_h3_attention_trace.py`
- `prepare_minimax_h3_condition_trace.py`
- `prepare_minimax_h3_boundary_trace.py`
- `h3_projection_trace.hpp`
- `h3_attention_trace.hpp`
- `h3_boundary_trace.hpp`
- `build_infinilm_extension.sh`

已有同名依赖与本包校验清单核对后可保留，使用本包版本。无需上传新wheel、模型或原始输入。

## 启动

- 入口：**`build_minimax_h3_projection_trace.sh`**。
- 环境：沿用已验证的 `scnet-dcu` 八卡批处理。
- 参数：**留空**。
- 时长：**28分钟**，内部构建超时25分钟。

新扩展发布到 `$WORK_HOME/diagnostics/h3-projection/<本次编号>/InfiniLM/`，独立标记为 `$WORK_HOME/state/infinilm-projection.success`。正式扩展、InfiniCore主库和已有attention/condition扩展保留。

## 返回结果

成功显示 `ISOLATED_PROJECTION_BUILD COMPLETE` 且退出码0后，返回：

1. 完整运行日志。
2. `minimax-h3/state/infinilm-projection.success`。

日志位于 `$WORK_HOME/logs/infinifabric-prepare-h3-projection-*` 和 `infinifabric-build-h3-projection-*`。失败时返回完整日志和首次失败阶段，不要重复重跑。

本轮仅构建，不生成视频。获得新扩展校验值后，再运行复用已有first-inputs文件的采样对比。原输入、上一轮Attention结果和各构建证据目录继续保留。

## 诊断含义

原输出投影与归约先执行，保存结果；随后在第一refiner输出投影范围内，用实际加载权重复算本地矩阵乘，再做一次受控SUM归约，并检查原输出在这些控制计算前后是否变化。始终返回原接口的Tensor。

原海光路径使用原地归约，控制归约使用单独输出，这是需要区分的接口行为。控制结果不是原接口内部中间值，也不能作为性能成绩。新增计算与同步可能影响调度，后续将结合原输出前后快照判断，不能仅凭跨构建结果变化判定根因。

单卡记录串行投影作为对照；八卡分别记录权重和局部结果。采样不证明完整权重或全张量完全正确。本次不激活九齿替换正式算子，也不宣称视频质量恢复。
