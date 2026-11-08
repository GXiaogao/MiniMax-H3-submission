# 九齿门控残差 → InfiniCore → H3 接入验证包

前置：NineToothed 0.26.0/Triton 3.8.0/海光 HIP 属性兼容补丁已通过 39 项 JIT；2026-10-27 07:49 独立 C++ HIP 启动的 33 项用例通过。本包进入统一接口及模型调用验证，**发布时尚未在服务器编译或运行**。

## 上传与执行

上传 `h3-gated-integration-v1.tar.gz` 到 `$WORK_HOME/scripts/`。在已经 source 初始化脚本的 Pod 中：

```bash
tar -xzf "$WORK_HOME/scripts/h3-gated-integration-v1.tar.gz" -C "$WORK_HOME/scripts"
run_logged h3_gated_integration.log bash "$WORK_HOME/scripts/h3-gated-integration-v1/run.sh"
```

构建会增量编译 InfiniCore 主库和 Python 绑定、InfiniLM Python 扩展。日志全部放在 `$WORK_HOME/logs` 下，不重复安装 wheel、不修改 `setup_env.sh`。默认 8 个编译任务，可在运行前设置 `H3_BUILD_JOBS`。

若构建已经成功，只需要复跑测试：

```bash
run_logged h3_gated_integration_retry.log bash "$WORK_HOME/scripts/h3-gated-integration-v1/run.sh" --verify-only
```

成功标志 `INTEGRATION CHECKS PASSED`、exit code 0。回传末尾 `EVIDENCE ZIP:` 的压缩包和外层日志；构建失败时再提供 `h3_gated_build_*` 目录中的 `.log`，不必上传 `installed_backup/`。保留运行命令、环境、用例数与退出码截图。失败同样保留日志；不要删除检查或放宽误差。

## 改动和部署

- 只接受 InfiniCore `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`、InfiniLM `65deb91f840ebaef752d028fad2f3217ce0950f4`，并验证已通过的 MHA/singleton、Qwen 视觉修复前置源码哈希。
- 两仓库受改文件全部预检后才写入；未知改动、部分应用、坏文件拒绝覆盖。写入失败按原始字节恢复。本包源代码清单和可阅读 patch 均保留。
- 新增 `infinicore::op::gated_residual`、Python 公共入口 `infinicore.gated_residual` 与 HYGON dispatcher。C++ 后端通过已验证的宿主启动器启动九齿生成的 HSACO；推理进程不调用 Python Triton/NineToothed，也不使用 Torch 实现被测计算。
- 构建时独立候选解释器重新导出三种 dtype 内核，核验 22 参数 ABI，并构建宿主启动库；资产放在 `$WORK_HOME/tools/h3-gated-bundle-<版本目录>/`。正常初始化 Python 执行所有公共接口与模型测试。
- H3 只替换 TransformerBlock 中 attention、MLP 的两处门控残差。`INFINILM_H3_NINETOOTHED_GATED=1` 启用，未设置或 `0` 使用原有 mul+add；其他值报错，启用失败不会退回旧计算。
- 测试子进程通过 `INFINICORE_H3_GATED_BUNDLE` 选择已校验资产。运行脚本不会向用户交互 Shell 或初始化脚本持久设置启用开关。
- 旧安装库/扩展备份位于构建日志的 `installed_backup/`，源码备份位于 `$WORK_HOME/logs/h3_gated_source_backup_*`。构建失败后不要混用新旧安装库；保留日志，按备份精确恢复或修复后重建，不做全仓库 reset。

## 验证项目

| 阶段 | 数量和要求 |
| --- | --- |
| 公共算子 | 36 个数值用例（含原30项、3原地更新、3个[1,N,H]）；CPU参考全等；8个拒绝用例，不得启动内核 |
| H3 开关关闭 | 6 个微型模型组合、18 次前向；原 CPU 参考与输入保护检查；九齿调用数 0 |
| H3 开关开启 | 同上；每次前向恰好 2 次九齿调用，共36次 |
| Encoder→H3 关闭/开启 | 每轮3对模型、12次链路；原Encoder/H3/端到端参考；关闭0次、开启24次九齿调用 |
| MHA | 原85项回归，含原MHA设备图用例 |
| 开关输出对照 | 18份H3与12份桥接NPZ分别比较开关前后同字段，要求dtype一致、逐元素相等，数值字段同时要求有限 |

父进程对每个子阶段核对退出码、run_id、数量及完成标志；失败/崩溃/超时后后续阶段不运行并保留ZIP。证据包包含本次导出的HSACO、IR、ABI元数据及其清单，记录实际加载的InfiniCore库哈希。

## 支持范围和限制

- `[N,H]` 或 `[1,N,H]`，三输入同形、同dtype、同device；FP32/FP16/BF16，保留 `residual + cast_dtype(gate*branch)` 的中间舍入，关闭FMA。
- 支持 6H gate stride/偏移、非连续输入输出、零stride广播；输出不得重叠，只允许与residual完全同一视图，拒绝其他输出/输入地址范围重叠。空维度、负stride、batch>1及超出int32边界不在本轮支持范围。
- 按 device/dtype 缓存最多三个 HIP 模块，保留至进程退出，避免异步执行或已记录图引用到卸载代码。新算子显式声明不进行设备图捕获；InfiniCore 的宿主图调度仍可保存该 op，但本轮模型测试关闭图编译。MHA的图回归不等于新算子的设备图验证。
- 单卡微型随机权重与合成输入，未替代官方配置/权重，未修改保护的评测程序。没有正式视频、多卡、Ref2VA或性能提升结论。
- 不开启全局 `ninetoothed=y`；这条新增后端独立使用九齿生成物，已验收MHA仍使用原InfiniOp组合实现。
- 正常关闭宿主模块时会检查错误；进程异常/清理路径的 `nodiscard` 编译警告来自前轮已验证宿主文件，本包保留该源文件以复用证据。

## 文件来源

`common.py`、`gated_residual.py`、`abi.py`、`native.py`、`native_launcher.cpp` 复用已验收独立启动包；`jit_checks.py` 只增加构造handle与launch的显式注入参数，保留全部输入、参考与容差。模型、Encoder参考和MHA验证脚本为先前已通过的原样副本；外围只读取调用计数，不替换模型计算。新增工作副本文件在 `payload/`，哈希在 `patch_manifest.json`，改动可读文件在 `changes.patch`。
