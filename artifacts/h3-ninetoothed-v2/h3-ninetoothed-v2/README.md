# 九齿 gfx906 新候选验证包（v2）

状态：本地准备及宿主测试阶段；没有海光设备实测结果。本包不会修改 InfiniCore/InfiniLM 源码、现有库、setup_env.sh 或旧工具链环境。

## 本轮内容

- 独立 venv：`$WORK_HOME/venvs/h3-ninetoothed-026-triton380`，继承系统海光 Torch 2.9.0。
- 固定 NineToothed 0.26.0、Triton 3.8.0、NumPy 1.26.4、SymPy 1.14.0、mpmath 1.3.0。
- 三阶段独立子进程：环境检查、9 项九齿加法、30 项本项目原创门控残差表达检查。
- 三种精度：FP32/FP16/BF16；gate 6H 行跨度、2H/5H 偏移、非整块 H、输入/输出非连续布局、singleton 零 stride、广播和同一内核连续调用。
- 门控残差按原有 `mul→add` 的中间 dtype 舍入；设置 `enable_fp_fusion=False`。九齿 0.26.0 API 未暴露该选项，因此仅在其生成的 host launch 增加编译选项，并保留变更前后源码与哈希。没有手写 Triton 计算内核。
- Torch 仅在独立验证程序中用于输入存储、CPU 参考和检查。本包不是正式模型推理实现。

官方 Triton 3.8.0 发布说明列出 gfx906 支持，但不构成对当前海光 DTK/HIP 的兼容保证：
https://github.com/triton-lang/triton/releases/tag/v3.8.0
https://github.com/triton-lang/triton/pull/9628

## 文件准备

本地 wheel 目录固定为 `E:/projects/minimax-h3-match/相关代码文件/wheel`。上传到服务器 `$WORK_HOME/wheels`：

1. `triton-3.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`（新增，约 248 MB）
2. `h3_toolchain_v2_manifest.json`（新增，不能用旧的 h3_toolchain_manifest.json 替代）
3. `ninetoothed-0.26.0-py3-none-any.whl`
4. `numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
5. `sympy-1.14.0-py3-none-any.whl`
6. `mpmath-1.3.0-py3-none-any.whl`

后四个若已在服务器 wheels 中且校验一致，无需重传。旧海光 Triton 3.5.1 包可保留，本包不会使用它。下载器 `prepare.py` 仅下载/校验，不在 Windows 安装这些 Linux wheel。

把 `artifacts/h3-ninetoothed-v2.tar.gz` 上传到 `$WORK_HOME/scripts`。进入大赛 Pod 的 root 容器终端后：

```bash
source /root/private_data/setup_env.sh
run_logged h3_toolchain_v2_unpack.log tar -xzf "$WORK_HOME/scripts/h3-ninetoothed-v2.tar.gz" -C "$WORK_HOME/scripts" &&
run_logged h3_toolchain_v2.log bash "$WORK_HOME/scripts/h3-ninetoothed-v2/run.sh"
```

不要手动 activate 新 venv，不用重新配置 xmake。`run.sh` 校验脚本包后安装/检查候选，并在独立解释器中验证。候选包已完整安装时跳过 pip。所有新日志、编译缓存、生成源码和 NPZ 都在 `$WORK_HOME/logs/h3_ninetoothed_v2_<时间_PID_随机串>/`。

## 判定和回传

全部通过须满足：三阶段 PASS、add 9 项、gated residual 30 项、`JIT CHECKS PASSED`、外层 exit code 0。检查要求有限值和与分步参考逐值一致；不能通过放宽容差掩盖中间舍入不同。

任何缺包、编译错误、数值差异、崩溃、超时或结果计数错误均返回非零；之后的阶段标记 NOT_RUN。失败时保留原始日志和已完成用例。无需反复运行旧候选或重新安装主环境。

请回传：

- `$WORK_HOME/logs/h3_toolchain_v2.log`
- 主日志末尾 `EVIDENCE ZIP:` 指向的 `h3_ninetoothed_v2_<...>.zip`。

ZIP 包括 summary.json、各阶段 result.json/日志、生成源码、编译产物/元数据和 NPZ，用于确认 gfx906 实际执行及下一轮 C++ ABI 适配。

即使本轮全通过，也会明确显示 `AOT / InfiniCore integration / H3 regression: NOT VALIDATED`。这不是错误：下一阶段才验证 HIP C++ 调用、InfiniCore 分发、H3 两处调用替换及回归。本轮不发布“已完成完整接入”或“性能提升”的结论。

## 源码关系

- `gated_residual.py`：本项目新增的九齿计算表达。
- `verify.py`：设备测试与证据采集，不进入参赛正式推理路径。
- `common.py`：校验、报告判定、进程隔离和生成 host launch 选项。
- `prepare.py`：本地离线包下载。
- `setup.py/setup.sh`：候选 venv 安装；不替换系统 Torch。
- `run.sh`：服务器统一入口。

已有 MHA、Qwen 视频修复和 Encoder→H3 证据保持原样。完整计划见 `docs/superpowers/plans/2026-10-26-ninetoothed-gated-residual.md`。
