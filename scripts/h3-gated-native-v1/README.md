# 九齿门控残差：独立 HIP C++ 启动验证

前置结果：2026-10-26 20:39 在 Z200SM_80/gfx906 的 NineToothed 0.26.0 / Triton 3.8.0 候选中，9 项 add 和 30 项 gated residual JIT 检查通过；原有 HIP 属性 ABI 兼容补丁已生效。

本包完成接入前的下一道检查：九齿表达生成内核，Triton 编译为参数格式固定的 HSACO，**宿主 C++ 使用 HIP API 启动**。它不安装依赖、不修改候选 venv、不修改 InfiniCore/InfiniLM，不改变 setup_env.sh。不包含手写 GPU 计算内核；Torch 只负责验证用张量、stream 与独立参考。

## 运行

上传 `h3-gated-native-v1.tar.gz` 至 `$WORK_HOME/scripts/`，在已初始化的 Pod 运行：

```bash
tar -xzf "$WORK_HOME/scripts/h3-gated-native-v1.tar.gz" -C "$WORK_HOME/scripts"
run_logged h3_gated_native.log bash "$WORK_HOME/scripts/h3-gated-native-v1/run.sh"
```

新进入 Pod 时先 `source /root/private_data/setup_env.sh`。本包使用已修复的独立候选解释器，**无需重新安装 wheel 或重复执行 HIP 修复**。要求可用的 g++；整个过程离线。

所有生成物写入 `$WORK_HOME/logs/h3_gated_native_<时间_标识>/`。返回最后打印的 `EVIDENCE ZIP` 和 `h3_gated_native.log`；成功时保留包含执行命令、33 项结果、`NATIVE CHECKS PASSED` 与 exit code 的截图。出现失败也保留 ZIP，不手动修改 ABI 检查以继续运行。

## 检查内容

1. 复查候选包版本、真实 gfx906/wave64、修复后 driver.c 哈希，以及 Torch/Triton 实际使用同一个 HIP 库。
2. 只编译宿主 C++ 启动库，动态加载当前已验证 HIP 库；无 CUDA 依赖。
3. 保留九齿生成的计算函数不变，通过显式 `ASTSource` 编译 20 个动态参数（4 个指针 + 16 个 int64）。不设置 constexpr 尺寸、不添加 stride 对齐假设、关闭 FP 融合。保存源码、编译请求、IR、HSACO 和 metadata。
4. 只接受 20 个显式参数 + 2 个 scratch 指针的实际 AMDGPU 参数表。要求 scratch/shared memory 均为零、wave64、256 线程；否则停止。原始 JIT 会专门化/删除常量参数，其 HSACO 不能直接套用本参数表。
5. C++ 用 `hipModuleLoad / hipModuleGetFunction / hipModuleLaunchKernel` 启动，接收原生 stream 和真实 shape/stride；记录真实成功调用次数。33 次计算都应发生在显式非默认 stream。
6. 复用前轮 30 项测试：FP32/FP16/BF16、6H gate 跨度和 2H/5H 偏移、零 stride、非整块尺寸、非连续输入输出、输入/保护区检查、连续复用。同一个 dtype 只导出和加载一次通用内核。
7. 额外三项原地 residual 更新。每次 C++ 输出同时与九齿 JIT 输出和原 CPU 参考比较；要求有限、逐元素相等，保留 NPZ。

## 来源与边界

- `gated_residual.py`、`common.py` 是 h3-ninetoothed-v2 已验证文件的原样副本。
- `jit_checks.py` 来自 v2 的 `verify.py`；仅为 `run_gated` 增加显式 `launch_factory` 参数并在原调用处使用该参数，测试尺寸、输入、舍入、CPU 参考和容差未变。
- `native.py` 是九齿生成函数的 HIP 导出适配，不宣称 NineToothed 内置 CUDA AOT 已适配海光。没有手写 Triton 计算函数。
- 这是单设备、正尺寸 2D 张量的独立验证包。共享/全局 scratch、graph capture、多线程并发、多卡、空张量、任意重叠视图均不在本轮支持范围。允许输出与 residual 完全相同的视图；实际集成还需实现公共接口约束。
- 成功只证明独立宿主启动和本轮 33 个数值用例。**InfiniCore 分发、H3 两处调用替换、模型回归与性能仍待验证**；不能作为完整初赛验收或性能提升证据。
- 本地 Windows 无海光设备；Python/Bash 语法、ABI 拒绝路径可在本地检查，真正 C++ 编译与设备执行以服务器回传为准。

失败会保存中间证据并返回非零；父进程检查子进程退出码、run_id、33 个完成记录和 C++ 计数，崩溃/超时/不完整结果不会判为通过。
