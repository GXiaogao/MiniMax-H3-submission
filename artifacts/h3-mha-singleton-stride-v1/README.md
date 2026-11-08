# 海光 MHA：兼容原生 unsqueeze 的 singleton 零 stride

日期：2026-10-25。状态：代码修复、部署验证已完成；C++ 编译及海光数值回归待服务器执行。

## 已取得的证据

Qwen3-VL 验证在 Pod mlhgp 的 FP32 模型中完成 68 项参数精确回读、6 项非法输入拒绝，以及 text_0（S=7）前向。纯文本输出最大绝对误差为 2.0116567611694336e-07，输入未改写。单图 metadata 已精确匹配，但 native_facade_forward 报：

```text
Hygon InfiniOp MHA: strides must be positive
RuntimeError: RankWorker stopped during run
```

后一句是工作线程因前面的异常退出的结果。FP16、BF16、双图与最后一次纯文本请求均未执行，不能标记通过。原生 JSON、粘贴日志及摘要已归档至 `logs/2026-10-25-ops/qwen3_vl_encoder-singleton-*`；报告中的三份脚本及八份 InfiniLM 文件哈希与本地匹配。

## 根因与修复范围

指定 InfiniCore `src/infinicore/tensor/view.cc` 的 `TensorImpl::unsqueeze` 插入 size=1、stride=0；指定 InfiniLM `qwen3_5_vision.cpp` 视觉注意力以 `q->unsqueeze(0)`、`k->unsqueeze(0)` 和同样的 v 调用 MHA。此前开发的海光 fallback 把所有零 stride 拒绝，因此错误发生在其布局检查，尚未执行该视觉 MHA 的 GEMM。

修复只调整 `mha_hygon_infiniop.cc`：非末维允许 size=1、stride=0；这些维度只有索引 0，不会移动地址，也不产生广播重叠。若序列维 size=1 且 stride=0，构造 GEMM 描述符时以 D 作为正 leading dimension，原始地址偏移不变。负 stride、非 singleton 零 stride、重叠、跨度溢出、非连续末维、输出输入别名继续拒绝。不改变 attention 数学、模型配置、参考输出或误差阈值，不增加复制/新内核。

这是基于已有 InfiniOp 后端的布局适配，不记为九齿贡献或性能优化。原始 `repos/` 不改；实际改动位于已有隔离 worktree。公共接口不变。

## 上传与执行

解压 `artifacts/h3-mha-singleton-stride-v1.zip`，将所有文件放入服务器：

```text
/root/private_data/scripts/h3-mha-singleton-stride-v1/
```

本包是旧 `h3-mha-infiniop-v1` 的增量升级；部署脚本校验指定 commit、旧后端内容及已有 dispatcher 哈希，遇到其他修改会停止。重复应用同一包不再改源码。不要在新修复后重新应用旧 v1 补丁包。

在已经初始化成功的终端依次执行，保留 `&&`，前一步失败即停止：

```bash
run_logged h3_mha_singleton_build.log bash "$WORK_HOME/scripts/h3-mha-singleton-stride-v1/build_h3_mha.sh" &&
run_logged h3_mha_singleton_regression.log python -X faulthandler -u "$WORK_HOME/scripts/h3-mha-singleton-stride-v1/verify_h3_mha_infiniop.py" --graph &&
run_logged qwen3_vl_encoder_retry2.log python -X faulthandler -u "$WORK_HOME/scripts/verify_qwen3_vl_encoder.py"
```

新终端仍先 source 原 setup_env.sh；无须修改它或重装 Python。构建保留现有 xmake 配置，仅 build/install `infinicore_cpp_api`，安装前保存旧共享库。InfiniLM 无需重编译。所有日志在 `$LOG_DIR`。

构建前强制运行 15 项无 GPU 的 C++ 布局检查：探针直接抽取待编译源文件中的校验和 sequence_stride 函数，以元数据替身验证 singleton、普通 packed/permute、非法广播/负 stride/重叠/溢出及 leading dimension；不把元数据检查当作 GPU 计算验证。

设备回归保留旧 52 项，新增 24 项 singleton 数值用例和 3 项非法广播检查，默认共 79 项。`--graph` 再执行原有三项与新增三项 singleton 图回放，共 85 项。新增用例通过 InfiniCore 原生 squeeze/unsqueeze 构造零 stride，包含 batch、head、sequence 和三轴同时 singleton，以及分配输出/显式 out、三种 dtype、两次改变输入。每项失败即停止。

全部成功时分别应见 `PASSED: 15 host layout checks (no GPU)`、MHA `PASSED: 85 cases`、Encoder `PASSED: image_text, 3 models, 12 forwards, 18 rejection checks`。这些是预期，不是已取得的服务器结果。回传失败步骤的完整日志和新 report；若全部通过，回传两个设备测试日志及 JSON。

## 本地验证和限制

56 项宿主测试通过，包含真实旧 v1 → 本包升级、未跟踪后端文件、重复应用及 dispatcher 被修改时拒绝覆盖。Python 语法、shell 语法、包清单/哈希和文件来源另行检查。

已完成实现者的差异核对。独立审查代理因用量限制未能完成，不记为独立审查通过；后续仍以服务器编译、MHA 回归和 Encoder 实测结果为准。

本地没有 C++ 编译器、Torch/HIP 或海光设备，不能完成原生编译和数值复测。服务器构建脚本包含 C++ 元数据检查和源码 syntax-only 编译。旧共享库 SHA256 `65ee955461917f4f572fcecc5a753d9c7188a80e08412bca07b3c73df37753ba` 是本次失败前版本；修复后应记录新库哈希，不能继续沿用旧哈希作为新成功证据。
