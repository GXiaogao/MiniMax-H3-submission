# InfiniFabric InfiniCore 与 InfiniLM 分阶段构建包

本包按顺序完成 InfiniCore 原生库、InfiniCore Python 绑定和 InfiniLM C++ 扩展。每阶段都有独立成功标记；这些构建结果不代表 MiniMax-H3 八卡模型适配或端到端生成已经完成。

## 上传文件

向小型 runner 项目上传以下三个文件，保持它们位于项目根目录：

1. `build_infinicore_native.sh`
2. `inspect_hygon_arch.py`
3. `InfiniCore-infinifabric-v3.tar.gz`

`README.md` 只供本地查阅，无需上传。第三个文件是经过固定 SHA256 的 3.8 MiB 源码包，用于在作业本地快速展开；不要再上传展开后的源码目录、模型、完整工具链压缩包或 AFS 中的构建目录。

## 批处理配置

- 入口文件：`build_infinicore_native.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 资源检查：页面显示“当前可以分配：8 张卡”后再提交；平台会按整节点分配，无单独卡数输入框
- 建议首次时长：28 分钟（当前平台允许的最长时长）

脚本默认使用 16 个编译任务，以适应平台单次最长 28 分钟的限制。只有日志明确出现内存不足时，才在平台支持的环境变量位置降低 `INFINICORE_BUILD_JOBS`；不要把它作为程序参数。

版本 2.1 开始使用海光专用构建配置：`hygon-dcu=y`、`cpu=n`、`omp=n`，关闭会重复编译一整套源文件的独立 CPU target；当时海光 target 中无条件加入的 CPU descriptor 规则仍然保留。

版本 2.3 进一步依据 `cpu=n`，只在校验后的临时源码副本中删除 `xmake/hygon.lua` 里无条件加入整套 CPU descriptor 的单条规则。公共 operator 层、海光 CUDA 源码、ATen 和 CCL 均保留；个人 AFS 源码与上传的固定源码包不会被修改。

版本 2.4 在动态依赖校验前自动定位当前 Python 环境的 `torch/lib`，核对 `libtorch.so`、`libtorch_cpu.so`、`libc10.so` 和 `libc10_hip.so` 后加入 `LD_LIBRARY_PATH`，避免把已经成功安装的 ATen 构建误报为缺少 PyTorch 运行库。

版本 2.5 修复 `cpu=n` 海光专用配置下的上下文初始化：临时源码只有在运行时报告 CPU 数量大于 0 时才创建 `CPU:0`，CPU 回退也会先检查槽位；Hygon 设备的枚举、默认 runtime 和 8 卡路径保持不变。重新构建前，旧的原生成功标记会移动到本次日志目录，避免修复失败后被后续阶段误用。

版本 2.6 处理批处理作业 UID 会变化而 AFS 二进制仍受创建者权限约束的问题。脚本在安装前把旧 `$WORK_HOME/.infini` 原子移动到 `$WORK_HOME/backups/infinicore-native-rebuild-*/previous-infini`，创建全新的安装树，并在当前作业结束前给目录增加“其他用户只读和可遍历”、给文件增加“其他用户只读”权限。脚本不会开放其他用户写权限；旧树保留为可恢复备份。

若后续作业出现 `_infinicore` 或 `.infini/lib/*.so: Permission denied`，请按顺序执行：先用版本 2.6 的 `build_infinicore_native.sh` 重建原生树；确认日志通过后，运行版本 1.3 的 `build_infinicore_python.sh`；最后运行当前版本的 `build_infinilm_extension.sh`。不要只重复运行第三阶段脚本。

## 2.3 已完成编译后的快速恢复

若日志已经同时出现 `[100%]: build ok`、四次 `install ok!`，但最后只因下列四项 `ldd` 结果失败：

```text
libtorch.so => not found
libtorch_cpu.so => not found
libc10.so => not found
libc10_hip.so => not found
```

不要重新编译。只上传 `validate_infinicore_native.sh`，按以下方式新建一次短批处理：

- 入口文件：`validate_infinicore_native.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：5 分钟；若页面最小值为 10 分钟则填 10 分钟

该脚本不需要源码包、架构探测脚本或 xmake 缓存，不执行任何 `xmake build`/`xmake install`。它复用 `$WORK_HOME/.infini` 中的安装结果，补齐 PyTorch 运行库路径，完成 `file`、`ldd`、SHA256 和成功标记。

快速恢复通过时，日志末尾必须同时出现：

```text
InfiniCore native validation: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

## 第二阶段：InfiniCore Python 绑定

原生校验通过后运行 `build_infinicore_python.sh`。该脚本只构建当前 Python 3.11 的 `_infinicore` 扩展，直接链接第一阶段已经验证的 `$WORK_HOME/.infini` 原生库，不重复编译整套 InfiniCore。

如果 runner 项目中仍保留第一阶段文件，只需新增上传 `build_infinicore_python.sh`。运行时需要项目根目录中同时存在：

```text
build_infinicore_python.sh
inspect_hygon_arch.py
InfiniCore-infinifabric-v3.tar.gz
```

批处理配置：

- 入口文件：`build_infinicore_python.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 建议时长：15 分钟；脚本完成后会主动退出

个人 AFS 还必须保留：

```text
$WORK_HOME/state/infinicore-native.success
$WORK_HOME/packages/xmake-repo-cache.tar.gz
$WORK_HOME/packages/pybind11-v3.1.0.zip
$WORK_HOME/.infini/lib/libinfinicore_cpp_api.so
```

通过时日志末尾必须同时出现：

```text
device_count: 8
InfiniCore Python binding: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

成功后扩展位于 `$WORK_HOME/Projects/InfiniCore/python/infinicore/lib/_infinicore*.so`，并写入 `$WORK_HOME/state/infinicore-python.success`。该阶段不调用 `pip install`，也不修改平台共享 Python；后续 InfiniLM 脚本会通过固定 `PYTHONPATH` 使用此包。

版本 1.1 针对 InfiniFabric 上 xmake 未接受 `pkg_searchdirs` 中 pybind11 压缩包的问题，先核对 `pybind11-v3.1.0.zip` 的固定 SHA256，再把其中的头文件直接展开到作业临时目录。临时 xmake 工程删除 pybind11 包下载声明并改用该头文件目录，因此不会再回退到 GitHub 下载或 Git 克隆。

版本 1.2 根据当前作业实际使用的 Python 解释器，通过 `sysconfig` 获取开发头文件目录，先检查 `Python.h`，再把该目录显式加入临时 `_infinicore` 目标。它修复了版本 1.1 已进入编译后因 xmake 的旧 `python.library` 规则未加入 Python 3.11 头文件目录而失败的问题。

版本 1.3 在构建开始时只验证第一阶段四个原生库对当前作业 UID 可读，不尝试修改上一个作业拥有的文件。新 `_infinicore` 在复制到 AFS 前由创建它的作业增加“其他用户只读”权限；不可读的旧扩展会跳过备份并由已校验的新扩展原子替换。

若版本 1.2 日志已经显示 `_infinicore` 构建、`file` 和 `ldd` 成功，扩展已经写入个人 AFS，但随后只因旧原生上下文在 `import-smoke` 报 `Device Type Not Supported`，且原生 2.5 修复已经通过，则不必再次编译绑定。只上传并运行 `validate_infinicore_python.sh`：

- 入口文件：`validate_infinicore_python.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：5 分钟；若平台最小值为 10 分钟则填 10 分钟

该脚本不运行构建或安装命令。它检查现有扩展的动态依赖，按规范包路径导入 `infinicore`，确认当前原生设备是 Hygon、Hygon 数量为 8、PyTorch 可见 8 卡，最后写入 `$WORK_HOME/state/infinicore-python.success`。

快速验证通过时日志末尾必须同时出现：

```text
hygon_count: 8
torch_device_count: 8
InfiniCore Python validation: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

## 第三阶段：InfiniLM C++ 扩展

InfiniCore Python 快速验证通过后，推荐把固定源码包上传到个人 AFS：

```text
$WORK_HOME/packages/InfiniLM-infinifabric-v3.tar.gz
```

runner 项目中只上传 `build_infinilm_extension.sh`。脚本优先使用项目根目录中的同名源码包；项目中不存在时，自动使用上述 AFS 文件。这样项目输入约为 14 KiB，可避开 `WORKSPACE_QUOTA_EXCEEDED`。如果项目额度正常，也可以继续把脚本和源码包同时放在项目根目录。

第二个文件为 10.5 MiB 固定源码包。脚本先核对源码包和 `$WORK_HOME/packages/pybind11-v3.1.0.zip` 的 SHA256，将源码、离线 pybind11 头文件和 xmake 状态展开到作业临时目录，再使用与 Torch/InfiniCore 相同的 C++11 ABI 构建官方 `_infinilm` 目标。它不会执行 `pip install`，也不会修改共享 Python。

批处理配置：

- 入口文件：`build_infinilm_extension.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：28 分钟

脚本默认使用 16 个并行编译任务，完整编译 InfiniLM 目标中声明的 C++ 源码。构建后先执行 `file`、`ldd` 和直接扩展导入检查，确认 `_infinilm.InferEngine` 与 `_infinilm.DistConfig` 存在，再原子写入：

```text
$WORK_HOME/state/infinilm-extension.success
```

版本 1.2 针对不同批处理作业 UID 变化的问题：构建开始时只检查 `_infinicore` 和四个 InfiniCore 原生库已由各自创建作业发布为可读，不再对旧作业拥有的文件执行 `chmod`。新 `_infinilm` 在复制到 AFS 前由当前作业增加“其他用户只读”权限。旧 `_infinilm` 若不可读，脚本会跳过备份并用重新构建、已校验的新产物原子替换。

版本 1.3 增加个人 AFS 源码包回退。项目内没有 `InfiniLM-infinifabric-v3.tar.gz` 时，从 `$WORK_HOME/packages` 读取并执行相同 SHA256 校验；不会接受未经校验的其他文件。

版本 1.5 覆盖海光专用 InfiniCore 关闭 CPU runtime 后的完整 FL2VA 原生调用链。除 Qwen3-VL/Qwen3.5 的 `input_offsets`、`image_bound`、`image_grid_thw` 与 rotary position 元数据外，还修复 InferEngine 最大长度计算、Static Attention 序列长度读取及 `skip_sampling` 空输出对 CPU runtime 的隐式依赖。小型标量和元数据改为显式同步 D2H/H2D，`skip_sampling` 输出分配在当前 rank 的设备上。补丁只应用于校验后的作业临时源码；固定源码包、个人 AFS 源码和 InfiniCore 原生库均不会被修改。

通过时日志末尾必须同时出现：

```text
InferEngine: OK
DistConfig: OK
InfiniLM extension build: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

该阶段只验证 C++ 扩展，不执行完整 `import infinilm`。MiniMax-H3 入口还需要 diffusers、accelerate、视频与图像处理等 Python 包；这些依赖将在下一阶段按服务器缺失项使用离线 wheel 处理，不能把扩展构建成功写成模型适配完成。

## 第四阶段：InfiniLM Python 运行时审计

扩展构建通过后，向 runner 项目上传以下两个小文件：

1. `audit_infinilm_runtime.sh`
2. `audit_infinilm_runtime.py`

批处理配置：

- 入口文件：`audit_infinilm_runtime.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：10 分钟

该阶段只读取现有环境，不安装依赖、不修改共享 Python。它会扫描 InfiniLM 与 `examples/minimax_h3.py` 的导入，记录服务器已装版本，并在独立子进程中依次验证 `infinicore`、`infinilm`、MiniMax-H3 适配类、Diffusers、Accelerate、Safetensors 和 Qwen3-VL 的导入接口。

网页作业退出码为 0 表示诊断材料已完整保存，不一定表示运行环境已经齐备。请以日志末尾的 `runtime_ready` 为准：

```text
DIAGNOSTIC COLLECTED; runtime_ready=0; helper_exit_code=3
```

表示审计正常结束并发现缺失或不兼容依赖；这是本阶段可预期的结果。请下载日志，尤其提供 `canonical import traceback`、所有 `distribution=...` 行和 `missing_import=...` 行，再据此生成最小离线安装包。

只有出现以下结果，脚本才会写入 `$WORK_HOME/state/infinilm-runtime-import.success`：

```text
canonical_import=PASSED
runtime_ready=true
DIAGNOSTIC COLLECTED; runtime_ready=1; helper_exit_code=0
```

不要在收到审计结果前上传并安装整套旧 wheel，也不要覆盖平台自带的 Torch 或共享解释器。

### 全量缺失配置检测

若基础审计只停在第一个 `ModuleNotFoundError`（例如 `janus`），上传以下三个小文件进行全量只读检测：

1. `diagnose_infinilm_full_runtime.sh`
2. `diagnose_infinilm_full_runtime.py`
3. `infinilm-local-wheel-inventory.json`

批处理配置：

- 入口文件：`diagnose_infinilm_full_runtime.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：10 分钟

清单只保存用户本地 `wheels` 目录中 38 个 wheel 的文件名、版本、依赖、平台标签、大小和 SHA256，不包含 wheel 本体。检测脚本逐项验证 InfiniCore、InfiniLM、MiniMax-H3 适配层、Accelerate、Diffusers MiniMax-H3 接口、Qwen3-VL、Janus、视频和图像运行时，并把 Mooncake 单列为条件依赖。一个接口失败不会阻断其他接口检测。

脚本还会按服务器当前 Python 平台标签判断本地 wheel 是否兼容，解析 `Requires-Dist`，并标出对 `torch`、`transformers`、`tokenizers`、`huggingface-hub`、`safetensors` 和 `numpy` 的潜在覆盖或版本冲突。它不会运行 `pip install`、不会修改共享 Python，也不会复制 wheel。

网页退出码为 0 只表示诊断收集完成。以日志中的以下字段判断环境状态：

```text
diagnostic_complete=true
runtime_ready=false
DIAGNOSTIC COLLECTED; runtime_ready=false; helper_exit_code=0
```

结果保存在 `$WORK_HOME/logs/infinifabric-full-runtime-diagnostic-*/`：

```text
full-runtime-diagnostic.json
missing-runtime.txt
probe-*.stdout.log
probe-*.stderr.log
summary.txt
```

当前本地 wheel 清单包含 `janus-2.0.0` 以及多数已知缺失项，但没有 `mooncake-transfer-engine`。同时，`diffusers-0.41.0` 要求更高版本的 `huggingface-hub` 和 `safetensors`；不得据此直接安装整个 wheel 目录，因为其中还包含 Transformers 5、通用 Triton、OpenCV 5 等可能覆盖平台既有运行栈的包。先运行检测，再按 `protected_stack_conflicts` 生成最小离线安装集合。

## 第五阶段：个人 AFS 私有运行时安装

全量检测确认 InfiniCore 和8卡正常，并将30个固定wheel上传到以下目录后：

```text
$WORK_HOME/packages/infinilm-runtime-wheels
```

向runner项目上传两个小文件：

1. `install_infinilm_private_runtime.sh`
2. `install_infinilm_private_runtime.py`

批处理配置：

- 入口文件：`install_infinilm_private_runtime.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：10分钟

安装器核对30个wheel的固定SHA256，按wheel标准安全解包到作业本地暂存目录。私有目录固定使用已配套的 `transformers-5.20.2`、`tokenizers-0.23.3`、`huggingface-hub-1.32.0` 和 `safetensors-0.8.0`，解决 `diffusers-0.41.0` 与平台旧版 Hugging Face 栈不兼容的问题。它禁止路径穿越、符号链接、文件冲突以及私有目标中出现 `torch` 和 `numpy`，不会修改平台共享解释器或设备适配版本。

作业本地暂存安装完成后，脚本验证8卡、InfiniCore、InfiniLM、Janus、MiniMax-H3适配类、Accelerate、Diffusers MiniMax-H3接口、Qwen3-VL、视频与图像依赖。版本1.6不再把约7000个展开后的小文件复制到AFS；固定wheel继续作为AFS中的持久来源，后续作业在本地工作区快速展开。验证全部通过后，只原子写入两个小状态文件：

```text
$WORK_HOME/state/infinilm-runtime-private.path
$WORK_HOME/state/infinilm-runtime-private.json
```

`.path` 指向已校验的wheel目录，JSON记录安装器版本、运行模式、30个wheel的文件名与SHA256、预期发行包版本和验证时间。这样避开AFS小文件复制瓶颈，也不会占用项目输入快照额度。

通过标志：

```text
RESULT: PASSED; exit_code=0; last_stage=complete
```

若平台现有的间接依赖不满足wheel要求，脚本会在 `local-runtime-validation` 停止，旧状态标记不会更新。此时提供日志末尾的 `ERROR: stage ...` 和 `validate-local.stderr.log`，再补充最小依赖；不要直接上传或安装整个旧wheel目录。

版本1.1在离线pip安装失败时把标准输出和错误输出末尾直接写入主运行日志，同时仍保留完整的 `pip-install.stdout.log` 与 `pip-install.stderr.log`，用于区分pip参数、wheel平台兼容性和本地目标写入问题。

版本1.2处理平台禁止读取 `/etc/debian_version` 的限制。该限制会让pip在生成User-Agent时提前崩溃，即使已经使用 `--no-index` 也无法进入wheel解包。新版本不启动pip，直接验证并安全解包固定wheel。

版本1.6根据实测日志移除AFS展开树发布：30个wheel校验约2秒、7002个文件本地解包约10秒、完整导入验证约55秒，而AFS `persistent-copy` 持续数分钟仍未完成。后续推理作业必须从状态文件所指wheel目录展开到各自的 `/dev/shm` 工作区，再把该本地目录加入 `PYTHONPATH`。

## 第六阶段：正式模型资产预检

私有运行时出现以下结尾后，再执行模型资产预检：

```text
RESULT: PASSED; exit_code=0; last_stage=complete
```

向runner项目上传两个新文件，并保留同目录的 `install_infinilm_private_runtime.py`：

1. `preflight_minimax_h3_assets.sh`
2. `preflight_minimax_h3_assets.py`

第一次运行不传参数：

- 入口文件：`preflight_minimax_h3_assets.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：10分钟

脚本读取 `$WORK_HOME/state/infinilm-runtime-private.json`，把30个wheel展开到本次作业的临时工作区，重新验证8卡和完整Python接口，然后只读检查个人AFS及 `/public/home/aclsylqidf/shared` 中的MiniMax-H3模型。它核对 `diffusers/model_index.json`、FL2VA的 `text_encoder`/`transformer` 权重、权重索引引用和可选Ref2VA分区；不会加载正式权重、联网下载或启动视频生成。

如果管理员给出了明确模型路径，可在程序参数中传入：

```text
--model-path /public/home/aclsylqidf/shared/<实际MiniMax-H3目录>
```

模型已经就绪时应看到：

```text
PREFLIGHT COMPLETE; runtime_ready=true; model_ready=true
```

未找到模型时，脚本仍以退出码0完成诊断并输出：

```text
PREFLIGHT COMPLETE; runtime_ready=true; model_ready=false
```

这表示运行时已通过而正式权重尚未定位。个人AFS当前额度有限，不应在未确认模型大小和共享目录之前直接上传完整权重；先提供该预检日志，再决定使用平台共享模型还是申请额外持久容量。

预检版本1.2对共享目录的目录存在性、文件类型探测和模型内容读取分别做权限容错。其他用户的私有共享模型即使能被枚举但不能读取，也只会被跳过，不再导致整个预检失败。

## 第七阶段：海光正式模型运行时前置审计

模型资产预检出现 `runtime_ready=true; model_ready=true` 后，先上传并运行：

1. `audit_minimax_h3_hygon_runtime.sh`
2. `audit_minimax_h3_hygon_runtime.py`
3. `install_infinilm_private_runtime.py`

运行配置：

- 入口文件：`audit_minimax_h3_hygon_runtime.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：10分钟

该阶段根据八卡 POD 部署记录补齐此前预检没有覆盖的环境条件：海光 PyTorch 兼容的 `torchvision`、`torchvision::nms`、Qwen3-VL processor、`einops`、Triton、FlashAttention、`FA_SO_PATH`，以及一个小规模 BF16 flash-only SDPA 与 math SDPA 的数值对照。脚本把私有 InfiniLM 运行时和海光增量依赖分别展开到两个作业临时目录，再通过 `PYTHONPATH` 叠加使用，避免第二组 wheel 被错误安装到非空目录。脚本不加载正式模型权重、不联网，也不安装软件包。

全部条件满足时应看到：

```text
HYGON_RUNTIME_AUDIT COMPLETE; native_load_prerequisites_ready=true
```

如果结尾为 `false`，请提供同一日志中的所有 `missing_prerequisite=` 行和 `AUDIT_JSON`。不要直接照搬旧 root POD 文档中的 `pip install` 或系统级符号链接命令；当前平台是普通用户作业环境，必须先根据实际 Torch、DTK 和 Python ABI 准备离线依赖。

实测当前环境缺少 `torchvision`、`einops`、厂商 Triton 和 FlashAttention。Windows 本地输入已经固定为：

- `einops-0.8.2-py3-none-any.whl`
- `pytest-9.0.3-py3-none-any.whl`
- `iniconfig-2.0.0-py3-none-any.whl`
- `packaging-24.2-py3-none-any.whl`
- `pluggy-1.5.0-py3-none-any.whl`
- `pygments-2.21.0-py3-none-any.whl`
- `triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`
- `vision-0.24.0.tar.gz`
- FlashAttention同ABI wheel的7个100 MiB以内分片，文件名以 `.part-000` 到 `.part-006` 结尾

在个人 AFS 的 `minimax-h3/packages` 下创建 `minimax-h3-hygon-runtime-inputs` 目录，把以上15个文件上传到该目录。厂商FlashAttention wheel在正式导入路径中还会无条件导入pytest，因此5个小型Python wheel属于运行时依赖。完整FlashAttention wheel为658.9 MB，超过网页128 MiB限制，不要上传完整文件；准备脚本会在本次作业临时目录重组并核对完整SHA256。

向项目上传并运行：

1. `prepare_minimax_h3_hygon_runtime.sh`
2. `prepare_minimax_h3_hygon_runtime.py`
3. `install_infinilm_private_runtime.py`

运行配置：入口 `prepare_minimax_h3_hygon_runtime.sh`、参数留空、批处理作业、28分钟。它从固定源码离线构建海光PyTorch兼容的torchvision 0.24.0，验证 `torchvision::nms`、`FA_SO_PATH` 与flash-only SDPA，再发布轻量状态文件。通过标志为：

```text
RESULT: PASSED; exit_code=0; last_stage=complete
```

准备成功后，用新版本的审计两个文件覆盖项目中的旧版本，并保留 `prepare_minimax_h3_hygon_runtime.py`，再运行一次审计。此时必须得到 `native_load_prerequisites_ready=true`。

## 第八阶段：正式原生权重分组件加载

模型资产预检出现 `runtime_ready=true; model_ready=true` 后，先只加载FL2VA的Qwen3-VL条件编码器。上传并保留以下文件：

1. `probe_minimax_h3_native_load.sh`
2. `probe_minimax_h3_native_load.py`
3. `install_infinilm_private_runtime.py`
4. `prepare_minimax_h3_hygon_runtime.py`

首次运行配置：

- 入口文件：`probe_minimax_h3_native_load.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：25分钟

默认参数等价于：

```text
--task fl2va --component encoder --gpus 0,1,2,3,4,5,6,7 --model-path /public/home/aclsylqidf/shared/MiniMax-H3
```

脚本在作业本地分别展开私有运行时和海光增量依赖，确认Torch与InfiniCore均看到8张卡，然后按正式入口相同的 `MiniMaxH3ConditionEncoder(..., tp_device_ids=[0..7])` 路径加载Qwen3-VL权重。当前InfiniCore采用不含独立CPU后端的海光精简构建；探针只在检测到原生CPU设备数为0时，把safetensors读出的CPU权重逐个暂存到第一张指定DCU，再交给原有InfiniCore权重加载路径。它记录加载耗时、原生参数键数量及加载前后每张卡的可用显存，不执行forward或视频生成。

通过标志：

```text
NATIVE_LOAD_PROBE PASSED; task=fl2va; component=encoder; devices=0,1,2,3,4,5,6,7
```

编码器通过后，再用同一入口把程序参数改为 `--component transformer`，单独验证H3 Transformer正式权重。分开验证能在28分钟作业上限内区分权重映射、动态库、显存或具体组件问题；单组件加载成功仍不等于完整FL2VA生成通过。

## 第九阶段：原生 FL2VA 两步功能冒烟

编码器和 Transformer 的正式权重加载均通过后，使用官方 `Projects/InfiniLM/examples/minimax_h3.py fl2va` 入口执行第一次原生前向和媒体输出验证。上传并保留以下文件：

1. `run_minimax_h3_fl2va_native_smoke.sh`
2. `run_minimax_h3_fl2va_native_smoke.py`
3. `install_infinilm_private_runtime.py`
4. `prepare_minimax_h3_hygon_runtime.py`

第一帧必须已经上传到个人 AFS：

```text
$WORK_HOME/inputs/fl2va/Golde33443.jpg
```

运行配置：

- 入口文件：`run_minimax_h3_fl2va_native_smoke.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 资源：整节点 8 张卡
- 页面运行时长：28 分钟；脚本内部限制为 25 分钟，给日志和资源回收保留时间

脚本固定调用原生 InfiniLM 后端，不传 `--legacy`。本次功能冒烟使用正式模型、8张卡、124帧、384×576、seed 42 和 `steps=2`，提示词为 `A dog falls asleep`。长时间没有模型输出时，脚本每30秒打印当前阶段的 heartbeat，不能仅凭短暂无新日志判断卡死。

版本1.3保留原生崩溃定位证据，并在加载模型前读取 `$WORK_HOME/state/infinilm-extension.success`。只有构建版本严格等于 `1.5-hygon-fl2va-no-cpu-runtime`、标记中的扩展文件存在时才继续，同时把扩展 SHA256 写入日志。旧扩展会立即失败并提示先重建，避免再次耗时加载模型后才段错误。

通过时日志末尾必须出现：

```text
NATIVE_FL2VA_SMOKE PASSED; task=fl2va; backend=infinilm; steps=2; frames=124; output=...
```

成功标志只会在官方入口正常返回，并且输出文件通过 576×384、124个可解码视频帧和音频流检查后打印。视频保存在：

```text
$WORK_HOME/outputs/fl2va-native-smoke/
```

完整日志、摘要与结构化结果保存在：

```text
$WORK_HOME/logs/infinifabric-run-minimax-h3-native-fl2va-smoke-*/
```

`steps=2` 结果只证明原生 FL2VA 功能链路能够前向、解码并封装媒体，不代表正式质量达标或初赛整体验收完成。两步冒烟通过后，下一阶段才把相同入口的步数恢复为 `steps=10`，执行正式质量与性能验证。

## 第十阶段：原生 FL2VA 十步正式验证

两步冒烟出现 `NATIVE_FL2VA_SMOKE PASSED` 后，上传并替换为以下两个当前版本文件：

1. `run_minimax_h3_fl2va_native_formal.sh`
2. `run_minimax_h3_fl2va_native_smoke.py`

`install_infinilm_private_runtime.py` 和 `prepare_minimax_h3_hygon_runtime.py` 继续使用第九阶段已经上传并通过的版本。正式入口与冒烟使用同一 Python helper，但通过 `--run-profile=formal` 生成独立证据，不修改冒烟的默认两步行为。

运行配置：

- 入口文件：`run_minimax_h3_fl2va_native_formal.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 资源：整节点 8 张卡
- 页面运行时长：28 分钟；脚本内部限制为 25 分钟

脚本拒绝任何程序参数覆盖，固定使用正式模型、8 张卡、124 帧、384×576、seed 42、`steps=10`、提示词 `A dog falls asleep` 和 `$WORK_HOME/inputs/fl2va/Golde33443.jpg`。运行前仍核对 InfiniLM 扩展构建版本 `1.5-hygon-fl2va-no-cpu-runtime` 并记录扩展 SHA256。

通过时日志末尾必须出现：

```text
stage=media-validation status=DONE resolution=576x384 frames=124 audio_streams=1
NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED; task=fl2va; backend=infinilm; quality_status=UNASSESSED; steps=10; frames=124; output=...
```

视频保存在：

```text
$WORK_HOME/outputs/fl2va-native-formal/
```

完整日志、摘要和 `minimax-h3-native-fl2va-formal.json` 保存在：

```text
$WORK_HOME/logs/infinifabric-run-minimax-h3-native-fl2va-formal-*
```

该结果用于记录当前十步样例的媒体完整性和各阶段耗时。`STRUCTURAL_PASSED` 只表示媒体结构检查通过；JSON 中 `quality_status=UNASSESSED` 表示画面与相似度尚未验证。单次运行不能替代官方相似度或正式性能验收；缺少统一参考输出、比较脚本和统一计时流程时，不报告相似度通过或正式性能成绩。现有文件名中的 `formal` 沿用历史命名，不代表当前十步参数已经得到主办方评测确认。

## 第十一阶段：Native FL2VA 画面质量诊断

2026-11-03 的十步作业虽然以旧标志 `NATIVE_FL2VA_FORMAL PASSED` 结束，但导出的视频是彩色网格，不能计为正确性通过。其视频/音频 VAE 同时出现直接 `.to(dtype=...)` 破坏 float32 保留模块的警告。本阶段只验证这一原因；它尚未被确认是唯一根因。

本地当前目录需上传或替换三个文件到**同一个平台项目**：

1. `run_minimax_h3_fl2va_native_quality_probe.sh`（新增入口）
2. `prepare_minimax_h3_vae_dtype_probe.py`（新增补丁器）
3. `run_minimax_h3_fl2va_native_smoke.py`（替换为 `1.3-minimax-h3-media-structure-scope` 版本）

同一项目中还须保留已通过的 `install_infinilm_private_runtime.py` 和 `prepare_minimax_h3_hygon_runtime.py`。完整上传包 `minimax-h3-fl2va-quality-probe-upload.zip` 包含这五个小脚本、说明和校验清单；模型、wheel、源码库及首帧继续从 AFS/共享目录读取，不重复上传到项目输入区。

运行配置：

- 入口文件：`run_minimax_h3_fl2va_native_quality_probe.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 页面运行时长：28 分钟；生成调用内部限制 25 分钟
- 资源：整节点 8 张卡，继续选用此前成功运行的 `scnet-dcu` 环境

固定与失败视频相同的诊断配置：`A dog falls asleep`、`Golde33443.jpg`、8 卡、124 帧、`steps=10`、seed 42、576×384。正式评测工作量仍需以最终指定配置核对；本阶段不通过改变工作量推断修复或加速。

脚本不会修改 `$WORK_HOME/Projects/InfiniLM/examples/minimax_h3.py`。它在作业本地目录生成副本，仅移除 VAE 和 Audio-VAE 的两处 `dtype=DTYPE`，保留组件加载时设置的混合精度。源码语句缺失、重复或不再是可执行调用时立即失败；已有临时目标文件也不会覆盖。

执行完成后应出现：

```text
probe_version=1.0-native-fl2va-preserve-vae-fp32
stage=prepare-vae-dtype-probe status=DONE
NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED; task=fl2va; backend=infinilm; quality_status=UNASSESSED; ...
QUALITY_PROBE RUN_COMPLETE; quality_status=UNASSESSED; inspect output frames before judging correctness.
```

视频保存到 `$WORK_HOME/outputs/fl2va-native-quality-probe/`。证据保存到 `$WORK_HOME/logs/infinifabric-run-minimax-h3-native-fl2va-quality-probe-*`：

- `run.log`：完整阶段日志及 InfiniLM 扩展 SHA256。
- `summary.txt`：退出码、最后阶段、输出和证据位置；预检或补丁失败也会写入。
- `vae-dtype-patch.json`、`vae-dtype-patch.log`：原始/副本 SHA256、补丁版本及错误。
- `minimax_h3.vae-probe.py`：实际运行入口的证据副本，保留到节点临时目录被回收之后。
- `minimax-h3-native-fl2va-formal.json`：沿用 helper 的结果文件名，但位于独立的 quality-probe 日志目录；结构成功为 `result=STRUCTURAL_PASSED`，画面状态为 `quality_status=UNASSESSED`。

运行后取回 **MP4 和完整 `run.log`**，同时保留上述 JSON/摘要。至少检查第 0、6、24、60、123 帧，确认能辨认狗与场景，是否仍为彩色块；视觉恢复也不等于已通过官方相似度检查。若恢复，再用统一参考输出及官方比较脚本回归；若仍是彩色块，保留本轮证据，下一步隔离 VAE encode/decode 与 Native Transformer 潜变量。当前无需重新编译 InfiniCore/InfiniLM，也不进入性能计分。

## 第十二阶段：独立 VAE 编解码重建

2026-11-04 精度试验已运行，相关警告消失，但五张输出抽帧仍为彩色噪声。下一项为独立 VAE 重建，说明见 `VAE-ROUNDTRIP-UPLOAD.md`，上传包为 `minimax-h3-vae-roundtrip-upload.zip`。

新增入口 `diagnose_minimax_h3_vae_roundtrip.sh` 与同名 Python helper，配合现有三个运行时 helper 使用。程序参数留空，沿用 scnet-dcu 批处理和 28 分钟时限。脚本仅在 `cuda:0` 加载现有视频 VAE，保存实际精度、权重加载信息及重复图片的编解码结果；不运行或替换 Qwen3-VL/Native Transformer，不作为正式生成或计分结果。

结果位于 `$WORK_HOME/logs/infinifabric-vae-roundtrip-*/`。取回 `vae-roundtrip-contact-sheet.jpg`、`vae-roundtrip.json` 与 `run.log`；如失败，取回已有报告和 `summary.txt`。`DIAGNOSTIC_COMPLETE` 不等于画面或官方相似度通过。

## 第十三阶段：实际 Native 生成链路观察

用户的独立 VAE 结果 `20261104T071638Z-871674` 已能恢复清晰狗图，22 帧平均 PSNR 为 27.5320 dB，未发现加载键差异或 FP32 参数违例。完整生成正确性仍未通过；此次不能替代实际单帧条件与去噪检查。

使用 `minimax-h3-native-chain-upload.zip`，解压后将 8 个脚本放到项目同一目录，入口 `diagnose_minimax_h3_native_chain.sh`，参数留空，scnet-dcu 批处理、28 分钟。详见 `NATIVE-CHAIN-UPLOAD.md`。固定原八卡诊断配置，观察实际首帧条件、Qwen 第 50 层、Native Transformer、调度更新与解码输入；不替换核心计算。

下载 `run.log`、`native-chain-summary.json`、`native-chain.jsonl`；保留条件和解码输入 NPZ。日志包含额外同步与存盘，不能据此计算加速收益；`DIAGNOSTIC_COMPLETE` 仅表示采集完成。

### 跨作业 `.so` 权限恢复

如果审计 traceback 在导入依赖之前出现以下错误：

```text
_infinicore.cpython-311-x86_64-linux-gnu.so: cannot open shared object file: Permission denied
```

这表示先前作业构建的 AFS 扩展没有保留跨作业读取权限。实测表明 AFS 拒绝 `chgrp`，并且即使文件显示 `nogroup + rwx`，补充组成员仍可能无法读取，因此不要继续运行 `repair_infinicore_runtime_permissions.sh`。

这类错误不能在后续作业中用 `chmod` 修复，因为作业 UID 已变化。必须从创建原生二进制的阶段重新发布，依次运行：

1. `build_infinicore_native.sh` 版本 2.6，28 分钟；
2. `build_infinicore_python.sh` 版本 1.3，15 分钟；
3. `build_infinilm_extension.sh` 版本 1.5，建议使用页面允许的 28 分钟；
4. `audit_infinilm_runtime.sh`，10 分钟。

每一步通过后再提交下一步。第三步的配置为：

- 入口文件：`build_infinilm_extension.sh`
- 程序参数：留空
- 使用方式：批处理作业
- 运行时长：28 分钟

第三步不会重编 InfiniCore 原生库；它要求前两步产物已经跨 UID 可读，并在写入 AFS 前固定新 `_infinilm` 的跨作业只读权限。通过时必须出现：

```text
build_version=1.5-hygon-fl2va-no-cpu-runtime
patch_profile=hygon-fl2va-no-cpu-runtime
InferEngine: OK
DistConfig: OK
InfiniLM extension build: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

修复通过后，再运行一次 `audit_infinilm_runtime.sh`，以获得排除权限错误后的真实 Python 依赖缺口。

## 已有前置条件

以下内容必须已经位于个人 AFS：

```text
$IFAB_AFS_HOME/minimax-h3/setup_infinifabric_env.sh
$IFAB_AFS_HOME/minimax-h3/tools/xmake/xmake
$IFAB_AFS_HOME/minimax-h3/packages/xmake-repo-cache.tar.gz
```

版本 2.2 不再从 AFS 逐文件复制 `Projects/InfiniCore` 或展开后的 xmake 仓库。源码包和仓库缓存会先核对，再直接解压到本次作业的本地工作区。

## 通过标志

运行日志末尾必须同时出现：

```text
hygon_arch=gfx936
InfiniCore native build: PASSED
RESULT: PASSED; exit_code=0; last_stage=complete
```

如果实际硬件报告 `gfx906`，第一行可以是 `hygon_arch=gfx906`。脚本不接受其他架构，也不允许八张卡架构不一致。

成功后会生成：

```text
$WORK_HOME/.infini/include/infinicore/tensor.hpp
$WORK_HOME/.infini/lib/libinfinicore_cpp_api.so
$WORK_HOME/state/infinicore-native.success
$WORK_HOME/logs/infinifabric-build-infinicore-native-*/run.log
$WORK_HOME/logs/infinifabric-build-infinicore-native-*/summary.txt
$WORK_HOME/logs/infinifabric-build-infinicore-native-*/hardware.json
$WORK_HOME/logs/infinifabric-build-infinicore-native-*/InfiniCore_xmake.conf
$WORK_HOME/logs/infinifabric-build-infinicore-native-*/artifacts.sha256
```

进入第二阶段前，请保留或下载最新目录中的 `run.log`、`summary.txt`、`hardware.json`、`InfiniCore_xmake.conf` 和 `artifacts.sha256`。

## 中断与重跑

脚本把可变 xmake 状态、源码副本和构建目录放在本次作业的临时工作区，避开个人 AFS 对 xmake 配置文件的 `EBUSY` 限制；日志和安装结果仍保存在个人 AFS。若作业因时间到期而终止，等待平台确认资源已释放，再用相同三个文件重新提交。新作业会重新构建，不复用上一次已释放节点的临时产物。

不要执行 `xmake clean`，不要删除 `$WORK_HOME/Projects/InfiniCore`、`$WORK_HOME/.xmake/repositories` 或 `$WORK_HOME/.infini`。成功标记只会在安装文件、动态依赖、配置快照和 SHA256 全部核对通过后原子更新。

若失败，请提供最新 `summary.txt` 中的 `last_stage`，以及同目录对应阶段日志。不要仅提供网页的退出码。

## 后续：Native H3 单次前向对照（2026-11-05）

Native 链路观察后的视频仍为彩色网格，条件 NPZ 完整且保留空间结构。下一轮使用 `diagnose_minimax_h3_transformer_parity.sh`，从原 Native 管线取得首次 H3 前向的输入/输出，再在独立进程用参考 H3 重放同一输入；正式路径仍为 Native。此诊断不输出新视频、不验证 Qwen 自身、不评估速度。

上传包 `minimax-h3-transformer-parity-upload.zip` 含六个脚本，完整步骤见 [TRANSFORMER-PARITY-UPLOAD.md](TRANSFORMER-PARITY-UPLOAD.md)。批处理、参数留空、28 分钟；取回新日志目录的 `transformer-parity.json` 和 `run.log`。本地脚本验证不能代替八卡数值结果。

## 本地校验记录

| 文件 | SHA256 |
|---|---|
| `tests/test_infinifabric_minimax_h3_vae_dtype_patch_contract.py` | `5F85B39D73AF45726880B9FCF7F4B7B41D559E8499DA532455439D07986D7AA6` |
| `tests/test_infinifabric_minimax_h3_native_fl2va_quality_probe_contract.py` | `B2A38B0C40834A1C2EEB51B4892109C99B9186F5872CFC11889243C13DD9FD60` |
| `run_minimax_h3_fl2va_native_quality_probe.sh` | `8D79D71D155608A38C1CB332FA58F82893CC1E942F76A66D269EBE37EF12BF21` |
| `prepare_minimax_h3_vae_dtype_probe.py` | `E0DE65E74C1A32D64D6DFEF6B6A94EA91E4E15A5F353EF1F081F78B0E56B2F53` |
| `inspect_hygon_arch.py` | `88490EF7C73F14F0480331DB87CABA709A27C09C7FFCB323D094D835F4B97F6F` |
| `build_infinicore_native.sh` | `74B34B930663EC189600B9566640EEACE30432136E9807BF87E7036FE5BBC3C9` |
| `validate_infinicore_native.sh` | `217A4407077B94AC148E56044DCCC25C27E7D35D59090B3BB577D19EBCE465DE` |
| `build_infinicore_python.sh` | `0995D5C6EDE1F4AEB2AA0BC2937F73F8EB696C16C0DFC5BC3FCA0D503343AD6A` |
| `validate_infinicore_python.sh` | `D8EB2285CD36251FE5AA487B2CC9B0DB52718CEAA970A2B2A9FFB381C9480E17` |
| `build_infinilm_extension.sh` | `78B75DA3A8D7A4A51981E67D816280FDFCA8D7A815AAF8EFE78F58235BC1486E` |
| `audit_infinilm_runtime.sh` | `1F72A1F44F4ADA0746D9DDB8C7DDAA3AA0584B358CA49E88E8DAEA91D354EFAB` |
| `audit_infinilm_runtime.py` | `55041488CC58BC2943F49620639CEA3955AF8DBE52385EC4E2B7DC0C6B5F064A` |
| `audit_minimax_h3_hygon_runtime.sh` | `2F1E71993F0A3454637C1FC8093D55A559F3A7FFC9D20303BF34907AF14F9D16` |
| `audit_minimax_h3_hygon_runtime.py` | `F5DAAB7848911E47C11C8FC8D98B1E9B47AA47700B75E823261020809AC67AE8` |
| `download_minimax_h3_hygon_runtime_inputs.ps1` | `D5685D4507F512A19934DBA322EC028CE35255FA67B045901030891EB40429F2` |
| `diagnose_infinilm_full_runtime.sh` | `A0C9EBDF064BE45F8A6E08A196FBDCC22F2D2781A0F3FB12EE08EFB3074E4BC0` |
| `diagnose_infinilm_full_runtime.py` | `B1FE9DE007A0B2F6EE84D94A4A2C7A55887637D11F4E3075B2138AE8E06BCAB8` |
| `infinilm-local-wheel-inventory.json` | `37282FD3F9040EB27C4E8C48AD52CDCE1975F52A9E4C74119A80DEAF043877CC` |
| `install_infinilm_private_runtime.sh` | `DEBD61867DAE2DD731E8280ECB15EA6DED6872B01ED89750663DC809E97750F5` |
| `install_infinilm_private_runtime.py` | `E7B811DBA6FD51D1F52558BB7B93D89817C4DFD8DB5DA4449A27D60AE84CFE18` |
| `preflight_minimax_h3_assets.sh` | `45C80C12A40F70B7FDAF257CE0EDBCBEC9329C6B9948BA46027923D9F10F12E6` |
| `preflight_minimax_h3_assets.py` | `B49A5DEB9300D8F8F879AF8F344EE5BC97885FDB670281378B97B95E0BA62E67` |
| `prepare_minimax_h3_hygon_runtime.sh` | `0D90D6522610BF576C5F13515918A371903DC7A06845A429E4455BA72DFC72AA` |
| `prepare_minimax_h3_hygon_runtime.py` | `1CAA354F1B63C1BEE4474453D64FA86EDE3AB71D2CAB1D1E8814A95B0D6D566D` |
| `probe_minimax_h3_native_load.sh` | `2A9EBAE9676E6979BD5C8EFC4D1BBBAB0F271174B6DF2AFA886ED13966AA5B2C` |
| `probe_minimax_h3_native_load.py` | `0CECEF5822AD06AA4896E6391A3A1DA4F04D245F22FEDCF1699C04487DE09B97` |
| `run_minimax_h3_fl2va_native_smoke.sh` | `04814C8A98F180FDC7BEC822347935AEAB43722DD5863B20938D7FABF0A19A32` |
| `run_minimax_h3_fl2va_native_smoke.py` | `DC3C685B565E0E0AF7D29E059764BB3AC9AC45388632D0686ED7F7DB3D998511` |
| `run_minimax_h3_fl2va_native_formal.sh` | `03445828591FC41C9A964D5C0B02088D8BAD2D5D22D5D0549A5DA365FF1F9D07` |
| `repair_infinicore_runtime_permissions.sh` | `85F0F9FD9D2D19EFCD5BCC51ACBADD37E974094D3380D63EAF06F11B56F259D5` |
| `split_flash_attention_wheel.ps1` | `E01EA3C27DA9AEF514B7A04999B198B327D9A1A73B37241F38439CAD403C6A9D` |
| `InfiniCore-infinifabric-v3.tar.gz` | `37414EDBC25545063462A3CBE47811CA0A672378058D78B761EC36E6FC3F8811` |
| `InfiniLM-infinifabric-v3.tar.gz` | `4C62340AA0AEF5FEC9E5A7F57EA7753EB6666E070B3A7E66AED5D722BAAD39E6` |
| `tests/test_infinifabric_hygon_arch.py` | `CEADD37C21F3F14776C07A6B3F6CD3154D11039DD3C290BF90A4872ABC166D38` |
| `tests/test_infinifabric_hygon_runtime_input_download_contract.py` | `1131743912BFCEA4F1C194E52C1EBC6022BCA74C74FDCF5A39FED150A5E395F8` |
| `tests/test_infinifabric_flash_wheel_split_contract.py` | `D7E7A3AD4C544FB47BF9DA06F52AB5BEEB0AF56D858755F626AB04258C1F4F05` |
| `tests/test_infinifabric_minimax_h3_hygon_runtime_prepare_contract.py` | `9F0788F8CB6853CD49FCFD93A481232A13709422042093740572B0D5FE3CE32B` |
| `tests/test_infinifabric_native_build_contract.py` | `EB9BF05B90B1FB2B57B526E85DFE7A9E4770606DF423C48B16639C8D83C1AE56` |
| `tests/test_infinifabric_native_validation_contract.py` | `CC0CC0165428FFE66A606C42E3C02D8677911A2BA7738F130DF1FF51BE47A529` |
| `tests/test_infinifabric_infinicore_python_build_contract.py` | `1A3E7E1252ED6952032E090E6DDCB791F753E3D19FA389F75ADCBFA6F43F6E49` |
| `tests/test_infinifabric_python_validation_contract.py` | `15B48BD4266116CBBE8F3B3B27EB15324F5D98C5B890B94D33FC81881BB6032A` |
| `tests/test_infinifabric_infinilm_extension_build_contract.py` | `B4AE30AC384D3824FA02CD3323DC5142DB507658A6E2BE738EAD4FD999F6FDCE` |
| `tests/test_infinifabric_infinilm_runtime_audit_contract.py` | `5EEACB383E38D550AE8AA2E427421B649BC4B70176F5F751623568A1B92E3B3A` |
| `tests/test_infinifabric_full_runtime_diagnostic_contract.py` | `F0DAB75E1193CC8A4DEA6648738B91E7912D20534D58DBBDC8B62760375D8353` |
| `tests/test_infinifabric_private_runtime_install_contract.py` | `953EE9A32C111DF59D4CD15B432AC3A791D3A24CE1C9935F7EB3A97D59ACEB58` |
| `tests/test_infinifabric_runtime_permission_repair_contract.py` | `E184C1FD2131BC614AFA0E5F91ACC45063D1F698C2B61C89CB5F5813857B697B` |
| `tests/test_infinifabric_minimax_h3_hygon_runtime_audit_contract.py` | `7FCC0E37C7E052E82C132D9F707DA89BD3614A65D3EE0F964A472FD5FA200C6E` |
| `tests/test_infinifabric_minimax_h3_native_fl2va_smoke_contract.py` | `C458FCFD9780E04B60DCCD0BAF4AA479376EFAC5F78027B4EBB93561FF56A679` |
| `tests/test_infinifabric_minimax_h3_native_fl2va_formal_contract.py` | `704A81EAD09BEC3FBC61C6699BE90D5257094BE0E482929DA6A415F47CFD1CEF` |

脚本使用 LF 换行。网页上传后不要用会将文件转换为 CRLF 的编辑器重新保存。

本地回归覆盖构建、离线依赖、跨作业权限、运行时准备、原生 FL2VA 入口与媒体结构判据。本轮新增实际 Bash 入口执行、固定参数、源码补丁、结果 JSON 语义及失败日志测试；这些测试不能替代 InfiniFabric 上的动态链接、模型前向与画面质量验证。

2026-11-04 本轮核验采用 Git Bash，已绕过此前 WSL `CreateInstance/E_ACCESSDENIED` 对本地检查的阻塞。具体测试数量、命令和未验证范围见 `docs/fl2va-quality-probe-validation-2026-11-04.md`。
