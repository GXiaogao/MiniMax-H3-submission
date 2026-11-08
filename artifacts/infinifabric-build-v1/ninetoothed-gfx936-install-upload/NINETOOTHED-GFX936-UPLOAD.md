# 当前八卡平台：九齿离线安装和 JIT 验证

本轮安装独立的 NineToothed0.26.0、SymPy1.14.0、mpmath1.3.0，复用已上传的海光 Triton3.5.1+das.opt1.dtk2604.torch290。依赖声明满足版本下限不等于实际后端兼容；必须以真实gfx936上九齿生成代码的运行结果为准。

系统Torch与NumPy、正式InfiniCore/InfiniLM扩展和启动配置保持原样。此步骤验证安装与独立JIT，不会自动启用模型中的九齿算子，也不替代当前Attention数值问题定位。

## 一、把三个 wheel 上传到个人 AFS

在Windows解压 `ninetoothed-gfx936-inputs.zip`。在平台“个人存储”创建并进入：

```
minimax-h3/packages/ninetoothed-gfx936-inputs
```

上传以下三个文件（校验清单可以一并上传）：

- `ninetoothed-0.26.0-py3-none-any.whl`
- `sympy-1.14.0-py3-none-any.whl`
- `mpmath-1.3.0-py3-none-any.whl`

已有完全相同文件时核对校验值即可。保留原有 `packages/minimax-h3-hygon-runtime-inputs` 中的海光Triton wheel；本轮无需重新上传它。不要把上述依赖安装到共享解释器。

## 二、把四个脚本上传到项目

在Windows解压 `ninetoothed-gfx936-install-upload.zip`，将四个运行文件上传到平台同一项目目录：

- `install_ninetoothed_gfx936.sh`
- `install_ninetoothed_gfx936.py`
- `ninetoothed_gfx936_probe.py`
- `install_infinilm_private_runtime.py`

最后一个文件复用已验证的安全wheel解包函数。已有同名文件请与本包校验清单核对。

## 三、批作业设置

- 入口：**`install_ninetoothed_gfx936.sh`**。
- 环境：沿用已验证的 `scnet-dcu` 八卡整节点环境。
- 程序参数：**留空**。
- 时长：**28分钟**。本轮仅使用GPU0做工具链检查，平台仍分配整节点。

脚本把纯Python包放入 `$WORK_HOME/python-envs/ninetoothed-gfx936/<本次编号>/site-packages`，海光Triton和编译缓存放在本次工作区。九齿原本固定写主目录的缓存行只在该独立副本中调整，记录改动前后哈希；不改HOME或伪造GPU架构。

之后依次核对包来源、Torch/NumPy身份、真实gfx936目标，并执行九齿生成的9组加法GPU用例（FP32/FP16/BF16，各含1元素、513元素、非连续513元素），检查输出、输入保护和输出保护区。

## 四、回传结果

本次日志目录：`$WORK_HOME/logs/infinifabric-install-ninetoothed-<本次编号>/`。

请返回：

1. `ninetoothed-install.json`
2. `ninetoothed-probe.json`（若环境预检失败，可能尚未生成）
3. 完整运行日志。

只有9组全部通过且原系统Torch/NumPy身份保持一致后，才发布 `$WORK_HOME/state/ninetoothed-gfx936.json`。完成信息为 `NINETOOTHED INSTALL/JIT VERIFIED; InfiniCore integration=NOT_TESTED`。

若出现导入、实际目标识别或JIT编译错误，返回上述报告与日志。安装候选文件仍保留在独立目录；失败不会切换正式推理环境。本脚本不自动更换Triton、安装新Torch或使用其他算子替代失败的九齿算子。

## 安装后的界限

`import`成功：仅说明包可导入。

9项JIT通过：说明这套九齿/海光Triton在本次gfx936环境能生成并运行这批测试算子。

InfiniCore实际调用、真实H3形状/精度/多卡回归、端到端视频质量与性能：仍需后续单独验证。后续进程需要按state记录加载该候选及匹配的job-local海光Triton；本轮不会全局激活它。
