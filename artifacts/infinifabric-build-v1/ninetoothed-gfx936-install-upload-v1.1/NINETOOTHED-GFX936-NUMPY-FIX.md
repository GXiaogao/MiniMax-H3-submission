# 九齿安装 v1.1：补齐私有 NumPy

运行 `20261106T094117Z-1933459` 已完成九齿/SymPy/mpmath/海光Triton的解包，但在NumPy版本门槛失败，尚未进入真实目标识别和九齿JIT。原日志未打印NumPy实际版本，因此不能断言具体版本值。开头rocm_smi警告不是本次退出原因。

v1.1固定在九齿独立候选目录中安装NumPy1.26.4，保留共享系统NumPy与Torch。候选进程校验私有NumPy版本和来源，并要求Torch/HIP身份保持一致；主进程仍在前后独立核对系统Torch和系统NumPy。

## 一、补传一个 wheel

Windows本地解压 `ninetoothed-gfx936-numpy-input.zip`，将其中唯一 `.whl` 上传到原个人AFS目录：

```
minimax-h3/packages/ninetoothed-gfx936-inputs
```

文件名：

```
numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

原有 `ninetoothed-0.26.0`、`sympy-1.14.0`、`mpmath-1.3.0` 三个wheel继续保留。原海光Triton文件继续复用，不需要重新上传。

## 二、替换两个 Python 文件

解压 `ninetoothed-gfx936-install-upload-v1.1.zip`，在项目中更新：

- `install_ninetoothed_gfx936.py`
- `ninetoothed_gfx936_probe.py`

包内也带有原 `.sh` 和解包辅助文件，方便核对；这两个文件内容未修改，可沿用。不要只改其中一个Python文件。

## 三、仍使用原入口运行

- 入口：**`install_ninetoothed_gfx936.sh`**。
- 环境：沿用 `scnet-dcu` 八卡批处理。
- 程序参数：**留空**。
- 时长：**28分钟**。

脚本创建新编号的独立候选目录，不需要删除上次失败目录，也不覆盖正式推理环境。日志中会打印系统及候选的实际包版本和路径。

完成后返回本次日志目录中的 `ninetoothed-install.json`、`ninetoothed-probe.json` 和完整运行日志。只有实际gfx936检查及9组九齿JIT用例通过、原系统身份检查通过，才发布成功状态；当前修复仅通过本地测试，真实设备结果仍待运行。
