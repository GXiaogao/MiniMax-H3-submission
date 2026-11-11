# 公开组件仓库与固定版本

本页列出模型适配及独立九齿Attention实验的固定源码入口。四个仓库均为public，分支统一为minimax-h3-preliminary。实际修改版commit与原上游commit分开记录；不得用安装版本号或历史基线代替修改版commit。

| 组件 | 公开修改仓库 | 分支 | 完整修改commit |
|---|---|---|---|
| InfiniCore | [InfiniCore-MiniMax-H3](https://github.com/GXiaogao/InfiniCore-MiniMax-H3) | minimax-h3-preliminary | [`74ca95443a709b8c5ee1403a02965d25df91d025`](https://github.com/GXiaogao/InfiniCore-MiniMax-H3/commit/74ca95443a709b8c5ee1403a02965d25df91d025) |
| InfiniLM | [InfiniLM-MiniMax-H3](https://github.com/GXiaogao/InfiniLM-MiniMax-H3) | minimax-h3-preliminary | [`1a066cdb1085ff604aeb00bae35d547cc8414e84`](https://github.com/GXiaogao/InfiniLM-MiniMax-H3/commit/1a066cdb1085ff604aeb00bae35d547cc8414e84) |
| ninetoothed | [ninetoothed-MiniMax-H3](https://github.com/GXiaogao/ninetoothed-MiniMax-H3) | minimax-h3-preliminary | [`43507458767b537974b36f40313c3615b8ebc2ac`](https://github.com/GXiaogao/ninetoothed-MiniMax-H3/commit/43507458767b537974b36f40313c3615b8ebc2ac) |
| triton | [triton-MiniMax-H3](https://github.com/GXiaogao/triton-MiniMax-H3) | minimax-h3-preliminary | [`f5c20bf89d439edab2ce9841d2c5275967171806`](https://github.com/GXiaogao/triton-MiniMax-H3/commit/f5c20bf89d439edab2ce9841d2c5275967171806) |

## 固定代码与源身份检查

```bash
git clone https://github.com/GXiaogao/MiniMax-H3-submission.git
cd MiniMax-H3-submission
mkdir -p component-checkouts
git clone --branch minimax-h3-preliminary https://github.com/GXiaogao/InfiniCore-MiniMax-H3.git component-checkouts/InfiniCore
git -C component-checkouts/InfiniCore checkout 74ca95443a709b8c5ee1403a02965d25df91d025
git clone --branch minimax-h3-preliminary https://github.com/GXiaogao/InfiniLM-MiniMax-H3.git component-checkouts/InfiniLM
git -C component-checkouts/InfiniLM checkout 1a066cdb1085ff604aeb00bae35d547cc8414e84
git clone --branch minimax-h3-preliminary https://github.com/GXiaogao/ninetoothed-MiniMax-H3.git component-checkouts/ninetoothed
git -C component-checkouts/ninetoothed checkout 43507458767b537974b36f40313c3615b8ebc2ac
git clone --branch minimax-h3-preliminary https://github.com/GXiaogao/triton-MiniMax-H3.git component-checkouts/triton
git -C component-checkouts/triton checkout f5c20bf89d439edab2ce9841d2c5275967171806
python3 scripts/verify_public_sources.py --checkouts component-checkouts
```

验证仅比较Git身份和源码SHA，不执行GPU测试。各组件保留上游许可证和子模块Git链接。Core/LM的competition/source-provenance.json记录固定源码归档与还原边界；LM选定源码及辅助头文件与已有服务器trace匹配。

## 从公开源码重建候选

在有既定DTK、厂商Torch/Python3.11、离线依赖和个人AFS的gfx936环境运行。Core先初始化third_party/spdlog；LM初始化third_party/spdlog与third_party/json。每步在可写作业工作区执行，单独申请28分钟。

```bash
git -C component-checkouts/InfiniCore submodule update --init third_party/spdlog
git -C component-checkouts/InfiniLM submodule update --init third_party/spdlog third_party/json
source "${IFAB_AFS_HOME}/minimax-h3/setup_infinifabric_env.sh"
# 每项作为独立作业运行；路径使用实际clone位置的绝对路径
python3 /absolute/component-checkouts/InfiniCore/competition/build_from_git.py native
python3 /absolute/component-checkouts/InfiniCore/competition/build_from_git.py python
python3 /absolute/component-checkouts/InfiniLM/competition/build_from_git.py extension
```

构建适配器从提交的Git树导出源码，依赖压缩文件仅用于xmake软件包索引及pybind头文件，不是参赛源码替代品。其SHA固定在代码。编译在job-local目录，安装到新的AFS .infini-public-20261111及python-envs/public-git-20261111/site-packages，不写共享解释器或替换既有.infini。新Git出口适配器尚需服务器重建确认，不能称空环境或完整模型复现通过。

## 保留的运行证据与限制

历史运行/比较入口仍在artifacts/infinifabric-build-v1；它们绑定旧回执、输入和候选，不能直接当作新公开Git构建的验收。原报告视频/SSIM44.643%和选定源码重建/首次前向证据均保持原始来源。九齿gfx906设备流性能48组均退化，正式八卡九齿调用、Qwen一致性、完整质量及新公开Git出口端到端运行仍待验证。独立Attention源码与测试位于Core的competition/attention-experiments，LLVM兼容配方位于Triton的competition/hygon-llvm-compat-v1。独立JIT 24次比较通过，AOT六种导出完成但BF16-tail失败，转换诊断在启动函数绑定阶段停止；正式模型调用和性能未获证。详见[Nine Attention实验与证据](NINETOOTHED-ATTENTION-EXPERIMENTS.md)。

原档案ZIP/服务器路径仅作历史来源佐证。提交形式为一份含全部仓库链接、分支、commit和复现说明的报告；不另提交源码ZIP或仓库表附件。报告含队伍联系信息，仅本地保留并交指定接收方，未上传公开GitHub。
