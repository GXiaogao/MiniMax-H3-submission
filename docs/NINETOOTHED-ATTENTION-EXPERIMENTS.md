# gfx936 九齿 Attention 实验与复现

## 源码位置与身份

- InfiniCore：[https://github.com/GXiaogao/InfiniCore-MiniMax-H3](https://github.com/GXiaogao/InfiniCore-MiniMax-H3)，分支`minimax-h3-preliminary`，commit`74ca95443a709b8c5ee1403a02965d25df91d025`。实验目录`competition/attention-experiments`。
- Triton兼容配方：[https://github.com/GXiaogao/triton-MiniMax-H3](https://github.com/GXiaogao/triton-MiniMax-H3)，分支`minimax-h3-preliminary`，commit`f5c20bf89d439edab2ce9841d2c5275967171806`，目录`competition/hygon-llvm-compat-v1`。
- NineToothed使用[公开组件清单](PUBLIC-SOURCE-REPOSITORIES.md)中固定0.26.0基础上的缓存隔离修改；启动函数重复替换问题发生在未修复的生成器路径，不将其写成已完成的Nine编译器修复。

原InfiniOp九齿DSL与MIT许可随实验公开；候选仅修正动态K/V块数上取整。C++ AOT启动器只做HIP Module装载、参数绑定和指定stream启动，不含自写设备计算。原始Attention/模型评测参考保持不变。
LLVM配方针对固定厂商compiler.py摘要e7d31b37a0a59c73e1bba6a5400af3af8a15f18b9a7d1b17b4c3644dee2c8401，仅处理GEP nuw/nusw解析兼容，并记录IR前后摘要，不替换厂商后端或分发其wheel。

## 服务器证据

本仓库[results/ninetoothed-attention-20261111](../results/ninetoothed-attention-20261111)保存去隐私的服务器JSON/日志、原文件SHA256与发布后SHA256。仅替换个人AFS前缀和临时作业根路径，数值/结果字段不变。原始日志和数组在本地及AFS保留；这里的报告不是可执行HSACO/so二进制。

| 实验及作业 | 结果 | 不能据此声称 |
|---|---|---|
| JIT 20261110T063310Z-2643063 | 12个开发用例、24次比较通过；FP16最大绝对误差0.0007062888，BF16为0.0048552823 | 正式接口、八卡模型、全形状或性能通过 |
| AOT 20261111T015938Z-364509 | 六种导出成功、14次比较通过，第15次BF16-tail失败；失败样本maxabs1.037658、relativeL2约0.808814 | AOT整体正确或BF16根因已确认 |
| 转换诊断 20261111T060308Z-612075 | `KeyError: launch_nine_cast_fp64_direct`；转换和Attention成对测试未运行 | BF16数值问题已修复或恶化 |

JIT用例最大batch2、Sq33、Sk65、Hq4、Hkv2，head dim64/128，覆盖单key、尾块、GQA、右下因果及带间隙布局。开发容差FP16 atol=rtol=0.01、BF16=0.025，不是赛事验收阈值。
JIT控制器85.77秒含环境准备、编译和验证，不能作为模型耗时或加速收益。正式模型当前SSIM44.643%，未因这些实验取得改善证据。

## 获取可上传源文件

先按公开清单clone并checkout各组件固定commit，再执行：

```bash
python3 scripts/verify_public_sources.py --checkouts component-checkouts
python3 component-checkouts/InfiniCore/competition/attention-experiments/export_upload.py jit /可写新目录/jit-upload
python3 component-checkouts/InfiniCore/competition/attention-experiments/export_upload.py aot /可写新目录/aot-upload
python3 component-checkouts/InfiniCore/competition/attention-experiments/export_upload.py bf16 /可写新目录/bf16-upload
```

每条命令先核对源清单并拒绝覆盖已有目录；输出为13、19、20个平铺文件。将选定目录全部文件上传到网站项目同一目录，入口分别为run_hygon_ninetoothed_attention.sh、run_hygon_attention_aot.sh、run_hygon_bf16_diagnostic.sh。每项单独申请28分钟，参数空。
服务器沿用已验证的私有Nine环境、state、厂商Triton wheel、DTK/Torch和AFS setup脚本；不修改共享解释器。BF16模式还只读复用AFS作业20261111T015938Z-364509完整导出目录。公开日志并不能替代这些前提；缺失会失败，非空环境一键安装。
BF16上传源码故意保留服务器实际失败状态，用于复现阻断，不是修复包。失败内核不可作为正式模型计算结果。

## 本地验证与限制

Core实验目录README说明宿主测试、可选wheel环境变量及准确运行边界。44项宿主契约测试通过，覆盖源摘要、错误输入、生命周期控制、ABI/LDS、IR兼容和打包；不等同设备编译、JIT启动函数绑定或数值验证。原源码生成测试未检查启动函数与元数据对应，已识别该覆盖缺口。
下一步先修复诊断命名/绑定并验证，再定位BF16 JIT/AOT数值偏差；随后才做正式InfiniOp/Core登记、真实Qwen/H3形状、八卡调用和受控质量/性能比较。未安装启用这部分独立实验到既有正式库。
