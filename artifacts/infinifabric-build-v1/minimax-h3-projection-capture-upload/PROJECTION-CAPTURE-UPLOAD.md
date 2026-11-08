# 输出投影与归约对照

上一步构建及导入成功。本包只做独立诊断，正式 Native 扩展保持原样，不生成完整视频，不代表质量或性能验收。

1. 解压 `minimax-h3-projection-capture-upload.zip`，将其中八个 `.py` / `.sh` 执行文件上传到当前平台项目同一目录；不要把整个依赖或模型目录上传进项目输入区。
2. 入口选 `diagnose_minimax_h3_projection.sh`，程序参数留空，使用 `scnet-dcu` 整节点八卡，时长 28 分钟。内部检测超时 25 分钟，保留失败日志。
3. 完成后在个人存储 `minimax-h3/logs/infinifabric-h3-projection-<本次ID>/` 下载 `projection-parity.json` 和 `run.log`。失败时还下载对应 `native-tp1`、`native-tp8` 或 `reference` 子目录中的 `worker.log`。

本包固定使用刚返回的诊断构建：

- 扩展目录：`diagnostics/h3-projection/20261106T162326Z-2351670/InfiniLM`
- 扩展 SHA256：`cbe5cad7446f359a39196bcae3fde17d08d818b5ba5c1c77778d75cc45595065`
- 构建证据：`logs/infinifabric-build-h3-projection-20261106T162328Z-2351688/`
- 状态标记：`state/infinilm-projection.success`
- 输入复用：`logs/infinifabric-transformer-parity-20261105T080500Z-408969/first-inputs.npz` 及其 `.json`。

上述文件、原正式扩展、私有运行时清单及离线 wheel、模型权重需保留；脚本会先核对哈希和诊断 Python 文件副本。无需重新构建或重新下载模型。

分别顺序运行 Native TP1、Native TP8 和独立参考：保存既有 24 处边界、每卡 Attention 边界及五个新增投影阶段。标准/预转置权重布局按原始形状匹配，参考局部矩阵乘从完整 MHA 与 checkpoint 权重按列切片后计算，再采样。

解释结果时须保留区别：`out_original` 是原组合调用的输出；`out_local` 是额外重算；`out_control_sum` 是额外的独立输出 SUM。原海光路径归约为原地操作，控制路径分配新输出。参考局部结果累加使用 FP32 并转回输出精度，主机累加使用 FP64，均不保证与设备归约的舍入顺序相同。参考与 Native 局部矩阵乘的输入 MHA 已有约 0.6% 差异，因此局部误差还包含上游差异，不能独立证明 GEMM 故障。新增同步也可能改变调度。

每个张量至多比较 65,536 个确定位置；完整张量有限性和正式生成质量未据此验证。`COMPARISON_COMPLETE` 仅表示三个进程及规定采样材料完整，不是数值合格结论或性能成绩。正式核心计算继续采用 Native；参考工具只在独立检测进程中使用。
