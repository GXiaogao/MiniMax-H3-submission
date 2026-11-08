# 下一步：独立 VAE 编解码重建

本次 Native 精度诊断运行成功，但抽帧仍为彩色噪声。下一步单独检查现有视频 VAE 能否把狗的图片编码后还原，不重复整段生成。

## 上传与启动

解压 `minimax-h3-vae-roundtrip-upload.zip`，把以下五个脚本上传到同一平台项目根目录；后三个沿用当前已通过的版本：

1. `diagnose_minimax_h3_vae_roundtrip.sh`（新增入口）
2. `diagnose_minimax_h3_vae_roundtrip.py`（新增诊断程序）
3. `run_minimax_h3_fl2va_native_smoke.py`
4. `install_infinilm_private_runtime.py`
5. `prepare_minimax_h3_hygon_runtime.py`

入口选 `diagnose_minimax_h3_vae_roundtrip.sh`；程序参数留空；使用此前成功的 scnet-dcu 环境、批处理作业、28 分钟。平台仍分配整节点 8 卡，诊断仅使用 `cuda:0`。模型、wheel、首帧沿用现有存储，无需下载、安装或重新编译。

## 这个测试做什么

- 只加载共享模型目录中的视频 VAE，保存实际权重加载信息、配置、实现哈希和参数精度统计。
- 首帧按现有 FL2VA 流程直接缩放到 576×384，再按指定 VAE 的像素均值/标准差归一化。
- 把这张图重复为 22 帧，使用 VAE 后验的 mode 做独立编码→解码检查，保存还原图片、有限值检查和逐帧重建误差。
- 22 帧满足该 VAE 的时间分块约束。这是独立辅助组件诊断，既不替换正式 124 帧生成，也不参与比赛性能或相似度计分；正式 Native 路径保持原样。
- 保持模型加载后的混合精度，仅移动设备，不再次强制转换所有 VAE 参数。出现缺失权重、精度违规、非有限值或异常形状时停止并保留报告。

## 运行后取回三个文件

目录：`$WORK_HOME/logs/infinifabric-vae-roundtrip-<运行编号>/`

1. `vae-roundtrip-contact-sheet.jpg`：处理后的输入及第 0、1、4、10、21 帧还原结果。
2. `vae-roundtrip.json`：权重加载、参数精度、张量统计和重建误差。
3. `run.log`：完整 Python 运行输出。

若失败，也取回已有 JSON、`run.log` 和 `summary.txt`，无需反复提交同样任务。

`DIAGNOSTIC_COMPLETE` 只表示收集结束，仍需检查还原图。如果还原图仍有噪声，继续定位 VAE 权重、实现或后端；如果还原正常，再检查正式首帧条件处理及 Native Transformer 输出。重建 PSNR 是本项自检指标，不能作为官方视频相似度成绩。
