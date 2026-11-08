# 首输入边界对照完成：显著偏差已出现在Qwen输出

用户返回作业 `20261107T133314Z-3329662` 的报告与日志，报告result=`INPUT_BOUNDARY_COMPARISON_COMPLETE`、last_stage=complete、completed_workers=[native,reference]，耗时380.0957秒含暂存和加载。**本轮已按设计完成，不是超时或中断。** 两边均在首次H3调用前主动捕获后停止；0/49表示没有执行去噪，generation_status=NOT_EXECUTED/H3_forward_executed=false是本局部诊断的预期状态，不是整轮未完成。

## 结果

| 边界 | 结果 |
| --- | --- |
| Qwen实际输入 | input_ids、attention_mask、image_grid_thw、mm_token_type_ids、pixel_values全部逐元素相同，原始dtype匹配，标量选项相同 |
| Qwen第50层隐藏状态 | 形状1×708×5120；相对L2 **0.7169845115537381**，余弦0.7758305402791983，MAE0.2191189618，最大绝对差21967 |
| 首次H3 encoder_hidden_states | 与上项比较指标一致，显著偏差已传入H3条件输入 |
| VAE原始像素/编码配方 | 像素逐元素一致；pixel_mean/std/编码seed等元数据相同 |
| 归一化VAE条件 | 形状1×24×1×58×48；相对L2 **0.03597584954278625**、余弦0.9993529616941227、MAE0.0228111877、最大绝对差0.8110432327 |
| 噪声及随机流 | 条件噪声、视频噪声、音频噪声3次输出、每次CPU RNG前后5056字节状态均完全相同；形状相同 |
| 首次H3音频输入 | 1×414×32，逐元素完全相同 |
| 首次H3视频输入 | 1×26448×96，相对L2 0.005877188395908812；包含条件行与随机视频行，不把该整体误差当作VAE单项误差 |
| H3布局/时间 | position_ids、token_tags、video/audio/text_indices、timestep/timestep_indices全部逐元素相同 |

输入图像、Qwen预处理张量、初始噪声和布局在本次观测边界一致；最大显著上游偏差已发生在Qwen第50层输出，优先排查编码器实际加载/视觉与语言子模块/位置处理/层输出语义。71.698%是相对L2，不是失败率。最大差21967而MAE0.219提示需要查看误差在token/通道中的分布，不能仅凭总体L2断定所有token均严重错误。

VAE条件的3.598%差异也是独立待查事实；既有精度审计已确认两侧配置不同，但本次不是受控的单因素VAE试验，不把全部差异直接归因于某个转换。H3本轮未执行，不能由本轮断言其没有残差或后续迭代不放大误差；全视频SSIM44.643%的全部成因尚未确定。没有改变原始参考基线、权重或正式配置。

## 原始证据

归档 `artifacts/fl2va-quality-evidence-2026-11-04/input-boundary-result-20261107T133314Z-3329662/`：

- input-boundary-parity.json SHA256 `7f85368a7bf95299ce6eaea2a4c6c305ce9588d6b43dd31fa9bcd6ce5deb0ba1`。
- run.log SHA256 `edb684925effe58ccb7196bb116632c5015bb13b0d4a9f5cd63a2f6262e20479`。
- 返回报告记录Native capture.json SHA `ddafde966ee877abfccb1cffe3e2d09625ed094086e740284791c884c831ba9b`、参考capture.json SHA `d571cd82d0d4f2dd75220ce296bf847113bc459d9fe04d6283ac80018d7f09c7`。这两个原件和数组尚未下载到本地。

## 下一项可复用材料

不用重跑视频或本轮捕获。服务器已保存两侧的数组，后续Qwen局部重放也可读取本轮原输入。

先取两侧第50层输出，用于本地独立重算与误差分布分析：

个人AFS目录 `minimax-h3/logs/infinifabric-h3-input-boundaries-20261107T133314Z-3329662/` 下：

| 来源 | 文件 |
| --- | --- |
| native/ | qwen-layer50.npz、qwen-layer50.json |
| reference/ | qwen-layer50.npz、qwen-layer50.json |

两组同名文件需按Native/Reference分目录或改名前缀区分。NPZ元数据应配对，不混用。保留该目录的qwen-inputs.npz/json及capture.json、其他边界数据，以便后续不重复预处理，先不用全部下载。没有根据本轮结果直接修改实现或开始提速。
