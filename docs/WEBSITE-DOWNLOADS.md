# 从运行网站下载材料

进入“个人存储”而非已结束实例的临时运行目录。下文路径以个人AFS的 minimax-h3/ 为根。
本地统一放本仓库 server-downloads/；该目录被Git忽略。不要把Native与Reference同名文件覆盖。

## 第一优先 继续定位Qwen 不必重跑

打开 logs/infinifabric-h3-input-boundaries-20261107T133314Z-3329662/。

| 网站相对文件 | 本地存放 |
|---|---|
| native/qwen-layer50.npz | server-downloads/input-boundaries/native/qwen-layer50.npz |
| native/qwen-layer50.json | server-downloads/input-boundaries/native/qwen-layer50.json |
| reference/qwen-layer50.npz | server-downloads/input-boundaries/reference/qwen-layer50.npz |
| reference/qwen-layer50.json | server-downloads/input-boundaries/reference/qwen-layer50.json |

同目录再取 native/capture.json、reference/capture.json（两份摘要原件尚未返回）；后续局部重放建议取两侧 qwen-inputs.npz 和 qwen-inputs.json。
这些是内部诊断数据，不得让正式生成入口复用它们代替实际编码。
input-boundary-parity.json和run.log已在本地results归档，无需重复下载。

## 第二优先 固定可复现版本

下载 setup_infinifabric_env.sh；state/ 中实际存在的运行时/候选清单JSON及成功回执；保留原始文件到 server-downloads，不直接公开。
保留 diagnostics/h3-load-sync/20261107T030125Z-2935713/ 的候选说明、构建记录及摘要；实际子文件名以网页目录为准，不假定每份都有success文件。
保存对应 logs/ 构建作业的 summary、版本/环境记录、完整构建日志和源码patch记录。
现有脚本读取这些文件；网页下载源码包不等于已拿到全部本地修改。

服务器需要另外采集、当前不能仅靠网页下载自动获得：InfiniCore、InfiniLM实际 HEAD/分支/子模块状态、git diff与未跟踪新增源码；python可执行路径/版本、pip freeze、Torch/HIP/DTK、九齿/Triton版本和wheel SHA256。
若服务器源码包不含.git，记录原始v3包SHA256、解包后的修改补丁及构建候选SHA256，commit注明不可用，不编造提交号。
模型配置/权重只记录路径与版本或摘要，不重复上传Git。

## 第三优先 提交成果与截图

- logs/infinifabric-h3-load-sync-generation-20261107T082020Z-3236632/：父及native子目录中的native-generation.json、run.log、worker.log、summary.txt（如存在）；native/fl2va-native-load-sync-50steps.mp4。
- logs/infinifabric-h3-reference-compare-20261107T124511Z-3527846/：reference-comparison.json、official-comparison.log已在results，无需重复取。保留reference/fl2va-reference-original-50steps.mp4与详细worker日志用于复核。
- 保存服务器显示8卡环境、实际入口、完成状态/输出指标的截图；图片必须是真实运行证据。
- 九齿gfx936安装/JIT报告及单卡门控残差正确性/性能原始JSON日志，如本地没有则补下载；具体目录按已有run_dir查找，不猜时间戳。

本地已有原版比较输出、媒体复核JSON和5张抽帧，不需要重新生成。视频/NPZ保存在server-downloads并做校验，Git只收必要小型结果与授权截图。

## 保留在服务器 不必整包下载

models、.infini、python-envs、packages/wheels、.xmake缓存、全量历史日志和大批中间数组。
当前候选库可作二进制取证保留在本地，但不是可复现源码的替代品，也不直接上传Git。
未完成新环境重建前不要删除服务器state、diagnostics或运行时目录。
