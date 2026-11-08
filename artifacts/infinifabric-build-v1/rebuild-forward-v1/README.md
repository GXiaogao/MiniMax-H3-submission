# 重建库首次前向验证

入口validate_rebuilt_load_sync_forward.sh，参数留空，scnet-dcu八卡批处理，28分钟（内部25分钟）。本包固定使用20261108T100156Z-2712909候选目录及189e52b7…新扩展摘要，先验证构建清单/源码/共享Core/Python副本，再加载执行。

保留此前单块报告297113ef…及其旧候选ee448772…身份，作为历史前置证据；不能把旧报告中的库摘要改为新库。使用原始同输入捕获，顺序执行新Native完整50主块首次前向与固定参考，仅用于同输入诊断，不执行视频或Qwen编码。图和trace关闭，比较所有输出元素，不自行添加合格阈值。

需要服务器原始first-inputs与历史load-sync-parity报告仍在原目录，不用上传大数组。若缺失或摘要不同先报错，不重新生成/篡改输入。新库与Core共享库/候选Python副本需在新作业可读取；读取失败会在模型执行前明确报错。

输出在logs/infinifabric-h3-rebuilt-full-forward-<新时间>-<PID>/：full-forward-parity.json、run.log、summary.txt，以及native/native-full-model.json、native/worker.log、reference/reference-model.json、reference/worker.log。优先下载根目录三份到本地server-downloads/rebuild-load-sync/forward-validation/。若失败再补对应worker.log。

本次只验证当前单组输入的首次前向；不代表Qwen偏差已修复、完整视频SSIM已提高、正式性能通过，也不是从空系统重建全部工具链。
