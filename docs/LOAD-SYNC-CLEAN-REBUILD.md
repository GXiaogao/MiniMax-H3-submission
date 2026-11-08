# 隔离干净重建操作

上传包位于本地server-downloads/rebuild-load-sync/load-sync-clean-rebuild-upload-v1.zip。解压后将14个文件全部上传到同一个新项目根目录，入口rebuild_load_sync_isolated.sh，参数留空，scnet-dcu八卡，申请28分钟，内部构建限时25分钟。源码包约10MiB，不上传整个submission目录。

从固定v3源码包重新解包、应用固定构建变换及候选补丁，编译前校验与旧服务器trace对应的3个修改文件和4个辅助头文件。新库写入diagnostics/h3-load-sync-rebuild/<新ID>/InfiniLM，成功回执只写入本轮logs目录，不替换state/infinilm-load-sync.success或当前正式库。

前提是个人AFS现有setup_infinifabric_env.sh、state/infinicore-python.success、共享Core库、packages内pybind11和xmake-repo缓存等仍保留。本次仍复用已安装的InfiniCore，不是整个工具链从空系统重建。脚本不安装模型、不执行视频生成；构建结束只做扩展导入smoke检查。

日志应出现REBUILD_VERSION、SOURCE_HASH_GATE PASSED、xmake-build和InfiniLM extension build: PASSED。找到最终run_dir对应的个人存储logs/infinifabric-rebuild-h3-load-sync-<新ID>/，下载：run.log、summary.txt、artifacts.sha256、source-hash-verification.json、infinilm-load-sync-rebuild.success。保存到本地server-downloads/rebuild-load-sync/。如失败先返回完整run.log及summary，不覆盖旧候选重试推理。

重建成功只代表源码门控/编译/导入完成，不证明首次前向、视频相似度或性能通过。新库摘要可能不同于旧库，必须先核对新回执再准备后续验证入口；不要直接改旧生成脚本的硬编码摘要，也不要把新库覆盖到旧库路径。
