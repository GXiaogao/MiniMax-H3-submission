# 服务器只读采集操作

## 上传与运行

1. 本地打开本仓库scripts，选择collect_server_submission.sh与collect_server_submission.py两个文件。
2. 在网站“我的项目”新建小项目，例如MiniMax-H3-collection，上传这两个文件到同一目录。不上传整个submission目录、模型或wheel。
3. 入口选collect_server_submission.sh，参数留空，使用已有scnet-dcu环境，批处理作业。申请20分钟（平台最多28分钟）；脚本内部限时15分钟，通常更短。不改变环境软件包。
4. 日志将依次显示SOURCE_COLLECTED和PROBE_START，结束打印COLLECTION_RESULT、DOWNLOAD与SHA256。
5. 到“个人存储→minimax-h3→logs”，找到本轮infinifabric-source-environment-<时间>-<PID>.zip及同名.zip.sha256，下载到本地server-downloads/source-environment/。建议把网页完整运行日志也下载到该目录。
6. 如果ZIP超过100MiB，下载全部.zip.part-000、001…及.zip.sha256，保留顺序，回传后在本地合并核验；不要将某一个分片改名成zip。

运行无需source旧初始化脚本，不会安装依赖、编译、执行模型、读取模型权重，也不修改Projects、diagnostics、state。它会在logs创建新的采集目录和ZIP，离线wheel只读取校验。
输出包含当前InfiniCore/InfiniLM及success回执所指load-sync候选文本源码、许可证、Git状态/补丁（有.git时）、基础解释器环境、状态文件和离线输入摘要。
文件/总量限制、权限错误、缺失源码将写入report.json；COLLECTED_WITH_GAPS表示材料存在缺口，不是模型错误，也不是构建/推理通过。
厂商GPU工具不可用或Torch探测超时会记录到environment子报告；私有依赖版本同时由state记录，pip freeze仅代表当前解释器，不能代表全部暂存运行时。

## 数据与Git

ZIP保留服务器个人绝对路径及pip freeze原文，默认放被Git忽略的server-downloads。不自动上传Git；先校验和筛选，去除凭据或私有下载链接后再发布需要的摘要/补丁。
当前本地的旧工作树补丁、离线v3包和服务器实际运行源码是三条来源，服务器采集回来后再合并核对，不伪造最终commit。
