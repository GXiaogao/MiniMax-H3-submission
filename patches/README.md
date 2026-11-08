# 补丁说明

这两份补丁针对 configs/local-source-versions.json 中的本地 HEAD。
InfiniLM 仅收集 csrc 改动，未混入无关的 llama 测试工作区改动。
git diff 不包含未跟踪文件；这些文件单独保存于 src/local-untracked，必须一并复核。
尚未在干净检出上验证补丁应用及重建。八卡装载修复由 artifacts/infinifabric-build-v1 的补丁脚本和 h3_weight_load_sync.hpp 提供。
本地旧工作树、v3源码包和服务器候选是不同版本来源，不可把它们当成同一个最终提交。
