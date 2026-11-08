# 离线依赖

八卡wheel输入位置及版本见 artifacts/infinifabric-build-v1 的 *wheels*.txt、*inventory*.json、下载PS1和运行包说明。
服务器个人AFS的 packages、wheels、python-envs 继续保留；不复制到Git。
待下载的 pip freeze 只是该解释器的快照，还必须同时记录私有site-packages暂存状态、厂商Torch/DTK版本和wheel SHA256。
服务器setup_infinifabric_env.sh和state/*.json属于本地取证材料，公开前去除个人绝对路径并核对敏感项。
