# 第三方来源与许可记录

本目录包含第三方修改与源码快照，并非全部代码均为本队原创。没有给整个仓库套用新的MIT等许可证。

| 组件 | 来源与版本 | 许可材料 |
|---|---|---|
| InfiniCore | 原离线v3源码包；本地旧工作树HEAD见configs，服务器实际版本待采集 | 原源码包内InfiniCore/LICENSE及各子模块许可证；licenses/本地副本 |
| InfiniLM | 原离线v3源码包与补丁；服务器实际版本待采集 | 原包第三方目录保留原许可；最终提交前核对上游本体许可，不能以spdlog许可替代InfiniLM本体许可 |
| NineToothed | 0.26.0，单卡门控残差及独立JIT证据范围见结果 | licenses/保存本地该版本许可证 |
| spdlog/json等 | 随固定源码包保留 | 各原包内部LICENSE/NOTICE，不删除 |
| 厂商Torch/Triton/Flash Attention等 | 依赖清单及厂商wheel，不重新分发wheel | 下载来源、实际构建版本与许可需随服务器环境回执最终核对 |

当前是开发快照。固定第三方完整清单和InfiniLM本体许可核对仍是最终提交待办，不将此表当作主办方许可验收。
