# 装载修复候选源码还原 2026年11月8日

## 结果与范围

从固定InfiniLM v3源码包出发，复用经过摘要门控的原始构建脚本中的海光基础源码变换，再运行已有load-sync补丁生成器。核对服务器构建trace中的11项摘要全部一致：H3/量化源码输入与输出、rank归档/基础变换后/候选输出，以及四个辅助头文件。
这包括7项输出文件摘要和4项输入阶段摘要，不是11个不同输出文件。基础变换还修改Qwen/视觉元数据、infer_engine和static_attn源码；这些没有独立服务器源码摘要，本次只证明按固定脚本生成，不擅自扩大摘要验证范围。

结果包含7个修改后的C++文件和4个新增头文件。补丁经git apply --check、实际应用，再与还原文件逐字节比较通过。

## 文件与复现

- scripts/restore_load_sync_source.py：只读固定输入、写入新目标目录，不编译、不运行模型。
- configs/load-sync-source-hashes.json：从实际服务器trace抽取的摘要，不含个人路径。
- src/reconstructed-load-sync/csrc/：选取的已修改源码及辅助头文件，不是完整InfiniLM树。
- patches/server-load-sync-restored.patch：相对固定InfiniLM v3源码包内部InfiniLM根目录的变更。
- results/server-collection-20261108/source-restoration.json：来源、摘要和补丁往返检查记录。

在仓库根目录，用Python3.11或更新版本执行：

```bash
python scripts/restore_load_sync_source.py --output /path/to/new-empty-candidate
```

目标目录必须不存在；源码归档或脚本不匹配时会拒绝，而不是猜测应用补丁。
若要使用patch，在完整解包的InfiniLM根目录先git apply --check，再git apply。应用这个patch后不要再重复执行包含相同基础变换/trace变换的旧构建器，否则会按设计拒绝重复打补丁。沿用旧完整构建流程时应由其生成器从干净v3原包开始，不手工再叠加本补丁。

## 尚未验证

没有在Windows编译海光候选，没有重建.so，没有声称恢复服务器缺失的完整临时工作树，也没有新的正确性或性能结果。还原得到的是带可关闭诊断的既有候选源码；正式推理应保持诊断关闭，不能把诊断采样时间当作正式性能。
服务器实际二进制仍由已有ee448772…摘要与构建回执追溯；源码摘要匹配不等于重建二进制一致。下一步需在统一服务器环境隔离干净重建、验证装载和首次前向，再更新最终可复现版本。
