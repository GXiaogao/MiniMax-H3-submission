# 从GitHub源码到服务器验证

本说明对应代码快照cc9d33038651f12b905cc11c07aad1a108c5cd43。已验证范围是在既有InfiniFabric/AFS环境中重新构建InfiniLM候选、校验来源并执行单组H3首次前向，不是空系统一键部署，也不是正式视频质量/性能通过。

## 1 下载固定代码

在Windows PowerShell中选择自己的工作目录执行：

```powershell
git clone https://github.com/GXiaogao/MiniMax-H3-submission.git
cd MiniMax-H3-submission
git checkout cc9d33038651f12b905cc11c07aad1a108c5cd43
git rev-parse HEAD
```

最后应打印完整commit。没有Git也可下载固定源码ZIP：https://github.com/GXiaogao/MiniMax-H3-submission/archive/cc9d33038651f12b905cc11c07aad1a108c5cd43.zip 。解压后进入带README的仓库根目录。

代码包含固定v3源码归档和变换脚本，不含模型权重、全部wheel或服务器历史大数组。不要向网站项目输入区上传整个仓库。

## 2 核对服务器前提

以下材料应已经存在个人AFS的minimax-h3下：

- setup_infinifabric_env.sh：当前平台环境配置。
- .infini/lib及state/infinicore-python.success：已构建且跨作业可读的InfiniCore。
- Projects/InfiniLM/python：原始Python适配层，隔离候选会复制它。
- packages/xmake-repo-cache.tar.gz、packages/pybind11-v3.1.0.zip及已有xmake、DTK和厂商Torch环境。
- 私有运行时state清单及其对应的离线wheel；前向验证另需历史输入与前置回执。
- 指定共享MiniMax-H3模型及权重，不能另换模型或量化版本。

如果是新账号/新机器，这些前提不能假定存在：先按artifacts/infinifabric-env-v1与artifacts/infinifabric-build-v1的原始准备/构建说明分阶段建立环境，逐项验证。该全流程还没有在全新环境完成验收，不能把下面候选重建称为全工具链重建。

## 3 从源码组装上传包

以下PowerShell命令在仓库根目录运行，生成的local-packages只作本地上传材料，不提交Git：

```powershell
$repoRoot = (Get-Location).Path
$sourceDir = Join-Path $repoRoot 'artifacts/infinifabric-build-v1'
$bundleDir = Join-Path $repoRoot 'local-packages/rebuild'
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
$names = @(
    'rebuild_load_sync_isolated.sh',
    'prepare_load_sync_clean_rebuild.py',
    'build_infinilm_extension.sh',
    'prepare_minimax_h3_load_sync_trace.py',
    'prepare_minimax_h3_projection_trace.py',
    'prepare_minimax_h3_attention_trace.py',
    'prepare_minimax_h3_condition_trace.py',
    'prepare_minimax_h3_boundary_trace.py',
    'h3_weight_load_sync.hpp',
    'h3_projection_trace.hpp',
    'h3_attention_trace.hpp',
    'h3_boundary_trace.hpp',
    'load-sync-source-hashes.json',
    'InfiniLM-infinifabric-v3.tar.gz'
)
foreach ($name in $names) {
    $file = Join-Path $sourceDir $name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing: $name" }
    Copy-Item -LiteralPath $file -Destination $bundleDir -Force
}
$zipPath = Join-Path $repoRoot 'local-packages/load-sync-clean-rebuild-upload.zip'
$files = $names | ForEach-Object { Join-Path $bundleDir $_ }
Compress-Archive -LiteralPath $files -DestinationPath $zipPath -Force
Write-Output $zipPath
```

这14个文件必须保持同目录。源码包约10MiB。不要预先手工应用server-load-sync-restored.patch后再运行本构建器；构建器自己从固定原包应用相同变换。

## 4 网站启动隔离重建

1. 解压上一步ZIP。在网站“我的项目”新建小项目，把14个文件上传到项目同一目录，而不是上传ZIP等待平台自动解压。
2. 入口选择rebuild_load_sync_isolated.sh，参数留空，现有scnet-dcu八卡批处理，申请28分钟，内部25分钟限时。
3. 观察源码摘要门控、xmake配置/编译、扩展导入。候选发布在diagnostics/h3-load-sync-rebuild/<新ID>/InfiniLM；新回执只写本轮logs，不覆盖旧state或正式库。
4. 完成后按日志的run_dir进入个人存储，下载run.log、summary.txt、artifacts.sha256、source-hash-verification.json、infinilm-load-sync-rebuild.success。
5. 保存每一轮独立文件夹，核对exit_code=0、last_stage=complete、七项源码输出摘要匹配、候选路径与库摘要一致。

遇失败先保留完整日志。不能通过删除摘要校验、改旧回执或覆盖旧库来跳过问题。

## 5 首次前向验证及历史绑定

artifacts/infinifabric-build-v1/rebuild-forward-v1的入口validate_rebuilt_load_sync_forward.sh已经验证过，但它固定绑定20261108T100156Z-2712909候选、189e52b7…摘要，以及旧单块报告和first-inputs的具体来源。

- 重放已记录的那次候选：保留上述AFS文件，上传该目录中的全部同目录文件，入口validate_rebuilt_load_sync_forward.sh，参数留空，八卡28分钟。
- 自己刚产生的新构建：先核对新回执、完整库摘要、源码与Core/Python来源，再为该候选准备独立验证配置/入口。当前仓库没有自动接收任意新回执的通用启动器。不要直接假定历史入口可用于任意新库，也不要把旧报告的库身份改成新库。

历史first-inputs为内部诊断输入，不用于正式生成缓存。缺失时先说明，不把随意新生成的数组伪装成同一轮输入。

验证输出为logs/infinifabric-h3-rebuilt-full-forward-<新ID>/的full-forward-parity.json、run.log、summary.txt；失败时再取native/reference的worker.log。COMPARISON_COMPLETE只表示对照完成，还需检查数值误差与范围；本轮不做Qwen编码、完整视频或性能验收。

## 6 完整生成与提交材料

历史生成入口run_minimax_h3_load_sync_generation.sh及比较入口run_minimax_h3_reference_compare.sh也含固定候选与证据绑定。它们展示了已有实验，不是任意新环境通用部署入口。切换新候选后，先验证其身份与正确性，再准备对应生成配置；不能只替换.so或忽略SHA256门控。

README中两份视频都是本队服务器生成的输出。Native与PyTorch参考都使用首帧图片和提示词；参考输出不是Native输入。当前SSIM44.643%、Qwen偏差及九齿范围如实保留。

更多细节见[隔离重建](LOAD-SYNC-CLEAN-REBUILD.md)、[源码还原](LOAD-SYNC-SOURCE-RESTORATION.md)、[重建前向结果](REBUILT-FORWARD-RESULT.md)、[两条生成路径](NATIVE-AND-REFERENCE.md)。
