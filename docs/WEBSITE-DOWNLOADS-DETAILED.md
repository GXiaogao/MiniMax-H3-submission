# 网站下载操作详解 2026年11月8日

## 网页通用操作

1. 打开 InfiniFabric 左侧“个人存储”，找到 minimax-h3，点击“打开”。不要进入“我的项目”上传区，也不要找已结束实例的临时run目录。
2. 按下文路径逐级点击目录行的“打开”。路径均相对于 minimax-h3/，不是Windows路径。
3. 找到文件后点击其行右侧“下载”。浏览器可能直接保存到 D:/下载，也可能弹出另存为。
4. 下载后移入本项目 server-downloads 下对应目录；同名的 run.log、capture.json 等一定按任务及native/reference分目录保存，不混用浏览器自动添加的(1)/(2)后缀。
5. 网页找不到指定文件时先查看所在目录，不要重跑生成或修改文件名。记录缺失项，提供该目录列表即可进一步确认。
6. “下载”不会删除服务器原件。无需为了取文件重新申请8卡或启动生成。网页若没有目录下载功能，就逐个下载本清单的小文件，不必打包整个logs。

本机提交根目录：E:/projects/minimax-h3-match/相关文档/MiniMax-H3-submission。
以下“本地”均在该根目录内。server-downloads被Git忽略，文件收齐后还需要筛选、脱敏和校验才进入正式公开材料。

## A 现在优先取的6份小文件

四个qwen-layer50文件已下载，不用再取。这些数组的数值/元数据配对仍需后续核验，本次只确认文件存在和NPZ容器可读。

先进入 logs/infinifabric-h3-input-boundaries-20261107T133314Z-3329662/：

| 进入子目录 | 下载文件 | 本地位置 |
|---|---|---|
| native | capture.json | server-downloads/input-boundaries/native/capture.json |
| reference | capture.json | server-downloads/input-boundaries/reference/capture.json |

capture.json记录捕获阶段与数组来源，用于核对两侧数据是否来自同一轮。

返回minimax-h3根目录，再进入state，优先下载下面4个明确被现有链路使用的文件，统一保存到server-downloads/state/：

- infinilm-extension.success：原始InfiniLM扩展的来源/摘要/构建记录线索。
- infinilm-load-sync.success：本次隔离装载修复候选的回执及run_dir。
- infinilm-runtime-private.json：私有离线Python运行时清单。
- minimax-h3-hygon-runtime.json：海光辅助运行时清单。

.success是普通文本回执文件，不是可执行程序，保留扩展名，不改成截图。
如果上述回执内容指向后来新建的候选，保留它并说明，不能假设它还是本次历史实验版本。

## B 接着取的环境与构建文件

在minimax-h3根目录下载setup_infinifabric_env.sh，保存为server-downloads/environment/setup_infinifabric_env.sh。
在state/再下载以下实际存在的文件，保存到server-downloads/state/：

- infinicore-python.success：InfiniCore Python绑定构建回执，扩展构建脚本明确读取它。
- ninetoothed-gfx936.json：已成功发布的九齿环境状态文件；本地此前只有安装/探测报告，没有此原件。

其他state回执先保留服务器原件，不用无差别全下载。

### 如何找到正确的构建日志

1. 本地用记事本打开刚下载的infinilm-load-sync.success。
2. 查找run_dir=，按该值在个人存储中进入对应logs目录。不要仅凭目录时间最新就当作本次候选。
3. 在该目录下载以下实际存在的文本：summary.txt、run.log、artifacts.sha256、InfiniLM_xmake.conf，以及以.trace.json结尾的补丁记录。保存在server-downloads/build/load-sync/。
4. 对infinilm-extension.success和infinicore-python.success重复上述操作，分别保存到server-downloads/build/infinilm-original/和server-downloads/build/infinicore-python/。Core日志中的配置快照可能不叫InfiniLM_xmake.conf，只取其实际配置文件，不强求同名。
5. 如果run.log只是一层外部启动日志，按其中的下一层run_dir继续取真正构建summary、run.log、artifacts.sha256和补丁记录；不要只保存“构建成功”的截图。

diagnostics/h3-load-sync/20261107T030125Z-2935713/ 是已记录候选源码所在目录，内部InfiniLM是源码树，不是一个“候选说明.json”文件。前一版清单这里较笼统，**现在不用在网页里逐个下载整个InfiniLM**，也不用猜manifest文件名。
候选.so及共享库先保留服务器原件；完整源码快照或修改补丁需下一步单独采集，见E。

## C 为后续局部重放提前取4份文件 可稍后再做

回到logs/infinifabric-h3-input-boundaries-20261107T133314Z-3329662/：

| 子目录 | 文件 | 本地目录 |
|---|---|---|
| native | qwen-inputs.npz、qwen-inputs.json | server-downloads/input-boundaries/native/ |
| reference | qwen-inputs.npz、qwen-inputs.json | server-downloads/input-boundaries/reference/ |

这批用于内部同输入诊断，不是正式生成的预计算输入。先下载A和B即可继续当前文件核对，不必等待C全部取齐。

## D 成果备份与截图 提交前完成

进入logs/infinifabric-h3-load-sync-generation-20261107T082020Z-3236632/，保存到server-downloads/generation/并保留子目录结构：

- 根目录native-generation.json、run.log、summary.txt（如存在）。
- native/native-generation.json、native/worker.log。
- native/fl2va-native-load-sync-50steps.mp4（本地已有同一份可复用，无须再取）。

旧v1.1完成次数门控曾误报，此次原生作业exit非零不应被改写。它与后续媒体复核及原版比较报告配套说明，不能只保留其中一份“成功”输出。

进入logs/infinifabric-h3-reference-compare-20261107T124511Z-3527846/：

- reference/fl2va-reference-original-50steps.mp4，保存到server-downloads/reference-compare/reference/（已有同一份则跳过）。
- reference/worker.log，若存在，保存同目录；若名字不同，按网页实际日志名保留。
- reference-comparison.json、official-comparison.log及五张抽帧已归档到本地results，通常不必重新下载。

截图保存到server-downloads/screenshots/，至少保留：

- 实际8卡运行环境或硬件探测输出。
- 实际生成入口与运行状态，以及日志中真实完成阶段/媒体复核结果。
- 原版比较的Size、Length和Average SSIM 44.643%输出。
- 九齿JIT九组用例通过的真实输出（有原始截图则复用）。

截图可用Windows截图工具选取页面；展示任务名、关键结果和实际环境，避开登录凭据。截图只是补充，不能代替JSON和完整日志。

## E 不能只靠“点击下载”得到的内容 暂不用你手工执行命令

下面信息若服务器尚未输出成文件，网页里不会凭空有对应文件，需要下一步通过只读采集作业生成材料后再下载：

- InfiniCore/InfiniLM实际源码HEAD、分支、子模块版本、git diff和未跟踪新增源码。
- 无.git时的v3源码包SHA256、当前源码与基线的差异、候选构建来源。
- Python路径/版本、pip freeze、Torch/HIP/DTK、九齿/Triton版本、实际wheel SHA256及运行时暂存状态。

不要把Windows本地旧HEAD填成服务器八卡版本；不能仅上传.so就称为源码可复现。目前无需手工下载Projects全部文件、逐个点击源码，后续应生成源码补丁或快照归档。

## F 本地已有 不用下载或重跑

results中已有：input-boundary-parity.json/run.log、reference-comparison.json/official-comparison.log、媒体复核JSON与抽帧。
九齿安装报告ninetoothed-install.json、ninetoothed-probe.json和run.log已从本地开发归档补入 results/ninetoothed-success-20261106T124640Z-1947749/。
单卡九齿性能原始材料仍在开发目录，本次不用到网页盲找历史任务；最终打包时需与报告按具体实验对应。

models、.infini、python-envs、packages、wheels、.xmake缓存、大批历史中间数组不必全量下载。不删除服务器state、diagnostics或现有运行时。

## 完成后最小检查

A的2个capture.json和4个state文件，加B的setup脚本、2个state文件，是本轮明确的9份小文件；构建日志另按回执run_dir取。
下载后保留原文件名/任务目录，不覆盖另一侧文件。文件大小为0、下载到的是HTML登录页或浏览器报错时，不能算已下载成功。
