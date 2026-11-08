"""Assemble a reviewable submission copy without changing the working sources."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('E:/projects/minimax-h3-match/相关文档/MiniMax-H3-submission')
records = []
allowed = {'.py','.sh','.ps1','.cpp','.cc','.c','.h','.hpp','.lua','.toml','.cfg','.ini','.json','.jsonl','.md','.txt','.patch','.diff','.cmake'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p, rel):
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha(dest) != sha(p):
        raise RuntimeError('Refusing to overwrite different file: '+str(dest))
    shutil.copy2(p, dest)
    records.append(dict(path=rel.as_posix(), source=str(p), bytes=p.stat().st_size, sha256=sha(p)))

def write(rel, text):
    p=OUT/rel; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('Already exists: '+str(p))
    p.write_text(text, encoding='utf-8')

# Preserve package-relative imports and helper closure, excluding binary inputs.
for base in ['artifacts/infinifabric-build-v1','artifacts/infinifabric-env-v1',
             'artifacts/h3-mha-singleton-stride-v1','artifacts/h3-ninetoothed-v2',
             'artifacts/h3-encoder-bridge-v1','scripts','tests']:
    for p in (ROOT/base).rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__']): continue
        if p.suffix in allowed or p.name.startswith('SHA256SUMS') or p.name in ['CMakeLists.txt','Makefile'] or p.name=='minimax-h3-template-first-frame.png':
            if p.stat().st_size <= 2*1024*1024: copy(p,p.relative_to(ROOT))

# Exact offline source archives used by the existing eight-card build scripts.
for name in ['InfiniCore-infinifabric-v3.tar.gz','InfiniLM-infinifabric-v3.tar.gz']:
    p=ROOT/'artifacts/infinifabric-build-v1'/name
    copy(p,p.relative_to(ROOT))

# Retain the latest gated-residual implementation, not only the obsolete v2.
for name in ['h3-gated-rounding-v3.tar.gz','h3-gated-benchmark-v1.tar.gz']:
    p=ROOT/'artifacts'/name; copy(p,p.relative_to(ROOT))
    with tarfile.open(p) as tf:
        for member in tf.getmembers():
            rel=Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts: continue
            if rel.suffix not in allowed and not rel.name.startswith('SHA256SUMS'): continue
            data=tf.extractfile(member).read()
            target=OUT/'src'/p.name.removesuffix('.tar.gz')/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise RuntimeError(str(target))
            target.write_bytes(data)

docnames=['competition-requirements.md','preliminary-template-status-2026-11-07.md',
 'fl2va-input-boundary-result-2026-11-07.md','fl2va-reference-compare-result-2026-11-07.md',
 'fl2va-full-forward-load-sync-result-2026-11-07.md','h3-gated-v3-acceptance-2026-10-27.md',
 'h3-gated-performance-results-2026-10-28.md','ninetoothed-gfx936-success-2026-11-06.md']
for name in docnames: copy(ROOT/'docs'/name,Path('docs')/name)
evidence=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04'
for folder in ['input-boundary-result-20261107T133314Z-3329662',
               'reference-compare-result-20261107T124511Z-3527846',
               'media-recovery-result-20261107T090136Z-3046211']:
    for p in (evidence/folder).rglob('*'):
        if p.is_file() and p.suffix in {'.json','.log','.txt','.png'} and p.stat().st_size<5*1024*1024:
            copy(p,Path('results')/folder/p.relative_to(evidence/folder))
report=next((ROOT/'reports/preliminary-2026-11-07').glob('*.docx'))
copy(report,Path('reports')/report.name)

versions=[]
for dirname, roots in [('InfiniCore-h3-mha',['include','src','python']),('InfiniLM-qwen-video',['csrc'])]:
    repo=ROOT/'worktrees'/dirname
    def git(*args): return subprocess.check_output(['git','-C',str(repo),*args])
    patch=git('diff','--binary','HEAD','--',*roots)
    dest=OUT/'patches'/f'{dirname}.patch';dest.parent.mkdir(exist_ok=True);dest.write_bytes(patch)
    # git diff omits untracked implementation files; preserve those explicitly.
    untracked=git('ls-files','--others','--exclude-standard','-z','--',*roots).decode('utf-8').split('\0')
    for rel in filter(None,untracked):
        p=repo/rel
        if p.is_file() and p.suffix in allowed: copy(p,Path('src/local-untracked')/dirname/rel)
    versions.append(dict(repository=dirname,local_head=git('rev-parse','HEAD').decode().strip(),
                         scope=roots,warning='Local snapshot; not the final server eight-card commit.'))
write('configs/local-source-versions.json',json.dumps(versions,ensure_ascii=False,indent=2))
write('patches/README.md','''# 补丁说明

这两份补丁针对 configs/local-source-versions.json 中的本地 HEAD。
InfiniLM 仅收集 csrc 改动，未混入无关的 llama 测试工作区改动。
git diff 不包含未跟踪文件；这些文件单独保存于 src/local-untracked，必须一并复核。
尚未在干净检出上验证补丁应用及重建。八卡装载修复由 artifacts/infinifabric-build-v1 的补丁脚本和 h3_weight_load_sync.hpp 提供。
本地旧工作树、v3源码包和服务器候选是不同版本来源，不可把它们当成同一个最终提交。
''')
write('.gitignore','''__pycache__/
*.py[cod]
.venv/
.env
*.whl
*.so
*.o
*.a
*.npz
*.npy
*.f32
*.mp4
models/
wheels/
downloads/
server-downloads/
build/
.xmake/
''')
write('README.md','''# MiniMax-H3 FL2VA 适配阶段提交目录

本目录是2026年11月8日从实际开发目录复制的可审阅材料，保留原开发目录，不是最终验收版本。
当前 Native 完整生成768×928、124帧；原版比较 Average SSIM 44.643%。Qwen第50层输出相对L2为71.698%，仍在排查。
九齿已有gfx906单卡门控残差接入及gfx936独立JIT证据，正式八卡模型中的九齿调用及收益尚未验证；没有整体加速结论。

## 目录

- artifacts/：保持原运行包内部路径的源码、补丁生成器、构建/运行/诊断脚本。infinifabric-build-v1 是当前八卡材料，其他包是历史单卡证据。
- src/：九齿v3和性能包解包后的文本源码，以及 git diff 未覆盖的新增实现。
- patches/：本地工作树改动和适用基线说明。
- scripts/、tests/：原有开发辅助程序和测试；旧Pod脚本不是八卡推荐入口。
- configs/：本地版本来源。服务器实际版本仍待补取。
- docs/：规则整理和关键结果记录。
- results/：本地已有的原始JSON、日志和抽帧。只作验证证据，不是推理输入缓存。
- reports/：阶段Word初稿，队伍、仓库和最终commit待填写。
- requirements/：依赖位置与下载边界说明。
- server-downloads/：只在本地收取服务器材料，已被Git忽略。

## 当前入口

在 artifacts/infinifabric-build-v1 查看 LOAD-SYNC-BUILD-UPLOAD.md、FULL-FORWARD-LOAD-SYNC-UPLOAD.md、LOAD-SYNC-GENERATION-UPLOAD.md、REFERENCE-COMPARE-UPLOAD.md。
生成入口为 run_minimax_h3_load_sync_generation.sh，参考比较为 run_minimax_h3_reference_compare.sh；diagnose_* 为诊断，不是正式生成。
包说明包含旧版本历史，真实调度完成门控以当前v1.2代码及回执为准。
各作业分开提交，申请28分钟；模型和wheel放AFS，不放平台项目输入目录。
现有候选路径及回执存在绑定，尚不能宣称本目录能一键干净重建；需补服务器实际源码、候选构建来源并验证。

## Git状态

本次仅整理文件，没有执行 git add、commit 或推送，没有更改仓库可见性。
先核对 docs/WEBSITE-DOWNLOADS.md 和 manifests，再选择提交文件。
模型、wheel、私密凭据、大型中间张量和视频不进Git。源码包保留是为了固定现有离线构建输入。
第三方代码遵守各自原许可证，本队许可证和最终依赖许可清单待确定。
''')
write('requirements/README.md','''# 离线依赖

八卡wheel输入位置及版本见 artifacts/infinifabric-build-v1 的 *wheels*.txt、*inventory*.json、下载PS1和运行包说明。
服务器个人AFS的 packages、wheels、python-envs 继续保留；不复制到Git。
待下载的 pip freeze 只是该解释器的快照，还必须同时记录私有site-packages暂存状态、厂商Torch/DTK版本和wheel SHA256。
服务器setup_infinifabric_env.sh和state/*.json属于本地取证材料，公开前去除个人绝对路径并核对敏感项。
''')
for side in ['native','reference']: (OUT/'server-downloads/input-boundaries'/side).mkdir(parents=True,exist_ok=True)
write('docs/WEBSITE-DOWNLOADS.md','''# 从运行网站下载材料

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
''')
write('docs/ORGANIZATION-PLAN.md','''# 目录整理记录 2026年11月8日

按已约定的独立提交目录实施：复制源码与运行包→保存本地补丁及新增文件→归档关键证据和报告→列服务器下载缺口→检查文件摘要。
保留原目录；不提交、不推送、不改变可见性。二进制依赖与模型不打包。当前是开发快照，不是已验收的最终提交。
''')
write('manifests/copied-files.json',json.dumps(records,ensure_ascii=False,indent=2))
files=[p for p in OUT.rglob('*') if p.is_file() and '.git' not in p.relative_to(OUT).parts]
write('manifests/SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in sorted(files)))
print(json.dumps(dict(files=len(files)+1,bytes=sum(p.stat().st_size for p in files),destination=str(OUT)),ensure_ascii=False))
