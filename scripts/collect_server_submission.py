"""Read-only source/environment collection. Outputs only under the new run folder."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

EXTENSIONS={'.c','.cc','.cpp','.cxx','.h','.hpp','.cu','.cuh','.hip','.py','.pyi',
            '.lua','.cmake','.toml','.cfg','.ini','.json','.sh','.txt','.md','.yaml','.yml','.in'}
SKIP={'.git','__pycache__','.xmake','.venv','build','dist','models','weights','node_modules'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def command(args, cwd=None, timeout=45):
    try:
        p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=timeout,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_TERMINAL_PROMPT':'0',
                                'PYTHONDONTWRITEBYTECODE':'1'})
        return {'command':args,'exit_code':p.returncode,
                'stdout':p.stdout.decode('utf-8','replace'),'stderr':p.stderr.decode('utf-8','replace')}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'command':args,'exit_code':None,'error':str(e)}

def collect(work, out, probe=True):
    work=Path(work).resolve();out=Path(out).resolve()
    sources={'InfiniCore':work/'Projects/InfiniCore','InfiniLM':work/'Projects/InfiniLM'}
    marker=work/'state/infinilm-load-sync.success'
    if marker.is_file():
        values=dict(line.split('=',1) for line in marker.read_text(errors='replace').splitlines() if '=' in line)
        ext=Path(values.get('extension',''))
        # Candidate source must be under the user's work directory.
        for parent in ext.parents:
            if parent.name=='InfiniLM' and parent.is_relative_to(work): sources['InfiniLM-load-sync']=parent;break
    for src in sources.values():
        if out==src or out.is_relative_to(src): raise ValueError('Output must not be within a collected source tree')
    if out.exists() or out.with_suffix('.zip').exists(): raise ValueError('Output already exists')
    out.mkdir(parents=True)
    report={'scope':'collection-not-build-or-inference-validation','python':sys.executable,
            'python_version':sys.version,'work_home':str(work),'missing_sources':[],
            'sources':{},'skipped':[],'source_errors':[],'limits':{'file_bytes':2*1024**2,'total_source_bytes':150*1024**2}}
    total=0
    def save(name,obj):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,src in sources.items():
        if not src.is_dir(): report['missing_sources'].append(name);continue
        files=[]
        for folder,dirs,names in os.walk(src,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in SKIP and not (Path(folder)/d).is_symlink())
            for filename in sorted(names):
                p=Path(folder)/filename;rel=p.relative_to(src)
                if p.is_symlink() or not (p.suffix.lower() in EXTENSIONS or filename in {'LICENSE','COPYING','NOTICE','Makefile'}):continue
                try:
                    size=p.stat().st_size
                    if size>2*1024**2 or total+size>150*1024**2:
                        report['skipped'].append(f'{name}/{rel}');continue
                    dest=out/'sources'/name/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(p,dest);total+=size;files.append(rel.as_posix())
                except OSError as e:report['source_errors'].append({'file':str(p),'error':str(e)})
        git={}
        if (src/'.git').exists():
            deleted=command(['git','ls-files','--deleted'],cwd=src)
            paths=files+[p for p in deleted.get('stdout','').splitlines() if Path(p).suffix.lower() in EXTENSIONS]
            for key,args in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
                             'status':['status','--porcelain=v1'],
                             'submodules':['submodule','status','--recursive'],
                             'diff':['diff','--no-ext-diff','--no-textconv','HEAD','--',*paths]}.items():
                git[key]=command(['git',*args],cwd=src)
            for key,result in git.items():
                if result.get('exit_code')!=0: report['source_errors'].append({'repository':name,'git_command':key,'error':result.get('error',result.get('stderr'))})
        else:git={'status':'NO_GIT_METADATA; do not infer a commit from local Windows repositories'}
        save(Path('versions')/(name+'.json'),{'source':str(src),'files':len(files),'git':git})
        report['sources'][name]={'path':str(src),'files':len(files)}
        print('SOURCE_COLLECTED',name,len(files),flush=True)
    for filename in ['setup_infinifabric_env.sh','state/infinicore-python.success','state/infinilm-extension.success',
                     'state/infinilm-load-sync.success','state/infinilm-runtime-private.json',
                     'state/minimax-h3-hygon-runtime.json','state/ninetoothed-gfx936.json']:
        p=work/filename
        if p.is_file() and not p.is_symlink():
            dest=out/'runtime'/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    packages=sorted([{'name':d.metadata.get('Name'),'version':d.version} for d in importlib.metadata.distributions()],key=lambda d:str(d['name']))
    save(Path('environment/packages.json'),packages)
    if probe:
        for name,args in {'pip-freeze':[sys.executable,'-m','pip','freeze','--all'],
                         'hardware':['rocm-smi','--showproductname','--showmeminfo','vram'],
                         'dtk':['hipcc','--version'],
                         'torch':[sys.executable,'-c','import json,torch; print(json.dumps(dict(torch=torch.__version__,hip=torch.version.hip,devices=torch.cuda.device_count())))']}.items():
            print('PROBE_START',name,flush=True);save(Path('environment')/(name+'.json'),command(args,timeout=60))
    # Record offline archives and wheel hashes, never model weights.
    inputs=[]; started=time.monotonic(); read_bytes=0
    for dirname in ['packages','wheels']:
        base=work/dirname
        for folder,dirs,names in os.walk(base,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
            for name in sorted(names):
                if not (name.endswith('.whl') or name in {'InfiniCore-infinifabric-v3.tar.gz','InfiniLM-infinifabric-v3.tar.gz'}):continue
                p=Path(folder)/name
                if p.is_symlink():continue
                try:
                    size=p.stat().st_size
                    item={'file':str(p.relative_to(work)),'bytes':size}
                    if read_bytes+size <= 4*1024**3 and time.monotonic()-started<300:
                        item['sha256']=digest(p);read_bytes+=size
                    else:item['hash_status']='SKIPPED_BUDGET'
                    inputs.append(item)
                except OSError as e:report['source_errors'].append({'file':str(p),'error':str(e)})
            if time.monotonic()-started>330:break
    save(Path('environment/offline-inputs.json'),inputs)
    report['result']='COLLECTED_WITH_GAPS' if report['missing_sources'] or report['skipped'] or report['source_errors'] else 'COLLECTED'
    save(Path('report.json'),report)
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(out).as_posix()}\n' for p in files),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out))
    archive_hash=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n',encoding='utf-8')
    # Browser transfer limit is 128MiB; each part stays below it.
    if archive.stat().st_size>100*1024**2:
        with archive.open('rb') as f:
            i=0
            while True:
                data=f.read(100*1024**2)
                if not data: break
                part=archive.with_name(archive.name+f'.part-{i:03d}');part.write_bytes(data);i+=1
    print('COLLECTION_RESULT='+report['result'],flush=True)
    print('DOWNLOAD='+str(archive),flush=True)
    print('SHA256='+archive_hash,flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work-home',type=Path);args=ap.parse_args()
    work=args.work_home or Path(os.environ['IFAB_AFS_HOME'])/'minimax-h3'
    run='infinifabric-source-environment-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    collect(work,work/'logs'/run)
