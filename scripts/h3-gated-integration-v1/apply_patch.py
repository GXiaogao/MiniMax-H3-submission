"""Install exact verified source payloads across both repositories; rollback write failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time


def digest(data):
    return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()


def safe_path(root,name):
    if '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
        raise ValueError('Unsafe manifest path')
    path=Path(root)/name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes its root')
    return path


def replace_file(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.h3-gated-tmp')
    if tmp.exists(): raise RuntimeError('Stale/concurrent temporary file: '+str(tmp))
    try:
        with tmp.open('xb') as f:f.write(data)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def install(repos,package,manifest,backup):
    states=[]
    seen=set()
    for item in manifest['prerequisites']:
        p=safe_path(repos[item['repo']],item['path'])
        if not p.is_file() or digest(p.read_bytes())!=item['sha256']:
            raise RuntimeError('Prerequisite source mismatch: '+str(p))
    for item in manifest['files']:
        key=(item['repo'],item['path'])
        if key in seen: raise ValueError('Duplicate target')
        seen.add(key)
        path=safe_path(repos[item['repo']],item['path'])
        payload=safe_path(package,item['repo']+'/'+item['path']).read_bytes()
        if digest(payload)!=item['after']:raise RuntimeError('Payload hash mismatch: '+str(path))
        before=path.read_bytes() if path.is_file() else None
        sha=digest(before) if before is not None else None
        if sha not in (item['before'],item['after']):raise RuntimeError('Unknown edit: '+str(path))
        states.append((item,path,before,payload,sha))
    if not states:raise ValueError('Empty patch')
    if all(s[4]==s[0]['after'] for s in states):return 'already-applied'
    if not all(s[4]==s[0]['before'] for s in states):raise RuntimeError('Partially applied source state')
    backup.mkdir(parents=True,exist_ok=False)
    for item,path,before,payload,sha in states:
        if before is not None:
            p=backup/item['repo']/item['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(before)
    (backup/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    written=[]
    try:
        for item,path,before,payload,sha in states:
            current=path.read_bytes() if path.is_file() else None
            if current!=before:raise RuntimeError('Concurrent source edit: '+str(path))
            replace_file(path,payload)
            written.append((path,before))
            if digest(path.read_bytes())!=item['after']:raise RuntimeError('Post-write hash mismatch')
    except Exception:
        for path,before in reversed(written):
            if before is None:path.unlink()
            else:replace_file(path,before)
        raise
    return 'applied'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    home=Path(os.environ['WORK_HOME']).resolve();package=Path(__file__).resolve().parent
    manifest=json.loads((package/'patch_manifest.json').read_text())
    repos={name:home/'Projects'/name for name in manifest['commits']}
    for name,commit in manifest['commits'].items():
        actual=subprocess.check_output(['git','-C',str(repos[name]),'rev-parse','HEAD'],text=True).strip()
        if actual!=commit:raise RuntimeError('Baseline commit mismatch: '+name)
    if args.check:
        for item in manifest['prerequisites']:
            p=safe_path(repos[item['repo']],item['path'])
            if digest(p.read_bytes())!=item['sha256']:raise RuntimeError('Prerequisite mismatch: '+str(p))
        for item in manifest['files']:
            p=safe_path(repos[item['repo']],item['path']);sha=digest(p.read_bytes()) if p.is_file() else None
            if sha not in (item['before'],item['after']):raise RuntimeError('Unknown edit: '+str(p))
        print('Source preflight OK')
        return
    backup=home/'logs'/('h3_gated_source_backup_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    print('Source:',install(repos,package/'payload',manifest,backup),'backup:',backup)


if __name__=='__main__':main()
