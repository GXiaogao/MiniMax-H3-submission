#!/usr/bin/env python3
"""Snapshot the additive source patch and LF scripts, then create reproducible tar.gz."""
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'scripts/h3-gated-integration-v1'
REPOS={'InfiniCore':ROOT/'worktrees/InfiniCore-h3-mha','InfiniLM':ROOT/'worktrees/InfiniLM-qwen-video'}
FILES={'InfiniCore':['include/infinicore/ops/gated_residual.hpp','include/infinicore/ops.hpp',
       'src/infinicore/ops/gated_residual/gated_residual.cc','src/infinicore/ops/gated_residual/gated_residual_hygon.cc',
       'src/infinicore/pybind11/ops/gated_residual.hpp','src/infinicore/pybind11/ops.hpp',
       'python/infinicore/ops/gated_residual.py','python/infinicore/__init__.py'],
       'InfiniLM':['csrc/models/minimax_h3/minimax_h3_transformer.cpp']}
PREREQUISITES={'InfiniCore':['src/infinicore/ops/multi_head_attention/mha.cc','src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc'],
 'InfiniLM':['csrc/models/qwen3_5/qwen3_5_vision.cpp','csrc/models/qwen3_5/qwen3_5_vision.hpp']}


def data(path):return path.read_bytes().replace(b'\r\n',b'\n')
def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    manifest={'commits':{'InfiniCore':'f32e2fbb2a9b2ca912b5711f4f3836c403196a74','InfiniLM':'65deb91f840ebaef752d028fad2f3217ce0950f4'},'files':[],'prerequisites':[]}
    paths=[];diff=[]
    for repo,names in FILES.items():
        for name in names:
            before=PACKAGE/'before'/repo/name;old=data(before) if before.exists() else None
            new=data(REPOS[repo]/name);p=PACKAGE/'payload'/repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new);paths.append(p)
            manifest['files'].append({'repo':repo,'path':name,'before':sha(old) if old is not None else None,'after':sha(new)})
            diff.extend(difflib.unified_diff((old or b'').decode().splitlines(keepends=True),new.decode().splitlines(keepends=True),fromfile='a/'+repo+'/'+name if old is not None else '/dev/null',tofile='b/'+repo+'/'+name))
    for repo,names in PREREQUISITES.items():
        for name in names:manifest['prerequisites'].append({'repo':repo,'path':name,'sha256':sha(data(REPOS[repo]/name))})
    (PACKAGE/'patch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (PACKAGE/'changes.patch').write_text(''.join(diff),encoding='utf-8',newline='\n')
    paths += [p for p in PACKAGE.iterdir() if p.is_file() and p.name!='SHA256SUMS']
    for p in paths:
        value=data(p)
        if b'\r' in value:raise ValueError('Unexpected CR: '+str(p))
        if p.read_bytes()!=value:p.write_bytes(value)
    sums=PACKAGE/'SHA256SUMS';sums.write_text(''.join(sha(p.read_bytes())+'  '+p.relative_to(PACKAGE).as_posix()+'\n' for p in sorted(paths)),encoding='ascii',newline='\n');paths.append(sums)
    output=ROOT/'artifacts/h3-gated-integration-v1.tar.gz'
    with output.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for p in sorted(paths):
            b=p.read_bytes();entry=tarfile.TarInfo('h3-gated-integration-v1/'+p.relative_to(PACKAGE).as_posix());entry.size=len(b);entry.mtime=0;entry.mode=0o755 if p.suffix=='.sh' else 0o644;tar.addfile(entry,io.BytesIO(b))
    with tarfile.open(output,'r:gz') as tar:
        for line in tar.extractfile('h3-gated-integration-v1/SHA256SUMS').read().decode().splitlines():
            want,name=line.split('  ');actual=tar.extractfile('h3-gated-integration-v1/'+name).read()
            if sha(actual)!=want or b'\r' in actual:raise RuntimeError('Archive mismatch')
    print(output);print('SHA256:',sha(output.read_bytes()));print('Members:',len(paths),'; server build/validation pending')


if __name__=='__main__':main()
