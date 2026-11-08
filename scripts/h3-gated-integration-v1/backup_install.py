"""Preserve installed libraries before replacing any builds. No imports from the frameworks."""
import json
import os
from pathlib import Path
import shutil
from common import sha256

home=Path(os.environ['WORK_HOME']);target=Path(os.environ['LOG_DIR'])/'installed_backup';target.mkdir()
files=[('InfiniCore_cpp',Path(os.environ['INFINI_ROOT'])/'lib/libinfinicore_cpp_api.so')]
for repo,package in [('InfiniCore','infinicore'),('InfiniLM','infinilm')]:
    root=home/'Projects'/repo/'python'/package
    files.extend((repo+'/'+p.relative_to(root).as_posix(),p) for p in root.rglob('*.so'))
records=[]
for name,path in files:
    if not path.is_file():raise RuntimeError('Expected installed library missing: '+str(path))
    dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    records.append({'original':str(path),'backup':str(dest),'sha256':sha256(path)})
(target/'files.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print('Installed libraries backed up:',target)
