#!/usr/bin/env python3
"""Build the offline InfiniLM video-boundary source patch from its worktree."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT/'worktrees/InfiniLM-qwen-video'
BASE = '65deb91f840ebaef752d028fad2f3217ce0950f4'
NAMES = ['csrc/models/qwen3_5/qwen3_5_vision.cpp',
         'csrc/models/qwen3_5/qwen3_5_vision.hpp']


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.filemode=false',
                                    '-c', 'core.excludesFile=', '-C', str(WORKTREE), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if git('rev-parse', 'HEAD').decode().strip() != BASE:
        raise RuntimeError('Unexpected worktree baseline')
    git('diff', '--check')
    folder = ROOT/'artifacts/qwen3-vl-video-boundary-v1'
    folder.mkdir(parents=True, exist_ok=True)
    patch = git('diff', '--no-ext-diff', '--binary', '--no-color', BASE, '--', *NAMES).replace(b'\r\n', b'\n')
    if not patch:
        raise RuntimeError('Empty native patch')
    (folder/'qwen_video_boundaries.patch').write_bytes(patch)
    manifest = {'baseline_commit': BASE, 'patch_sha256': sha(patch), 'files': {}}
    for name in NAMES:
        before = git('show', f'{BASE}:{name}').replace(b'\r\n', b'\n')
        after = (WORKTREE/name).read_bytes().replace(b'\r\n', b'\n')
        manifest['files'][name] = {'before': sha(before), 'after': sha(after)}
    (folder/'qwen_video_patch_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    for name in ['apply_qwen_video_patch.py', 'build_qwen_video.sh', 'check_qwen_video_layout.py']:
        (folder/name).write_bytes((ROOT/'scripts'/name).read_bytes().replace(b'\r\n', b'\n'))
    (folder/'README.md').write_bytes((ROOT/'docs/qwen3-vl-video-boundary-fix.md').read_bytes().replace(b'\r\n', b'\n'))
    members = sorted(p for p in folder.iterdir() if p.is_file() and p.name != 'bundle_manifest.json')
    (folder/'bundle_manifest.json').write_text(json.dumps({
        'package': folder.name, 'native_device_validation': 'pending',
        'files': {p.name: sha(p.read_bytes()) for p in members}}, indent=2)+'\n', encoding='utf-8', newline='\n')
    members.append(folder/'bundle_manifest.json')
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(members):
            info = zipfile.ZipInfo(folder.name+'/'+path.name, date_time=(2026,10,25,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, path.read_bytes())
    (archive.with_suffix('.zip.sha256')).write_text(sha(archive.read_bytes())+'  '+archive.name+'\n', encoding='ascii')
    print(archive)
    print('sha256:', sha(archive.read_bytes()))


if __name__ == '__main__':
    main()
