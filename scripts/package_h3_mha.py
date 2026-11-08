#!/usr/bin/env python3
"""Local maintainer utility: reproducibly assemble the offline MHA patch bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT / 'worktrees/InfiniCore-h3-mha'
BASELINE = ROOT / 'repos/InfiniCore'
COMMIT = 'f32e2fbb2a9b2ca912b5711f4f3836c403196a74'
FILES = ['src/infinicore/ops/multi_head_attention/mha.cc',
         'src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc']


def git(repo, *args, allowed=(0,)):
    result = subprocess.run(['git', '-c', 'core.filemode=false', '-c', 'core.autocrlf=false',
                             '-c', 'core.excludesFile=', *args], cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout.replace(b'\r\n', b'\n')


def digest(data):
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def main():
    for repo in (BASELINE, WORKTREE):
        if git(repo, 'rev-parse', 'HEAD').decode().strip() != COMMIT:
            raise RuntimeError(f'Wrong baseline commit: {repo}')
    payload = git(WORKTREE, 'diff', 'HEAD', '--', FILES[0])
    payload += git(WORKTREE, 'diff', '--no-index', '--', '/dev/null', FILES[1], allowed=(1,))
    if not payload:
        raise RuntimeError('No patch was generated')
    package = ROOT / 'artifacts/h3-mha-infiniop-v1'
    package.mkdir(parents=True, exist_ok=True)
    patch_file = package / 'h3_mha_infiniop.patch'
    patch_file.write_bytes(payload)
    git(BASELINE, 'apply', '--check', str(patch_file))
    git(WORKTREE, 'apply', '--reverse', '--check', str(patch_file))
    manifest = {'baseline_commit': COMMIT, 'patch_sha256': digest(payload), 'files': {}}
    for name in FILES:
        original = BASELINE / name
        manifest['files'][name] = {'before': digest(original.read_bytes()) if original.exists() else None,
                                   'after': digest((WORKTREE / name).read_bytes())}
    (package / 'h3_mha_patch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                                     encoding='utf-8', newline='\n')
    for name in ('build_h3_mha.sh', 'apply_h3_mha_patch.py', 'inspect_h3_mha_registration.py',
                 'verify_h3_mha_infiniop.py'):
        (package / name).write_bytes((ROOT / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n'))
    shutil.copyfile(ROOT / 'docs/h3-mha-infiniop-baseline.md', package / 'README.md')
    archive = package.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for item in sorted(package.iterdir()):
            if item.is_file():
                output.write(item, f'{package.name}/{item.name}')
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed')
    print(f'Patch checks: forward on original and reverse on worktree passed.')
    print(f'Patch SHA256: {manifest["patch_sha256"]}')
    print(f'Archive: {archive} ({archive.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
