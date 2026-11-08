#!/usr/bin/env python3
"""Apply the exact Qwen video patch once; refuse to overwrite modified target files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def normalized(data):
    return data.replace(b'\r\n', b'\n')


def sha(data):
    return hashlib.sha256(normalized(data)).hexdigest()


def git(repo, *args, data=None):
    return subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'core.filemode=false',
                           '-c', 'core.excludesFile=', '-C', str(repo), *args],
                          input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def apply(repo, package, logs):
    repo, package, logs = map(lambda p: Path(p).resolve(), (repo, package, logs))
    manifest = json.loads((package / 'qwen_video_patch_manifest.json').read_text(encoding='utf-8'))
    payload = normalized((package / 'qwen_video_boundaries.patch').read_bytes())
    if sha(payload) != manifest['patch_sha256']:
        raise RuntimeError('patch checksum mismatch; upload the complete package again')
    commit = git(repo, 'rev-parse', 'HEAD').decode().strip()
    if commit != manifest['baseline_commit']:
        raise RuntimeError(f'baseline commit mismatch: {commit}')
    states = []
    for name, hashes in manifest['files'].items():
        path = repo / name
        if path.is_symlink() or not path.resolve().is_relative_to(repo):
            raise RuntimeError(f'target path escapes repository: {name}')
        current = sha(path.read_bytes()) if path.exists() else None
        states.append((current == hashes['before'], current == hashes['after']))
        if current not in (hashes['before'], hashes['after']):
            raise RuntimeError(f'target file was modified; will not overwrite: {name}')
    if not states:
        raise RuntimeError('empty patch manifest')
    if all(after for _, after in states):
        print('Qwen video patch already applied with matching file hashes; no source changes.')
        return 'already-applied'
    if not all(before for before, _ in states):
        raise RuntimeError('partial patch state mismatch; inspect the files before continuing')
    git(repo, 'apply', '--check', '-', data=payload)
    logs.mkdir(parents=True, exist_ok=True)
    record = logs / f'qwen_video_patch_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    record.mkdir()
    (record / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (record / 'source_before.patch').write_bytes(git(repo, 'diff', 'HEAD', '--', *manifest['files']))
    (record / 'applied.patch').write_bytes(payload)
    git(repo, 'apply', '-', data=payload)
    for name, hashes in manifest['files'].items():
        if sha((repo / name).read_bytes()) != hashes['after']:
            raise RuntimeError(f'post-apply checksum mismatch: {name}; inspect {record}')
    print(f'Qwen video patch applied. Baseline commit: {commit}; evidence: {record}')
    return 'applied'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    work = Path(os.environ['WORK_HOME']).resolve()
    logs = Path(os.environ['LOG_DIR']).resolve()
    if logs != work / 'logs':
        raise RuntimeError(f'LOG_DIR must be {work / "logs"}, got {logs}')
    apply(args.repo, Path(__file__).resolve().parent, logs)


if __name__ == '__main__':
    main()
