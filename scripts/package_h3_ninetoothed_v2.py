#!/usr/bin/env python3
"""Create the LF-normalized, reproducible prerequisite toolchain package."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/h3-ninetoothed-v2'
FILES = ('README.md', 'common.py', 'gated_residual.py', 'prepare.py', 'setup.py',
         'setup.sh', 'run.sh', 'verify.py')


def main():
    paths = [SOURCE / name for name in FILES]
    for path in paths:
        data = path.read_bytes().replace(b'\r\n', b'\n')
        if b'\r' in data:
            raise ValueError(f'Unexpected bare CR: {path}')
        if path.read_bytes() != data:
            path.write_bytes(data)
    info = SOURCE / 'package_info.json'
    info.write_text(json.dumps({'schema': 1, 'name': 'h3-ninetoothed-v2',
        'candidate': 'ninetoothed026-triton380-gfx906',
        'scope': 'candidate JIT and original gated-residual expression verification',
        'device_tested': False, 'infinicore_integration_complete': False,
        'expected_add_cases': 9, 'expected_gated_cases': 30}, indent=2) + '\n',
        encoding='utf-8', newline='\n')
    paths.append(info)
    sums = SOURCE / 'SHA256SUMS'
    sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n'
                           for p in sorted(paths)), encoding='ascii', newline='\n')
    paths.append(sums)
    target = ROOT / 'artifacts/h3-ninetoothed-v2.tar.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as file, gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for path in sorted(paths):
                data = path.read_bytes()
                entry = tarfile.TarInfo('h3-ninetoothed-v2/' + path.name)
                entry.size = len(data)
                entry.mode = 0o755 if path.suffix == '.sh' else 0o644
                entry.mtime = 0
                archive.addfile(entry, io.BytesIO(data))
    with tarfile.open(target, 'r:gz') as archive:
        for line in archive.extractfile('h3-ninetoothed-v2/SHA256SUMS').read().decode().splitlines():
            expected, name = line.split('  ', 1)
            assert hashlib.sha256(archive.extractfile('h3-ninetoothed-v2/' + name).read()).hexdigest() == expected
    print(target)
    print('SHA256', hashlib.sha256(target.read_bytes()).hexdigest())
    print('Members', len(paths), '; package integrity PASS; device/C++ integration NOT TESTED')


if __name__ == '__main__':
    main()
