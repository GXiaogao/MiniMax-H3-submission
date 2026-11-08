#!/usr/bin/env python3
"""Build a reproducible LF-only native-launch verification package."""
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/h3-gated-native-v1'
FILES = ('README.md', 'abi.py', 'common.py', 'gated_residual.py', 'jit_checks.py',
         'native.py', 'native_launcher.cpp', 'verify_native.py', 'run.sh')


def main():
    paths = [SOURCE / name for name in FILES]
    for p in paths:
        data = p.read_bytes().replace(b'\r\n', b'\n')
        if b'\r' in data:
            raise ValueError(f'Bare CR: {p}')
        if data != p.read_bytes():
            p.write_bytes(data)
    sums = SOURCE / 'SHA256SUMS'
    sums.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n'
                           for p in sorted(paths)), encoding='ascii', newline='\n')
    paths.append(sums)
    output = ROOT / 'artifacts/h3-gated-native-v1.tar.gz'
    with output.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for p in sorted(paths):
                data = p.read_bytes()
                info = tarfile.TarInfo('h3-gated-native-v1/' + p.name)
                info.size, info.mtime, info.mode = len(data), 0, 0o755 if p.suffix == '.sh' else 0o644
                tar.addfile(info, io.BytesIO(data))
    with tarfile.open(output, 'r:gz') as tar:
        for line in tar.extractfile('h3-gated-native-v1/SHA256SUMS').read().decode().splitlines():
            want, name = line.split('  ')
            actual = tar.extractfile('h3-gated-native-v1/' + name).read()
            if hashlib.sha256(actual).hexdigest() != want or b'\r' in actual:
                raise ValueError('Archive mismatch: ' + name)
    print(output)
    print('SHA256', hashlib.sha256(output.read_bytes()).hexdigest())
    print('Members', len(paths), '; archive verified; native GPU execution pending')


if __name__ == '__main__':
    main()
