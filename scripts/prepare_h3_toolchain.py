#!/usr/bin/env python3
"""Download fixed Linux CPython 3.11 toolchain wheels; install nothing locally."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile

TRITON = 'triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
TRITON_URL = 'https://download.sourcefind.cn:65024/file/4/triton/DAS1.8/' + TRITON
PACKAGES = {'ninetoothed': '0.26.0', 'numpy': '1.26.4', 'sympy': '1.14.0', 'mpmath': '1.3.0'}


def open_url(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)


def candidates():
    yield TRITON, TRITON_URL, None
    for name, version in PACKAGES.items():
        with open_url(f'https://pypi.org/pypi/{name}/{version}/json') as response:
            release = json.load(response)
        files = [item for item in release['urls'] if (
            item['filename'].endswith('-py3-none-any.whl') or
            (name == 'numpy' and '-cp311-cp311-' in item['filename'] and
             'manylinux_2_17_x86_64' in item['filename']))]
        if len(files) != 1:
            raise RuntimeError(f'{name}: expected one matching Linux/portable wheel, got {len(files)}')
        item = files[0]
        yield item['filename'], item['url'], item['digests']['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url, published_hash in candidates():
        path = args.dest / filename
        if not path.is_file():
            temporary = path.with_suffix('.whl.partial')
            print(f'Downloading {filename}', flush=True)
            for attempt in range(3):
                try:
                    with open_url(url) as response, temporary.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    print(f'Retrying {filename}', flush=True)
                    time.sleep(2)
            path_to_check = temporary
        else:
            path_to_check = path
        digest = hashlib.sha256(path_to_check.read_bytes()).hexdigest()
        if published_hash and digest != published_hash:
            raise RuntimeError(f'{filename}: published SHA256 mismatch')
        with zipfile.ZipFile(path_to_check) as wheel:
            metadata_names = [name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise RuntimeError(f'{filename}: invalid wheel metadata')
            metadata = wheel.read(metadata_names[0]).decode()
            print('\n'.join(line for line in metadata.splitlines() if line.startswith(('Name:', 'Version:', 'Requires-Dist:'))), flush=True)
        if path_to_check != path:
            path_to_check.replace(path)
        records.append({'file': filename, 'url': url, 'sha256': digest,
                        'publisher_sha256': published_hash})
        print(f'SHA256 {digest}', flush=True)
    manifest = args.dest / 'h3_toolchain_manifest.json'
    manifest.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Ready: {manifest}. No packages have been installed.', flush=True)


if __name__ == '__main__':
    main()
