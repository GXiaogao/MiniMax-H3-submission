#!/usr/bin/env python3
"""Fetch a fixed Linux cp311 wheel set from PyPI. Install nothing locally."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from common import EXPECTED, CANDIDATE, MANIFEST, check_wheel, sha256, write_json


def open_url(url):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)


def choose_release(name, version):
    with open_url(f'https://pypi.org/pypi/{name}/{version}/json') as response:
        release = json.load(response)
    candidates = []
    for entry in release['urls']:
        filename = entry['filename']
        matches = filename.endswith('-py3-none-any.whl')
        if name in ('triton', 'numpy'):
            matches = '-cp311-cp311-' in filename and 'x86_64' in filename and 'manylinux' in filename
        if matches and not entry.get('yanked'):
            candidates.append(entry)
    if len(candidates) != 1:
        raise ValueError(f'{name}: expected one supported Linux cp311/portable wheel; found {len(candidates)}')
    item = candidates[0]
    return {'name': name, 'version': version, 'file': item['filename'],
            'url': item['url'], 'size': item['size'],
            'sha256': item['digests']['sha256'], 'publisher_sha256': item['digests']['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name, version in EXPECTED.items():
        entry = choose_release(name, version)
        path = args.dest / entry['file']
        if not path.is_file():
            tmp = path.with_name(path.name + '.partial')
            print('Downloading:', entry['file'], entry['size'], 'bytes', flush=True)
            with open_url(entry['url']) as response, tmp.open('wb') as output:
                while data := response.read(1024 * 1024):
                    output.write(data)
            if sha256(tmp) != entry['sha256']:
                raise ValueError(f'Published hash mismatch: {tmp}')
            tmp.replace(path)
        check_wheel(args.dest, entry)
        records.append(entry)
        print('Verified:', path.name, entry['sha256'], flush=True)
    write_json(args.dest / MANIFEST, {'schema': 2, 'candidate': CANDIDATE, 'wheels': records,
                                    'device_verified': False,
                                    'source': 'https://github.com/triton-lang/triton/releases/tag/v3.8.0'})
    print('Ready:', args.dest / MANIFEST, '; nothing installed locally.', flush=True)


if __name__ == '__main__':
    main()
