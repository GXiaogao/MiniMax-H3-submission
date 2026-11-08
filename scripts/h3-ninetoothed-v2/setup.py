#!/usr/bin/env python3
"""Install into a separate system-site-packages venv; never activate it globally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import EXPECTED, MANIFEST, check_wheel, validate_manifest


def run(command, env):
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Run in the Linux Python 3.11 competition Pod')
    entries = validate_manifest(json.loads((args.wheels / MANIFEST).read_text(encoding='utf-8')))
    wheel_paths = [str(check_wheel(args.wheels, entry)) for entry in entries]
    for entry in entries:
        print('SHA256 / metadata OK:', entry['file'], flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null')
    torch_probe = ('import torch,json; assert torch.__version__.split("+")[0]=="2.9.0"; '
                   'assert torch.version.hip; print(json.dumps([torch.__version__,torch.__file__,torch.version.hip]))')
    baseline = subprocess.check_output([sys.executable, '-c', torch_probe], env=env, text=True).strip()
    if not (args.venv / 'pyvenv.cfg').is_file():
        if args.venv.exists():
            raise RuntimeError(f'Refusing non-venv directory: {args.venv}')
        run([sys.executable, '-m', 'venv', '--system-site-packages', str(args.venv)], env)
    python = str(args.venv / 'bin/python')
    run([python, '-c', 'import sys,pathlib; assert sys.version_info[:2]==(3,11); '
         'assert sys.prefix!=sys.base_prefix; '
         'assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve()', str(args.venv)], env)
    installed_probe = '''import importlib.metadata as m,json,pathlib,sys
expected=json.loads(sys.argv[1]); root=pathlib.Path(sys.prefix).resolve()
for name,version in expected.items():
    d=m.distribution(name)
    assert d.version==version and pathlib.Path(d.locate_file('')).resolve().is_relative_to(root), name
'''
    current = subprocess.run([python, '-c', installed_probe, json.dumps(EXPECTED)],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if current.returncode:
        # --ignore-installed avoids uninstalling inherited base packages. The
        # explicit prefix/no-user prevents pip configuration redirecting writes.
        run([python, '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps',
             '--no-user', '--ignore-installed', '--prefix', str(args.venv), *wheel_paths], env)
    else:
        print('Pinned candidate packages already installed locally; skipping pip.', flush=True)
    run([python, '-c', installed_probe, json.dumps(EXPECTED)], env)
    after = subprocess.check_output([python, '-c', torch_probe], env=env, text=True).strip()
    if after != baseline:
        raise RuntimeError(f'Torch changed: before={baseline}; after={after}')
    base_after = subprocess.check_output([sys.executable, '-c', torch_probe], env=env, text=True).strip()
    if base_after != baseline:
        raise RuntimeError('Base Torch changed during setup')
    print('Torch unchanged:', baseline, flush=True)
    print('CANDIDATE INSTALLED. GPU execution and C++ integration are NOT validated.', flush=True)
    print('Candidate Python:', python, flush=True)


if __name__ == '__main__':
    main()
