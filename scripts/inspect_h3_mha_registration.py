#!/usr/bin/env python3
"""Inspect the loaded Hygon MHA dispatchers without executing an MHA kernel.

Run through run_logged after setup_env.sh. A tiny host-only C++ helper is
compiled under /tmp, with best-effort cleanup that cannot replace the probe
result. No project source, installed library, build configuration, or Python
package is changed.

Exit 0: all three entries are present (NOT a correctness/performance pass).
Exit 1: at least one entry is missing. Exit 2: probe could not be completed.
"""

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HELPER_SOURCE = r'''
#include <infinicore/ops/mha.hpp>

extern "C" int h3_mha_registration_mask() noexcept {
    using Op = infinicore::op::MultiheadAttention;
    constexpr auto device = infinicore::Device::Type::HYGON;
    const auto plan = Op::plan_dispatcher().lookup(device);
    const auto run = Op::run_dispatcher().lookup(device);
    const auto cleanup = Op::cleanup_dispatcher().lookup(device);
    return (plan != nullptr ? 1 : 0)
         | (run != nullptr ? 2 : 0)
         | (cleanup != nullptr ? 4 : 0);
}
'''


def say(message):
    print(message, flush=True)


@contextmanager
def probe_workspace():
    # Avoid the persistent/shared scripts directory. A mapped helper library
    # may prevent removal on some filesystems; retain the diagnostic result.
    temporary = tempfile.TemporaryDirectory(prefix='.h3-mha-probe-', dir='/tmp')
    try:
        yield Path(temporary.name)
    finally:
        try:
            temporary.cleanup()
        except OSError as exc:
            say(f'PROBE CLEANUP WARNING: {temporary.name}: {exc}; probe result is unchanged')


def loaded_core_library():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        path = parts[5]
        if Path(path).name == 'libinfinicore_cpp_api.so':
            paths.add(Path(path).resolve())
        elif path.endswith('libinfinicore_cpp_api.so (deleted)'):
            raise RuntimeError('loaded InfiniCore library was deleted/replaced; start a fresh process')
    if len(paths) != 1:
        raise RuntimeError(f'expected exactly one loaded libinfinicore_cpp_api.so, found {sorted(map(str, paths))}')
    return paths.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('run this probe inside the Linux competition Pod')

    work = Path(os.environ.get('WORK_HOME', '/root/private_data'))
    prefix = Path(os.environ.get('INFINI_ROOT', '/root/.infini'))
    config = work / 'Projects/InfiniCore/.xmake/linux/x86_64/xmake.conf'
    config_text = config.read_text()
    awareness = re.search(r'\["mutual-awareness"\]\s*=\s*(true|false)', config_text)
    if awareness is None:
        raise RuntimeError(f'cannot determine dispatcher layout flag from {config}')
    say(f'python: {sys.executable}')
    say(f'config snapshot: {config}')
    for line in config_text.splitlines():
        if any(key in line for key in ('flash-attn', 'hygon-dcu', 'hygon-arch', 'mutual-awareness', 'ninetoothed')):
            say(line.strip())

    # Public package import loads its canonical extension and normal libraries.
    # Do not load the extension again under a different module name.
    import torch
    import infinicore

    say(f'torch: {torch.__version__}; HIP: {torch.version.hip}')
    say(f'infinicore: {infinicore.__file__}')
    library = loaded_core_library()
    say(f'loaded library: {library}')
    say(f'library SHA256: {hashlib.sha256(library.read_bytes()).hexdigest()}')
    for name in ('flash_attn', 'flash_attn_2_cuda', 'ninetoothed', 'ntops', 'triton'):
        spec = importlib.util.find_spec(name)
        say(f'{name}: {spec.origin if spec is not None else "NOT FOUND"}')

    include = prefix / 'include'
    for header in ('infinicore/ops/mha.hpp', 'infinicore/ops/common/dispatcher.hpp'):
        if not (include / header).is_file():
            raise RuntimeError(f'missing installed header: {include / header}')
    # Lookup is inlined from this header. A different header could read the
    # wrong layout, so require agreement with the source of the configuration.
    header = Path('infinicore/ops/common/dispatcher.hpp')
    source_header = work / 'Projects/InfiniCore/include' / header
    normalized = lambda p: p.read_bytes().replace(b'\r\n', b'\n')
    if normalized(source_header) != normalized(include / header):
        raise RuntimeError('installed dispatcher header differs from configured source')
    # Also check the configured build artifact against the actually loaded
    # library, so the feature flag above is not silently applied to another build.
    built = work / 'Projects/InfiniCore/build/linux/x86_64/release/libinfinicore_cpp_api.so'
    if not built.is_file() or hashlib.sha256(built.read_bytes()).digest() != hashlib.sha256(library.read_bytes()).digest():
        raise RuntimeError('loaded library differs from the configured release build; resolve library selection before interpreting dispatcher layout')
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('g++ is unavailable; this diagnostic needs only the existing host C++ compiler')

    with probe_workspace() as folder:
        source = folder / 'probe.cpp'
        output = folder / 'probe.so'
        source.write_text(HELPER_SOURCE, encoding='utf-8')
        command = [compiler, '-std=c++17', '-shared', '-fPIC', '-O0',
                   f'-I{include}', str(source), str(library),
                   f'-Wl,-rpath,{library.parent}', '-o', str(output)]
        if awareness.group(1) == 'true':
            command.insert(1, '-DENABLE_MUTUAL_AWARENESS')
        say('Compiling a host-only registration probe (no project rebuild).')
        subprocess.run(command, check=True)
        helper = ctypes.CDLL(str(output))
        probe = helper.h3_mha_registration_mask
        probe.argtypes = []
        probe.restype = ctypes.c_int
        mask = probe()
        for label, bit in (('plan', 1), ('run', 2), ('cleanup', 4)):
            say(f'HYGON MHA {label}: {"PRESENT" if mask & bit else "MISSING"}')
        if mask != 7:
            say('RESULT: MHA backend registration is incomplete. Do not call this MHA path until repaired.')
            return 1
        say('RESULT: all entries are present. Registration check passed; operator correctness tests are still required.')
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        say(f'PROBE ERROR: {type(exc).__name__}: {exc}')
        sys.exit(2)
