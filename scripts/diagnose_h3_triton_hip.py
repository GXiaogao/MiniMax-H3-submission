#!/usr/bin/env python3
"""Collect HIP property ABI evidence; never install, patch, or validate kernels."""
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import traceback
import uuid
import zipfile


# Compile only a host layout printer. No GPU code and no HIP library linkage.
LAYOUT_C = r'''
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
#include <hip/hip_version.h>
#include <stddef.h>
#include <stdio.h>
#define FIELD(T, F) printf("[%zu,%zu]", offsetof(T,F), sizeof(((T*)0)->F))
#define LAYOUT(T) do { \
  printf("{\"size\":%zu,\"arch\":", sizeof(T)); FIELD(T,gcnArchName); \
  printf(",\"warp\":"); FIELD(T,warpSize); \
  printf(",\"multiprocessors\":"); FIELD(T,multiProcessorCount); \
  printf(",\"shared_memory\":"); FIELD(T,sharedMemPerBlock); printf("}"); \
} while(0)
int main(void) {
  printf("{\"header_version\":%d,\"R0000\":", HIP_VERSION);
  LAYOUT(hipDeviceProp_tR0000);
  printf(",\"R0600\":"); LAYOUT(hipDeviceProp_t);
  printf("}\n");
  return 0;
}
'''


def arch_value(raw):
    terminated = b'\0' in raw
    value = raw.split(b'\0', 1)[0]
    try:
        text = value.decode('ascii')
        valid = all(32 <= byte < 127 for byte in value)
    except UnicodeDecodeError:
        text, valid = None, False
    return {'hex': value.hex(), 'text': text, 'valid_ascii': valid,
            'nul_terminated': terminated}


def layout_evidence(row, torch_arch):
    if row.get('rc') != 0 or not row.get('guard_ok') or not torch_arch:
        return 'inconclusive'

    def matches(key):
        value = row[key]
        arch = value['arch']
        return (arch['valid_ascii'] and arch['nul_terminated'] and arch['text']
                and arch['text'].split(':')[0] == torch_arch.split(':')[0]
                and value['warp'] == 64 and value['multiprocessors'] > 0)

    old, new = matches('R0000'), matches('R0600')
    if old and not new:
        return 'R0000_matches_only'
    if new and not old:
        return 'R0600_matches_only'
    return 'ambiguous' if old and new else 'neither_matches'


def save(folder, report):
    temp = folder / 'report.json.tmp'
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temp.replace(folder / 'report.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def loaded_hip():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'libamdhip64' in line})


def collect(folder, report):
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use Linux Python 3.11 in the competition candidate venv')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use h3-ninetoothed-026-triton380/bin/python')
    report['overrides'] = {key: os.environ.get(key) for key in
                           ('TRITON_LIBHIP_PATH', 'TRITON_OVERRIDE_ARCH', 'HSA_OVERRIDE_GFX_VERSION')}
    if report['overrides']['TRITON_OVERRIDE_ARCH'] or report['overrides']['HSA_OVERRIDE_GFX_VERSION']:
        raise RuntimeError('Remove architecture overrides before diagnosing the actual device')
    import torch
    import triton
    if triton.__version__ != '3.8.0' or not Path(triton.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Expected isolated candidate Triton 3.8.0')
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('A working vendor HIP Torch device is required')
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    torch_arch = getattr(props, 'gcnArchName', '')
    report['torch'] = {'version': torch.__version__, 'path': torch.__file__, 'hip': torch.version.hip,
                       'device': device, 'properties': str(props), 'arch': torch_arch}
    report['hip_libraries_before'] = loaded_hip()
    save(folder, report)
    print('TORCH:', json.dumps(report['torch']), flush=True)

    from triton.backends.amd import driver
    base = Path(driver.__file__).parent
    report['triton'] = {'version': triton.__version__, 'driver': str(base)}
    report['source_hashes'] = {}
    evidence = folder / 'source-evidence'
    for name in ('driver.c', 'driver.py', 'include/hip/hip_runtime_api.h',
                 'include/hip/hip_deprecated.h', 'include/hip/hip_version.h'):
        src = base / name
        dst = evidence / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        report['source_hashes'][name] = sha(src)
    library = driver._get_path_to_hip_runtime_dylib()
    report['selected_hip_library'] = {'path': library, 'realpath': os.path.realpath(library)}
    print('HIP LIBRARY:', library, flush=True)
    save(folder, report)

    compiler = shutil.which('cc') or shutil.which('gcc')
    if not compiler:
        raise RuntimeError('Host C compiler not found; layout offsets must not be guessed')
    source, binary = folder / 'layout.c', folder / 'layout'
    source.write_text(LAYOUT_C, encoding='utf-8', newline='\n')
    command = [compiler, '-std=c11', '-I' + str(base / 'include'), str(source), '-o', str(binary)]
    report['layout_compile_command'] = command
    with (folder / 'layout_compile.log').open('wb') as out:
        compiled = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=45)
    if compiled.returncode:
        raise RuntimeError('Host layout compilation failed; see layout_compile.log')
    layout_output = subprocess.check_output([str(binary)], timeout=10)
    layouts = json.loads(layout_output)
    report['layouts'] = layouts
    print('HEADER LAYOUTS:', json.dumps(layouts), flush=True)
    save(folder, report)

    lib = ctypes.CDLL(library)
    report['versions'] = {}
    for name in ('hipDriverGetVersion', 'hipRuntimeGetVersion'):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        value = ctypes.c_int(-1)
        rc = fn(ctypes.byref(value))
        report['versions'][name] = {'rc': rc, 'value': value.value}
    # Match Triton 3.8's use of hipDriverGetVersion for hipGetProcAddress.
    version = report['versions']['hipDriverGetVersion']
    if version['rc'] != 0 or version['value'] <= 0:
        raise RuntimeError('Cannot obtain the HIP version used by Triton symbol resolution')
    addresses = {}
    for name in ('hipGetDeviceProperties', 'hipGetDevicePropertiesR0000', 'hipGetDevicePropertiesR0600'):
        try:
            addresses['dlsym:' + name] = ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
        except AttributeError:
            addresses['dlsym:' + name] = None
    get_proc = lib.hipGetProcAddress
    get_proc.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                        ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
    get_proc.restype = ctypes.c_int
    pointer, status = ctypes.c_void_p(), ctypes.c_int(-1)
    rc = get_proc(b'hipGetDeviceProperties', ctypes.byref(pointer), version['value'], 0, ctypes.byref(status))
    report['get_proc_address'] = {'rc': rc, 'status': status.value, 'address': pointer.value,
                                  'requested_version': version['value']}
    if rc == 0 and pointer.value:
        addresses['hipGetProcAddress:hipGetDeviceProperties'] = pointer.value
    report['addresses'] = {name: hex(value) if value else None for name, value in addresses.items()}
    report['queries'] = {}
    save(folder, report)

    # The buffer exceeds both header-defined structures; a sentinel tail detects
    # writes past the larger structure. No ctypes structure layout is guessed.
    maximum = max(layouts[key]['size'] for key in ('R0000', 'R0600'))
    if maximum > 8192:
        raise RuntimeError('Unexpected HIP structure size')
    for label, address in addresses.items():
        if not address:
            continue
        print('QUERY:', label, flush=True)
        report['active_query'] = label
        save(folder, report)
        storage = ctypes.create_string_buffer(16384)
        ctypes.memset(storage, 0xA5, len(storage))
        function = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)(address)
        rc = function(ctypes.cast(storage, ctypes.c_void_p), device)
        raw = storage.raw
        row = {'rc': rc, 'guard_ok': raw[maximum:] == b'\xA5' * (len(raw) - maximum)}
        for key in ('R0000', 'R0600'):
            view = {}
            for field in ('arch', 'warp', 'multiprocessors', 'shared_memory'):
                offset, size = layouts[key][field]
                value = raw[offset:offset + size]
                view[field] = arch_value(value) if field == 'arch' else int.from_bytes(value, sys.byteorder)
            row[key] = view
        row['layout_evidence'] = layout_evidence(row, torch_arch)
        report['queries'][label] = row
        (folder / (label.replace(':', '_') + '.bin')).write_bytes(raw)
        save(folder, report)
        print(json.dumps(row), flush=True)
        if not row['guard_ok']:
            raise RuntimeError('HIP wrote beyond the known layouts; stopping native queries')
    report.pop('active_query', None)
    save(folder, report)

    # Reproduce the public Triton failure only after collecting raw ABI evidence.
    try:
        report['triton_target'] = {'ok': True, 'value': str(triton.runtime.driver.active.get_current_target())}
    except Exception as exc:
        detail = {'ok': False, 'type': type(exc).__name__, 'message': str(exc)}
        if isinstance(exc, UnicodeDecodeError):
            detail['bytes_hex'] = exc.object.hex()
        report['triton_target'] = detail
    report['hip_libraries_after'] = loaded_hip()
    report['collection_complete'] = True
    print('TRITON TARGET:', json.dumps(report['triton_target']), flush=True)


def worker(folder):
    report = {'python': sys.executable, 'python_version': platform.python_version(),
              'host': platform.node(), 'toolchain_validated': False, 'collection_complete': False}
    save(folder, report)
    try:
        collect(folder, report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        save(folder, report)
    return 0 if report['collection_complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    root = args.log_dir
    if root is None:
        if not os.environ.get('WORK_HOME'):
            parser.error('source /root/private_data/setup_env.sh first (WORK_HOME missing)')
        root = Path(os.environ['WORK_HOME']) / 'logs'
    folder = (root / ('h3_hip_abi_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['TRITON_CACHE_DIR'] = str(folder / 'triton-cache')
    command = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()), '--worker', str(folder)]
    print('REPORT:', folder / 'report.json', flush=True)
    with (folder / 'diagnostic.log').open('wb') as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env,
                                start_new_session=sys.platform == 'linux')
        try:
            rc = proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            if sys.platform == 'linux':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            rc = 124
    print((folder / 'diagnostic.log').read_text(encoding='utf-8', errors='replace'))
    path = folder / 'report.json'
    report = json.loads(path.read_text()) if path.is_file() else {'collection_complete': False}
    report['worker_exit_code'] = rc
    report['toolchain_validated'] = False
    if rc:
        report['collection_complete'] = False
        report.setdefault('error', f'Worker terminated: {rc}')
    save(folder, report)
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in folder.rglob('*'):
            if file.is_file() and file.name != 'layout' and file.suffix not in ('.so', '.pyc'):
                z.write(file, file.relative_to(folder))
    print('EVIDENCE ZIP:', archive)
    print('Diagnostic only. No installation or source patch performed. Toolchain NOT validated.')
    return 0 if rc == 0 and report['collection_complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
