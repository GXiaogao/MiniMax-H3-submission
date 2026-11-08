#!/usr/bin/env python3
"""Repair the proven DTK 26.04 property ABI in the isolated Triton 3.8 candidate."""
import difflib
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from datetime import datetime
import traceback
import uuid
import zipfile

BASE_SHA = '921ac82f0170b5405c131395b2900c6fdab84484da96c98cf1d688610007fc7c'
WHEEL = 'triton-3.8.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
WHEEL_SHA = '68988ac85d5e7086baeda0ddc175af9667db7529b3c5e11a5c0601b8bef2200a'
# The exact diagnostic shipped before this repair; computed from local source.
DIAG_SHA = '532b762f1c272d1b0fe1f721584a4d008fdcd104967beb04739709fb6186d030'
RUNTIME_VERSION = 60326113
HEADERS = {'include/hip/hip_runtime_api.h': 'f0ed814c4f9cd8bcf8905f358acda397ffbcbb8395bb1a65973c5b547ec455c0',
           'include/hip/hip_deprecated.h': '9334c9333f6e28a3c385b62b29d003d69ed798c3f3a6bf94ca4d6f3b7ed9b33d',
           'include/hip/hip_version.h': 'f2f49e44ccd0509582566a32ff9cf1b104e45b3a5dfc2356208b8f52b7b6c3de',
           'driver.py': '24b908043301e5dfcdbc4cdb17b9b7313e650cdfd4098b2ebfb998af8994be4e'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_inside(path, root):
    if not Path(path).resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError(f'Refusing path outside candidate venv: {path}')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected source context; refusing approximate patch')
    return source.replace(old, new, 1)


def patched_source(original):
    if digest(original) != BASE_SHA:
        raise RuntimeError('Unknown Triton driver.c; refusing to patch')
    source = original.decode('utf-8')
    source = replace_once(source, '#include <hip/hip_runtime_api.h>',
        '#include <hip/hip_runtime_api.h>\n#include <hip/hip_deprecated.h>\n#include <string.h>\n#include <limits.h>')
    source = replace_once(source, 'hipGetDeviceProperties, true, hipDeviceProp_t *prop,',
                          'hipGetDeviceProperties, true, hipDeviceProp_tR0000 *prop,')
    source = replace_once(source, '  if (hipVersion == -1)\n    return false;\n', '''  if (hipVersion == -1)
    return false;
  // H3 DTK compatibility: verified legacy property ABI, not a general ROCm patch.
  if (hipVersion != 60326113) {
    PyErr_SetString(PyExc_RuntimeError,
                    "H3 HIP property patch requires the verified DTK runtime 60326113");
    dlclose(lib);
    return false;
  }
''')
    source = replace_once(source, '''  hipDeviceProp_t props;
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipGetDeviceProperties(&props, device_id));
''', '''  // DTK returns R0000. Reserve room for either known ABI to bound a mismatch.
  union {
    hipDeviceProp_tR0000 legacy;
    hipDeviceProp_t current;
  } storage = {0};
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipGetDeviceProperties(&storage.legacy, device_id));
  hipDeviceProp_tR0000 props = storage.legacy;
  if (!memchr(props.gcnArchName, 0, sizeof(props.gcnArchName)) ||
      !props.gcnArchName[0] || (props.warpSize != 32 && props.warpSize != 64) ||
      props.multiProcessorCount <= 0 || props.sharedMemPerBlock == 0 ||
      props.sharedMemPerBlock > INT_MAX) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid HIP legacy device properties");
    return NULL;
  }
''')
    source = replace_once(source, '      props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,',
                          '      (int)props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,')
    return source.encode('utf-8')


def validate_probe(report):
    try:
        assert report['collection_complete'] is True and not report.get('worker_exit_code', 0)
        assert report['torch']['version'].split('+')[0] == '2.9.0'
        assert report['torch']['hip'] == '6.3.26093'
        arch = report['torch']['arch']
        assert arch.split(':')[0] == 'gfx906'
        library = report['selected_hip_library']['realpath']
        assert library.startswith('/opt/dtk-26.04/hip/lib/libgalaxyhip.so.')
        for name in ('hipDriverGetVersion', 'hipRuntimeGetVersion'):
            assert report['versions'][name] == {'rc': 0, 'value': RUNTIME_VERSION}
        for key in ('TRITON_OVERRIDE_ARCH', 'HSA_OVERRIDE_GFX_VERSION'):
            assert not report['overrides'][key]
        assert report['get_proc_address']['rc'] == 0 and report['get_proc_address']['status'] == 0
        addresses = report['addresses']
        assert addresses['dlsym:hipGetDeviceProperties'] is not None
        assert addresses['dlsym:hipGetDeviceProperties'] == addresses['hipGetProcAddress:hipGetDeviceProperties']
        assert addresses['dlsym:hipGetDevicePropertiesR0600'] is None
        for label in ('dlsym:hipGetDeviceProperties', 'hipGetProcAddress:hipGetDeviceProperties'):
            row = report['queries'][label]
            assert row['rc'] == 0 and row['guard_ok'] is True
            assert row['layout_evidence'] == 'R0000_matches_only'
            old = row['R0000']
            assert old['arch']['text'] == arch and old['arch']['valid_ascii'] and old['arch']['nul_terminated']
            assert old['warp'] == 64 and old['multiprocessors'] > 0 and old['shared_memory'] > 0
        for name, expected in HEADERS.items():
            assert report['source_hashes'][name] == expected
    except (AssertionError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError('Fresh ABI evidence does not match the verified DTK candidate; no patch allowed') from exc


def write_atomic(path, data):
    mode = stat.S_IMODE(path.stat().st_mode)
    tmp = path.with_name(path.name + '.h3-' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.chmod(mode)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def transaction(driver, original, folder, verify):
    patched = patched_source(original)
    before = driver.read_bytes()
    if before not in (original, patched):
        raise RuntimeError('Candidate driver has unknown edits; no files changed')
    for name, value in (('driver.c.original', original), ('driver.c.before', before),
                        ('driver.c.patched', patched)):
        with (folder / name).open('xb') as stream:
            stream.write(value)
    diff = ''.join(difflib.unified_diff(original.decode().splitlines(True), patched.decode().splitlines(True),
                                       fromfile='upstream/driver.c', tofile='h3-dtk/driver.c'))
    (folder / 'driver.patch').write_text(diff, encoding='utf-8', newline='\n')
    if driver.read_bytes() != before:
        raise RuntimeError('Source changed during backup; refusing concurrent write')
    try:
        if before != patched:
            write_atomic(driver, patched)
        result = verify()
        if not isinstance(result, dict) or result.get('passed') is not True:
            raise RuntimeError('Property post-check did not pass')
        return result
    except BaseException:
        current = driver.read_bytes()
        if current not in (before, patched):
            raise RuntimeError('Source changed concurrently; backup retained, refusing to overwrite it')
        if current == patched and before != patched:
            write_atomic(driver, before)
        raise


POSTCHECK = r'''
import json,sys,hashlib
from pathlib import Path
import torch,triton
from triton.backends.amd import driver
before=json.loads(Path(sys.argv[1]).read_text())
device=torch.cuda.current_device()
p=torch.cuda.get_device_properties(device)
t=triton.runtime.driver.active.get_current_target()
actual=triton.runtime.driver.active.utils.get_device_properties(device)
expected=before['queries']['hipGetProcAddress:hipGetDeviceProperties']['R0000']
assert t.backend=='hip' and str(t.arch)=='gfx906' and t.warp_size==64, str(t)
assert actual['arch']==p.gcnArchName==expected['arch']['text']
assert actual['warpSize']==expected['warp']==p.warp_size
assert actual['multiprocessor_count']==expected['multiprocessors']==p.multi_processor_count
# Torch 2.9 HIP does not expose shared_memory_per_block; use the fresh native probe.
assert actual['max_shared_mem']==expected['shared_memory']
source=Path(driver.__file__).with_name('driver.c')
assert hashlib.sha256(source.read_bytes()).hexdigest()==sys.argv[2]
result={'passed':True,'target':str(t),'properties':actual,'driver_sha256':sys.argv[2],
 'torch':torch.__version__,'torch_path':torch.__file__,'hip':torch.version.hip,
 'hip_library':driver._get_path_to_hip_runtime_dylib(),
 'loaded_hip_libraries':sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
   if 'libamdhip' in s or 'libgalaxyhip' in s}), 'gpu_kernels_validated':False}
assert result['torch']==before['torch']['version'] and result['torch_path']==before['torch']['path']
assert result['hip']==before['torch']['hip']
assert result['hip_library']==before['selected_hip_library']['path']
Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
'''


def child(command, folder, log_name, timeout):
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONOPTIMIZE'):
        env.pop(name, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['TRITON_CACHE_DIR'] = str(folder / (log_name + '-cache'))
    with (folder / log_name).open('wb') as output:
        p = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            raise
    print((folder / log_name).read_text(encoding='utf-8', errors='replace'), flush=True)
    if rc:
        raise RuntimeError(f'{log_name} failed: {rc}; see preserved log')


def main():
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 11) or sys.flags.optimize:
        raise RuntimeError('Requires Linux Python 3.11 without -O')
    work = Path(os.environ['WORK_HOME']).resolve()
    candidate = work / 'venvs/h3-ninetoothed-026-triton380'
    if Path(sys.prefix).resolve() != candidate.resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError('Run with the isolated h3-ninetoothed-026-triton380/bin/python')
    distribution = metadata.distribution('triton')
    if distribution.version != '3.8.0':
        raise RuntimeError('Expected candidate Triton 3.8.0')
    driver = Path(distribution.locate_file('triton/backends/amd/driver.c')).resolve()
    require_inside(driver, candidate)
    folder = work / 'logs' / ('h3_hip_compat_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    status = {'passed': False, 'scope': 'HIP property query only', 'gpu_kernels_validated': False,
              'script_sha256': file_hash(__file__), 'driver': str(driver)}
    print('REPAIR REPORT:', folder / 'repair.json', flush=True)
    # Lock only this isolated venv while its host driver source is updated.
    import fcntl
    with (candidate / '.h3-hip-compat.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wheel = work / 'wheels' / WHEEL
            if file_hash(wheel) != WHEEL_SHA:
                raise RuntimeError('Original wheel missing/corrupt; refusing source repair')
            with zipfile.ZipFile(wheel) as archive:
                original = archive.read('triton/backends/amd/driver.c')
            patched = patched_source(original)
            if driver.read_bytes() not in (original, patched):
                raise RuntimeError('Unknown installed driver changes; no patch allowed')
            diag = work / 'scripts/diagnose_h3_triton_hip.py'
            if file_hash(diag) != DIAG_SHA:
                raise RuntimeError('Diagnostic script differs from the reviewed version; upload the original file')
            probe = folder / 'before-probe'; probe.mkdir()
            child([sys.executable, '-X', 'faulthandler', '-u', str(diag), '--worker', str(probe)],
                  folder, 'before.log', 90)
            evidence = json.loads((probe / 'report.json').read_text())
            validate_probe(evidence)
            status.update(before_sha256=file_hash(driver), original_sha256=BASE_SHA,
                          patched_sha256=digest(patched), baseline_report=str(probe / 'report.json'))
            def postcheck():
                source = folder / 'postcheck.py'
                source.write_text(POSTCHECK, encoding='utf-8', newline='\n')
                path = folder / 'postcheck.json'
                child([sys.executable, '-X', 'faulthandler', '-u', str(source),
                       str(probe / 'report.json'), digest(patched), str(path)], folder, 'after.log', 90)
                return json.loads(path.read_text())
            status['postcheck'] = transaction(driver, original, folder, postcheck)
            status['passed'] = True
            print('HIP PROPERTY CHECK PASSED. Run the original NineToothed JIT checks next.', flush=True)
        except BaseException as error:
            status['error'] = f'{type(error).__name__}: {error}'
            traceback.print_exc()
        finally:
            status['installed_driver_sha256'] = file_hash(driver)
            (folder / 'repair.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
            archive = folder.with_suffix('.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
                for path in folder.rglob('*'):
                    if path.is_file() and path.name != 'layout' and path.suffix not in ('.so', '.pyc'):
                        package.write(path, path.relative_to(folder))
            print('REPAIR EVIDENCE ZIP:', archive)
            print('JIT / AOT / InfiniCore integration / H3 regression: NOT VALIDATED')
    return 0 if status['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
