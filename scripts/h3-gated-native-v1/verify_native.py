#!/usr/bin/env python3
"""Standalone HIP C++ launch gate; does not install or patch InfiniCore/InfiniLM."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
import zipfile

from common import run_child, sha256, stage_passed, write_json


def run_device(report, folder, result_path):
    # Caches and all generated code remain within the evidence directory.
    home = folder / 'cache-home'
    home.mkdir()
    os.environ['HOME'] = str(home)
    os.environ['TRITON_CACHE_DIR'] = str(folder / 'triton-cache')
    import jit_checks
    from native import Adapter, build_adapter, export_kernel
    report['environment'] = jit_checks.environment()
    write_json(result_path, report)
    import torch
    import numpy as np
    binary, hip_path = build_adapter(folder)
    report['adapter'] = {'path': str(binary), 'sha256': sha256(binary), 'hip_library': str(hip_path)}
    adapter = Adapter(binary, hip_path)
    instances = {}

    def factory(handle, generated_folder):
        dtype_name = handle._kernel.__name__.removeprefix('h3_gated_residual_')
        export = export_kernel(handle, dtype_name, folder / ('export_' + dtype_name))
        native_handle = adapter.open(export)
        jit, jit_proof = jit_checks.launch_with_explicit_rounding(handle, generated_folder)
        proof = {'native_export': export, 'jit_reference': jit_proof, 'native_calls': 0,
                 'jit_comparisons': 0, 'streams': []}

        def launch(residual, gate, branch, out):
            # Separate output for JIT. The native output is computed only by C++ -> HIP -> HSACO.
            jit_out = torch.empty_like(out)
            jit(residual, gate, branch, jit_out)
            stream = adapter.launch(native_handle, dtype_name, (residual, gate, branch, out))
            torch.cuda.synchronize()
            jit_checks.check_values(out, jit_out)
            proof['native_calls'] = adapter.lib.h3_calls(native_handle)
            proof['jit_comparisons'] += 1
            if stream not in proof['streams']:
                proof['streams'].append(stream)

        instances[dtype_name] = (launch, proof)
        return launch, proof

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # The 30 fixtures, CPU oracle, guard/input checks and tolerances are unchanged.
            jit_checks.run_gated(report, folder, result_path, launch_factory=factory)
            for dtype_name in jit_checks.DTYPES:
                dtype = getattr(torch, dtype_name)
                host = [torch.randn(3, 129).to(dtype) for _ in range(3)]
                residual, gate, branch = [x.cuda() for x in host]
                expected = (host[0].float() + (host[1].float() * host[2].float()).to(dtype).float()).to(dtype)
                launch, proof = instances[dtype_name]
                label = dtype_name + '_inplace_3x129'
                report['current_case'] = label
                write_json(result_path, report)
                launch(residual, gate, branch, residual)
                metrics = jit_checks.check_values(residual, expected)
                if not torch.equal(gate.cpu(), host[1]) or not torch.equal(branch.cpu(), host[2]):
                    raise AssertionError('Inplace update changed gate/branch')
                np.savez(folder / (label + '.npz'), actual=residual.float().cpu().numpy(), expected=expected.float().numpy())
                report['cases'].append({'name': label, 'passed': True, 'error': metrics,
                                         'gate_branch_unchanged': True, 'inplace_residual': True})
                print('PASS', label, metrics, flush=True)
                write_json(result_path, report)
        report['native_launch_count'] = sum(p['native_calls'] for _, p in instances.values())
        report['jit_comparison_count'] = sum(p['jit_comparisons'] for _, p in instances.values())
        if report['native_launch_count'] != 33 or report['jit_comparison_count'] != 33:
            raise AssertionError('Wrong native/JIT launch count')
        if any(not p['streams'] or 0 in p['streams'] for _, p in instances.values()):
            raise AssertionError('Non-default stream evidence missing')
    finally:
        adapter.close()


def worker(args):
    folder = args.folder
    result_path = folder / 'result.json'
    result = {'stage': 'native-gated', 'run_id': args.run_id, 'passed': False, 'cases': [], 'kernels': [],
              'infinicore_integrated': False, 'h3_regression_passed': False}
    try:
        run_device(result, folder, result_path)
        result['passed'] = True
        return 0
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        return 1
    finally:
        write_json(result_path, result)
        print('RESULT:', result_path, 'passed=', result['passed'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.worker:
        if not args.folder or not args.run_id:
            parser.error('worker needs folder and run-id')
        return worker(args)
    work_home = os.environ.get('WORK_HOME')
    if not work_home or args.timeout <= 0:
        parser.error('Source setup_env.sh first; timeout must be positive')
    expected = (Path(work_home) / 'venvs/h3-ninetoothed-026-triton380').absolute()
    if Path(sys.prefix).absolute() != expected:
        parser.error(f'Use {expected}/bin/python')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    folder = Path(work_home) / 'logs' / ('h3_gated_native_' + run_id)
    folder.mkdir(parents=True)
    summary = {'run_id': run_id, 'passed': False, 'scope': 'NineToothed generic HSACO + standalone HIP C++ launch',
               'infinicore_integrated': False, 'h3_regression_passed': False,
               'script_hashes': {p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.is_file()}}
    write_json(folder / 'summary.json', summary)
    print('REPORT:', folder / 'summary.json', flush=True)
    command = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
               '--worker', '--folder', str(folder), '--run-id', run_id]
    with (folder / 'native.log').open('w', encoding='utf-8') as output:
        rc = run_child(command, output, args.timeout)
    print((folder / 'native.log').read_text(encoding='utf-8', errors='replace'), flush=True)
    value = None
    try:
        value = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    passed = stage_passed(rc, value, 'native-gated', run_id, 33)
    passed = passed and value.get('native_launch_count') == 33 and value.get('jit_comparison_count') == 33
    summary.update(passed=passed, exit_code=rc, expected_cases=33)
    write_json(folder / 'summary.json', summary)
    members = [p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.so', '.pyc')]
    write_json(folder / 'files.json', [{'path': p.relative_to(folder).as_posix(), 'sha256': sha256(p),
                                      'size': p.stat().st_size} for p in members])
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in members + [folder / 'files.json']:
            z.write(p, folder.name + '/' + p.relative_to(folder).as_posix())
    print('EVIDENCE ZIP:', archive, flush=True)
    print('NATIVE CHECKS PASSED' if passed else 'NATIVE CHECKS FAILED', flush=True)
    print('InfiniCore / H3 integration: NOT VALIDATED', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
