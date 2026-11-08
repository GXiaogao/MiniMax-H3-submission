#!/usr/bin/env python3
"""Actual NineToothed-generated add JIT; Torch is the test/reference harness."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
from pathlib import Path
import shutil
import traceback


def arrangement(left, right, output):
    return left.tile((128,)), right.tile((128,)), output.tile((128,))


def application(left, right, output):
    output = left + right


def numpy_compatible(version):
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:$|\+)', version)
    return bool(match) and tuple(map(int, match.groups())) >= (1, 26, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'vendor', 'run-dir', 'baseline'):parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = {'result': 'FAILED', 'cases': [], 'last_stage': 'environment', 'integration_status': 'NOT_TESTED'}
    def save():
        (args.run_dir / 'ninetoothed-probe.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        save()
        if os.environ.get('HSA_OVERRIDE_GFX_VERSION') or os.environ.get('TRITON_OVERRIDE_ARCH'):
            raise RuntimeError('architecture overrides are not allowed')
        import torch
        import numpy as np
        import triton
        report['baseline'] = baseline = json.loads(args.baseline.read_text())
        actual = {'torch_version': torch.__version__, 'torch_file': torch.__file__, 'hip': torch.version.hip,
                  'numpy_version': np.__version__, 'numpy_file': np.__file__}
        if actual != baseline:raise RuntimeError('candidate changed Torch/NumPy identity')
        if torch.__version__.split('+')[0] != '2.9.0' or not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('expected usable vendor HIP Torch2.9.0')
        if not numpy_compatible(np.__version__):raise RuntimeError('NumPy>=1.26.4 required')
        report['packages'] = {}
        for name, version, root in [('ninetoothed', '0.26.0', args.candidate/'site-packages'),
                                    ('sympy', '1.14.0', args.candidate/'site-packages'),
                                    ('mpmath', '1.3.0', args.candidate/'site-packages'),
                                    ('triton', '3.5.1+das.opt1.dtk2604.torch290', args.vendor)]:
            dist = metadata.distribution(name)
            if dist.version != version or not Path(dist.locate_file('')).resolve().is_relative_to(root.resolve()):
                raise RuntimeError('unexpected package version/origin: ' + name)
            report['packages'][name] = {'version': dist.version, 'path': str(dist.locate_file(''))}
        if not Path(triton.__file__).resolve().is_relative_to(args.vendor.resolve()):raise RuntimeError('wrong imported Triton')
        props = torch.cuda.get_device_properties(0)
        target = triton.runtime.driver.active.get_current_target()
        arch = str(getattr(props, 'gcnArchName', '')).split(':')[0]
        if arch != 'gfx936' or str(target.arch).split(':')[0] != arch or target.backend not in ('hip', 'hcu'):
            raise RuntimeError(f'expected actual gfx936 HIP/HCU target: Torch={arch}, Triton={target}')
        report.update(arch=arch, triton_target=str(target), device=str(props), numpy=np.__version__)
        import ninetoothed as nt
        if not Path(nt.__file__).resolve().is_relative_to((args.candidate/'site-packages').resolve()):raise RuntimeError('wrong imported NineToothed')
        report['last_stage'] = 'jit-add';save()
        generated = args.run_dir/'generated';generated.mkdir()
        report['generated_sources'] = []
        torch.manual_seed(20261106)
        for dtype_name in ('float32', 'float16', 'bfloat16'):
            launch = nt.make(arrangement, application,
                tuple(nt.Tensor(1, dtype=getattr(nt, dtype_name)) for _ in range(3)),
                kernel_name='gfx936_nt_add_' + dtype_name, num_warps=4, num_stages=1, max_num_configs=1)
            source = Path(launch._source);shutil.copyfile(source, generated/(dtype_name+'.py'))
            report['generated_sources'].append({'file': dtype_name+'.py',
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            dtype = getattr(torch, dtype_name)
            for n, stride in ((1, 1), (513, 1), (513, 2)):
                label = f'{dtype_name}-n{n}-stride{stride}'
                print('jit_case=START ' + label, flush=True)
                left_cpu = torch.randn(n*stride).to(dtype);right_cpu = torch.randn(n*stride).to(dtype)
                left, right = left_cpu.cuda(), right_cpu.cuda()
                backing = torch.full((n*stride+2,), -123, dtype=dtype, device='cuda')
                output = backing[1:1+n*stride:stride];output.fill_(float('nan'))
                expected = (left_cpu[::stride].float()+right_cpu[::stride].float()).to(dtype)
                launch(left[::stride], right[::stride], output);torch.cuda.synchronize()
                guard = torch.ones(backing.shape, dtype=torch.bool);guard[1:1+n*stride:stride] = False
                if not torch.equal(output.cpu(), expected):raise AssertionError('JIT output mismatch: ' + label)
                if not torch.equal(left.cpu(), left_cpu) or not torch.equal(right.cpu(), right_cpu):raise AssertionError('input mutated')
                if not torch.all(backing.cpu()[guard] == -123):raise AssertionError('output guard overwritten')
                np.savez(generated/(label+'.npz'), actual=output.float().cpu().numpy(), expected=expected.float().numpy())
                report['cases'].append({'name': label, 'passed': True, 'inputs_unchanged': True, 'guards_unchanged': True});save()
                print('jit_case=PASS ' + label, flush=True)
        report.update(result='JIT_VERIFIED', last_stage='complete')
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:
        for key in ('TRITON_CACHE_DIR', 'NINETOOTHED_CACHE_DIR'):
            folder = Path(os.environ.get(key, '/nonexistent'))
            if folder.is_dir():
                target = args.run_dir/('compiler-cache' if key=='TRITON_CACHE_DIR' else 'ninetoothed-cache')
                try:shutil.copytree(folder, target, dirs_exist_ok=True)
                except Exception as error:report.setdefault('cache_copy_errors', []).append(f'{key}: {error}')
        save()


if __name__ == '__main__':raise SystemExit(main())
