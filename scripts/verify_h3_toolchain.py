#!/usr/bin/env python3
"""Separate NineToothed JIT/AOT GPU checks. PyTorch is only test data/reference.

No existing InfiniCore MHA is called or modified. Nonzero exits and absent
results are failures, never skipped passes. Generated AOT files are retained.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

JIT_STAGES = ('environment', 'jit-add', 'jit-matmul', 'jit-attention')
STAGES = JIT_STAGES + ('aot-add', 'aot-matmul', 'aot-attention')
EXPECTED = {'triton': '3.5.1+das.opt1.dtk2604.torch290', 'ninetoothed': '0.26.0',
            'numpy': '1.26.4', 'sympy': '1.14.0', 'mpmath': '1.3.0'}


def aot_compatibility_issues(aot_path, triton_root):
    """Source-level blockers, not a replacement for execution/accuracy tests."""
    source = aot_path.read_text(encoding='utf-8')
    issues = []
    if (triton_root / 'tools/extra/hip/compile.cpp').is_file() and 'signature_hash}.c"' in source:
        issues.append('HIP produces .cpp; this NineToothed AOT expects .c.')
    if 'cuCtxGetId' in source:
        issues.append('AOT context lookup still uses CUDA cuCtxGetId.')
    if '"nvcc"' in source and '"-lcuda"' in source:
        issues.append('AOT shared-library command is CUDA-specific: nvcc / -lcuda.')
    if 'param_types.append(f"{stride_type}:16")' in source:
        issues.append('AOT declares unchecked stride alignment :16; strided inputs require repair.')
    return issues


def configure_caches(folder):
    # NineToothed 0.26 uses Path.home()/.ninetoothed and ignores JIT output_dir.
    # Only this process and its children get the temporary HOME; shell is unchanged.
    home = folder / 'home'
    home.mkdir(parents=True, exist_ok=True)
    os.environ['HOME'] = str(home)
    os.environ['TRITON_CACHE_DIR'] = str(folder / 'triton-cache')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')


def add_arrangement(left, right, output):
    return left.tile((128,)), right.tile((128,)), output.tile((128,))


def add_application(left, right, output):
    output = left + right


# Tiling adapted from NineToothed v0.26.0/tests/test_matmul.py (Apache-2.0).
# Upstream source and LICENSE are retained in repos/ninetoothed-v0.26.0.
def matmul_arrangement(left, right, output):
    out = output.tile((32, 32))
    lhs = left.tile((32, 32)).tile((1, -1)).expand((-1, out.shape[1]))
    lhs.dtype = lhs.dtype.squeeze(0)
    rhs = right.tile((32, 32)).tile((-1, 1)).expand((out.shape[0], -1))
    rhs.dtype = rhs.dtype.squeeze(1)
    return lhs, rhs, out


def matmul_application(left, right, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(left.shape[0]):
        accumulator += ntl.dot(left[k], right[k])
    output = accumulator


def environment():
    import torch
    for name, expected in EXPECTED.items():
        actual = metadata.version(name)
        print(f'{name}: {actual}', flush=True)
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {actual}')
    print('python:', sys.executable, 'torch:', metadata.version('torch'),
          'torch path:', torch.__file__, 'HIP:', torch.version.hip, flush=True)
    if sys.version_info[:2] != (3, 11) or not torch.version.hip:
        raise RuntimeError('Expected Python 3.11 and vendor HIP Torch.')
    if not torch.cuda.is_available():
        raise RuntimeError('No usable DCU. CPU execution is not a valid result.')
    props = torch.cuda.get_device_properties(0)
    print('device:', props, flush=True)
    import triton
    target = triton.runtime.driver.active.get_current_target()
    print('Triton target:', target, flush=True)
    if str(target.arch).split(':')[0] != 'gfx906' or target.backend not in ('hip', 'hcu'):
        raise RuntimeError(f'Expected Hygon gfx906 target; got {target}')
    import ninetoothed.aot as aot
    print('NineToothed AOT API:', aot.__file__, flush=True)
    if not hasattr(aot, '_HEADER_PATH'):
        raise RuntimeError('InfiniCore-required AOT header API is absent.')
    issues = aot_compatibility_issues(Path(aot.__file__), Path(triton.__file__).parent)
    print('NineToothed cache:', aot._HEADER_PATH.parent, flush=True)
    for issue in issues:
        print('AOT BLOCKER:', issue, flush=True)
    print('AOT preflight:', 'BLOCKED' if issues else 'no known static blocker; execution still required', flush=True)
    return issues


def check_result(label, output, expected, atol, rtol):
    import torch
    torch.cuda.synchronize()
    actual = output.detach().float().cpu()
    expected = expected.float().cpu()
    if not torch.isfinite(actual).all():
        raise AssertionError(f'{label}: nonfinite output (including unwritten values)')
    max_error = (actual - expected).abs().max().item()
    print(f'{label}: max_abs_error={max_error:.8g}; atol={atol}; rtol={rtol}', flush=True)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def worker(stage, folder, core):
    issues = environment()
    if stage == 'environment':
        return
    if stage.startswith('aot-') and issues:
        raise RuntimeError('AOT not run: resolve the reported source-level blockers first.')
    global ntl
    import torch
    import ninetoothed as nt
    import ninetoothed.language as ntl
    from ninetoothed import Tensor
    torch.manual_seed(20261022)
    caller = 'cuda' if stage.startswith('aot-') else 'torch'

    def make(arrangement, application, tensors, suffix):
        target_dir = folder / suffix
        target_dir.mkdir(parents=True, exist_ok=True)
        return nt.make(arrangement, application, tensors, caller=caller,
                       kernel_name='h3_check_' + suffix.replace('-', '_'),
                       output_dir=str(target_dir), num_warps=4, num_stages=1)

    if stage.endswith('-add'):
        kernel = make(add_arrangement, add_application,
                      tuple(Tensor(1, dtype=nt.float32) for _ in range(3)), stage)
        for size, stride in ((1, 1), (513, 1), (513, 2)):
            a_cpu = torch.randn(size * stride)
            b_cpu = torch.randn(size * stride)
            a, b = a_cpu.cuda()[::stride], b_cpu.cuda()[::stride]
            out = torch.full((size * stride,), float('nan'), device='cuda')[::stride]
            kernel(a, b, out)
            check_result(f'{stage} n={size} stride={stride}', out,
                         a_cpu[::stride] + b_cpu[::stride], 1e-6, 1e-6)
    elif stage.endswith('-matmul'):
        kernel = make(matmul_arrangement, matmul_application,
                      tuple(Tensor(2, dtype=nt.float16) for _ in range(3)), stage)
        for m, n, k in ((31, 33, 64), (64, 64, 128)):
            # Strided rows on the left and a transposed right operand.
            a_cpu = (torch.randn(m, k * 2) * 0.25).half()
            b_cpu = (torch.randn(n, k) * 0.25).half()
            a, b = a_cpu.cuda()[:, ::2], b_cpu.cuda().t()
            out = torch.full((m, n), float('nan'), device='cuda', dtype=torch.float16)
            kernel(a, b, out)
            check_result(f'{stage} {m}x{n}x{k}', out,
                         a_cpu[:, ::2].float() @ b_cpu.float().t(), 1e-2, 1e-2)
    else:
        path = core / 'src/infiniop/ops/flash_attention/ninetoothed/flash_attention.py'
        print('Attention source:', path, 'SHA256:', hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
        spec = importlib.util.spec_from_file_location('h3_core_nt_attention', path)
        attention = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = attention
        spec.loader.exec_module(attention)
        for dtype, nt_dtype in ((torch.float16, nt.float16), (torch.bfloat16, nt.bfloat16)):
            for head_dim in (64, 128):
                arrangement, application, tensors = attention.premake(
                    with_kv_cache=False, emb_dim=head_dim, is_causal=0,
                    with_attn_mask=0, causal_variant=2, dtype=nt_dtype,
                    block_size_m=32, block_size_n=32)
                suffix = f'{stage}_{str(dtype).split(".")[-1]}_{head_dim}'
                kernel = make(arrangement, application, tensors, suffix)
                # BSND input is packed QKV; views passed to FlashAttention are BHSD.
                for seq, heads, kv_heads in ((1, 1, 1), (17, 2, 2), (33, 4, 2)):
                    packed = (torch.randn(1, seq, heads + 2 * kv_heads, head_dim) * 0.25).to(dtype)
                    gpu = packed.cuda()
                    q = gpu[:, :, :heads, :].permute(0, 2, 1, 3)
                    k = gpu[:, :, heads:heads + kv_heads, :].permute(0, 2, 1, 3)
                    v = gpu[:, :, heads + kv_heads:, :].permute(0, 2, 1, 3)
                    out = torch.full_like(q, float('nan'))
                    length = torch.tensor([seq], dtype=torch.int32, device='cuda')
                    # Static arguments are still present in NineToothed's launch signature.
                    mask = torch.empty((1, 1, 1, 1), dtype=dtype, device='cuda')
                    scale = head_dim ** -0.5
                    kernel(q, k, v, length, mask, 0, scale, out, 0, 2)
                    q_ref = packed[:, :, :heads].permute(0, 2, 1, 3).float()
                    k_ref = packed[:, :, heads:heads + kv_heads].permute(0, 2, 1, 3).float().repeat_interleave(heads // kv_heads, dim=1)
                    v_ref = packed[:, :, heads + kv_heads:].permute(0, 2, 1, 3).float().repeat_interleave(heads // kv_heads, dim=1)
                    expected = torch.softmax((q_ref @ k_ref.transpose(-2, -1)) * scale, dim=-1) @ v_ref
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-2
                    check_result(f'{suffix} S={seq} H={heads}/{kv_heads}', out, expected, tolerance, tolerance)


def run_child(command, output, timeout):
    """Kill the whole compiler process group on Linux timeouts/interruption."""
    with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                          start_new_session=(os.name == 'posix')) as child:
        try:
            return child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == 'posix':
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                child.kill()
            child.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return 124


def run_all(folder, core, log_dir, timeout, stages=STAGES):
    results = []
    scope = 'full' if any(s.startswith('aot-') for s in stages) else 'jit'
    report = {'scope': scope, 'passed': False, 'aot_validated': False,
              'artifacts': str(folder), 'stages': results}
    for stage in stages:
        print(f'\n=== {stage} ===', flush=True)
        command = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                   '--stage', stage, '--artifacts', str(folder), '--core', str(core)]
        stage_log = log_dir / (stage + '.log')
        with stage_log.open('w', encoding='utf-8') as output:
            rc = run_child(command, output, timeout)
        text = stage_log.read_text(encoding='utf-8', errors='replace')
        print(text, end='', flush=True)
        # Require an explicit completion marker, including all numerical checks.
        passed = rc == 0 and f'STAGE PASSED: {stage}' in text
        results.append({'stage': stage, 'passed': passed, 'status': 'PASS' if passed else 'FAIL',
                        'exit_code': rc, 'log': str(stage_log)})
        (log_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'[{"PASS" if passed else "FAIL"}] {stage}; exit={rc}', flush=True)
        if stage == 'environment' and not passed:
            results.extend({'stage': name, 'passed': False, 'status': 'NOT_RUN', 'exit_code': None}
                           for name in stages[1:])
            print('Remaining GPU stages were NOT RUN because environment validation failed.', flush=True)
            break
    all_passed = len(results) == len(stages) and all(row['passed'] for row in results)
    report['passed'] = all_passed
    report['aot_validated'] = all_passed and scope == 'full'
    (log_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('REPORT:', log_dir / 'summary.json', flush=True)
    print(f'{scope.upper()} CHECKS PASSED' if all_passed else f'{scope.upper()} CHECKS FAILED', flush=True)
    if not report['aot_validated']:
        print('AOT NOT VALIDATED. InfiniCore MHA is NOT repaired by this script.', flush=True)
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    parser.add_argument('--core', type=Path)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--jit-only', action='store_true', help='test JIT and report AOT static blockers; does not validate AOT/MHA')
    parser.add_argument('--timeout', type=int, default=600, help='seconds per isolated stage')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not sys.platform.startswith('linux'):
        parser.error('Run on the Linux Hygon Pod; local syntax checks cannot validate the DCU.')
    work = Path(os.environ['WORK_HOME'])
    core = args.core or work / 'Projects/InfiniCore'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    folder = args.artifacts or work / 'Projects/h3-toolchain-check' / stamp
    folder.mkdir(parents=True, exist_ok=True)
    configure_caches(folder)
    if args.stage == 'all':
        log_dir = Path(os.environ['LOG_DIR']) / ('h3_toolchain_' + stamp)
        log_dir.mkdir(parents=True, exist_ok=True)
        return run_all(folder, core, log_dir, args.timeout, stages=JIT_STAGES if args.jit_only else STAGES)
    worker(args.stage, folder, core)
    print(f'STAGE PASSED: {args.stage}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
