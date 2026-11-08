"""Original NineToothed expressions for H3 gate*branch + residual.

Gate/branch multiplication rounds to the input dtype before residual addition,
matching the original separate mul/add. The launcher must disable FP fusion.
Tensor storage, references and checks live in verify.py, not these expressions.
"""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(residual, gate, branch, output):
    return (residual.tile((1, 128)), gate.tile((1, 128)),
            branch.tile((1, 128)), output.tile((1, 128)))


def fp32(residual, gate, branch, output):
    product = ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32)
    output = ntl.cast(residual, ntl.float32) + product


def fp16(residual, gate, branch, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.float16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.float16)


def bf16(residual, gate, branch, output):
    product = ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), ntl.bfloat16)
    output = ntl.cast(ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32), ntl.bfloat16)


def make(dtype_name):
    dtype, application = {'float32': (nt.float32, fp32), 'float16': (nt.float16, fp16),
                          'bfloat16': (nt.bfloat16, bf16)}[dtype_name]
    return nt.make(arrangement, application, tuple(Tensor(2, dtype=dtype) for _ in range(4)),
                   kernel_name=f'h3_gated_residual_{dtype_name}', num_warps=4, num_stages=1,
                   max_num_configs=1)


def add_arrangement(left, right, output):
    return left.tile((128,)), right.tile((128,)), output.tile((128,))


def add_application(left, right, output):
    output = left + right


def make_add(dtype_name):
    return nt.make(add_arrangement, add_application,
                   tuple(Tensor(1, dtype=getattr(nt, dtype_name)) for _ in range(3)),
                   kernel_name=f'h3_probe_add_{dtype_name}', num_warps=4, num_stages=1,
                   max_num_configs=1)
