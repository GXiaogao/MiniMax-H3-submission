"""Directed FP32 addition checked with exact rational arithmetic, without a GPU."""
import ast
from fractions import Fraction
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]/'scripts/h3-gated-rounding-v3'


def floor_sum(x,y):
    exact=Fraction(float(x))+Fraction(float(y))
    if not exact:
        return np.float32(0.0 if x==0 and y==0 and not np.signbit(x) and not np.signbit(y) else -0.0)
    maximum=Fraction(float(np.finfo(np.float32).max))
    if exact>maximum:return np.finfo(np.float32).max
    if exact < -maximum:return np.float32(-np.inf)
    with np.errstate(over='ignore'):
        result=np.float32(float(exact))
    if np.isposinf(result):result=np.finfo(np.float32).max
    if np.isneginf(result):result=-np.finfo(np.float32).max
    while Fraction(float(result))>exact:
        result=np.nextafter(result,np.float32(-np.inf))
    following=np.nextafter(result,np.float32(np.inf))
    while np.isfinite(following) and Fraction(float(following))<=exact:
        result=following;following=np.nextafter(result,np.float32(np.inf))
    return result


def triton_scalar_lowering():
    """Run the installed candidate wheel's real scalar-constant branch without a GPU.

    Only the IR builder is replaced by typed NumPy scalar storage. This checks
    the compiler boundary missed by the earlier mathematical-expression test.
    """
    wheels=Path('E:/projects/minimax-h3-match/相关代码文件/wheel')
    paths=list(wheels.glob('triton-3.8.0-*.whl'))
    if len(paths)!=1:
        raise unittest.SkipTest('The exact offline Triton 3.8.0 wheel is required')
    with zipfile.ZipFile(paths[0]) as archive:
        tree=ast.parse(archive.read('triton/language/semantic.py').decode())
    func=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='scalar_constant')
    func.returns=None
    for arg in func.args.args:arg.annotation=None
    env={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(paths[0]),'exec'),env)
    builder=SimpleNamespace(get_null_value=lambda ty:np.asarray(0,dtype=ty),
                            get_fp32=lambda x:np.asarray(x,dtype=np.float32),
                            get_int32=lambda x:np.asarray(x,dtype=np.int32))
    semantic=SimpleNamespace(builder=builder,tensor=lambda value,dtype:value)
    def lower(value,dtype):
        ty=SimpleNamespace(name='fp32' if dtype==np.float32 else 'int32',
                           to_ir=lambda builder:dtype,is_fp8=lambda:False)
        return env['scalar_constant'](semantic,value,ty)
    return lower


def expression(lower_scalars=None):
    module=ast.parse((PACKAGE/'gated_residual.py').read_text())
    func=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='fp32')
    func.body.append(ast.Return(ast.Name('output',ast.Load())))
    program=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    def cast(value,dtype,bitcast=False):
        if lower_scalars and isinstance(value,(float,int)):
            value=lower_scalars(value,np.float32 if isinstance(value,float) else np.int32)
        a=np.asarray(value)
        return a.view(dtype) if bitcast else a.astype(dtype)
    def where(condition,left,right):
        if lower_scalars:
            left=lower_scalars(left,np.float32 if isinstance(left,float) else np.int32) if isinstance(left,(float,int)) else left
            right=lower_scalars(right,np.float32 if isinstance(right,float) else np.int32) if isinstance(right,(float,int)) else right
        return np.where(condition,left,right)
    env={'ntl':SimpleNamespace(cast=cast,float32=np.float32,int32=np.int32,
                               where=where,abs=np.abs)}
    exec(compile(program,'NineToothed FP32 expression via NumPy','exec'),env)
    return env['fp32']


class DirectedAdditionTests(unittest.TestCase):
    def assert_bits(self,a,b):
        np.testing.assert_array_equal(np.asarray(a,dtype=np.float32).view(np.uint32),
                                      np.asarray(b,dtype=np.float32).view(np.uint32))

    def test_rounding_sentinel_reproduces_previous_mismatch(self):
        r=np.array([1.0],np.float32);p=np.array([3*2.0**-25],np.float32)
        self.assertNotEqual((r+p)[0],floor_sum(r[0],p[0]))
        self.assert_bits(expression()(r,np.ones_like(r),p,None),np.array([1.0],np.float32))

    def test_triton380_scalar_boundary_loses_negative_zero_literal(self):
        # Characterize the actual upstream method, not a handwritten imitation.
        value=triton_scalar_lowering()(-0.0,np.float32)
        self.assertEqual(value.view(np.uint32).item(),0)

    def test_directed_zero_sign_survives_real_triton_scalar_lowering(self):
        a=np.array([1.,-1.,0.,-0.,0.,-0.],np.float32)
        b=np.array([-1.,1.,0.,0.,-0.,-0.],np.float32)
        want=np.array([0x80000000,0x80000000,0,0x80000000,0x80000000,0x80000000],np.uint32)
        actual=expression(triton_scalar_lowering())(a,np.ones_like(a),b,None)
        np.testing.assert_array_equal(actual.view(np.uint32),want)

    def test_signed_zero_cancellation_and_overflow(self):
        m=np.finfo(np.float32).max;t=np.nextafter(np.float32(0),np.float32(1))
        a=np.array([0.,-0.,0.,-0.,1.,-1.,m,-m,m,-m,t,-t,1.,-1.],np.float32)
        b=np.array([0.,0.,-0.,-0.,-1.,1.,m,-m,-t,-t,t,t,t,t],np.float32)
        with np.errstate(all='ignore'):
            actual=expression()(a,np.ones_like(a),b,None)
            expected=np.array([floor_sum(x,y) for x,y in zip(a,b)],np.float32)
        self.assert_bits(actual,expected)

    def test_exponent_spread_against_fraction_oracle(self):
        rng=np.random.default_rng(20261027)
        bits=rng.integers(0,2**32,size=(2,6000),dtype=np.uint32)
        values=bits.view(np.float32);mask=np.isfinite(values).all(axis=0)
        a,b=values[:,mask]
        with np.errstate(all='ignore'):
            expected=np.array([floor_sum(x,y) for x,y in zip(a,b)],np.float32)
            actual=expression()(a,np.ones_like(a),b,None)
        self.assert_bits(actual,expected)

    def test_cpu_reference_independently_matches_fraction_oracle(self):
        path=PACKAGE/'rounding_reference.py'
        self.assertTrue(path.exists(),'Directed CPU reference is required')
        spec=importlib.util.spec_from_file_location('rd_reference_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        rng=np.random.default_rng(7)
        a=rng.integers(0,2**32,2000,dtype=np.uint32).view(np.float32)
        b=rng.integers(0,2**32,2000,dtype=np.uint32).view(np.float32)
        mask=np.isfinite(a)&np.isfinite(b);a,b=a[mask],b[mask]
        with np.errstate(all='ignore'):
            expected=np.array([floor_sum(x,y) for x,y in zip(a,b)],np.float32)
            actual=mod.add_down(a,b)
        self.assert_bits(actual,expected)


if __name__=='__main__':unittest.main()
