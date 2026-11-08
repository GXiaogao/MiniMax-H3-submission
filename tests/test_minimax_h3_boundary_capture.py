import importlib.util
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/diagnose_minimax_h3_boundaries.py'

class BoundaryCaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(),'boundary capture missing')
        s=importlib.util.spec_from_file_location('capture_test',SCRIPT)
        self.m=importlib.util.module_from_spec(s);s.loader.exec_module(self.m)

    def test_sampling_uses_same_integer_indices_as_cpp(self):
        self.assertEqual(self.m.sample_indices(10,4).tolist(),[0,3,6,9])
        self.assertEqual(self.m.sample_indices(1,1).tolist(),[0])
        with self.assertRaises(ValueError):self.m.sample_indices(2,3)

    def test_snapshot_roundtrip_records_shape_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=np.arange(12,dtype=np.float32).reshape(3,4)
            self.m.save_snapshot(p,'temb',x)
            samples,meta=self.m.read_snapshot(p,'temb')
            np.testing.assert_array_equal(samples,x.ravel())
            self.assertEqual(meta['shape'],[3,4])
            self.assertIn('data_sha256',meta)
            with self.assertRaises(FileExistsError):self.m.save_snapshot(p,'temb',x)

    def test_bad_size_and_nonfinite_samples_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.m.save_snapshot(p,'temb',np.array([1.,2.],dtype=np.float32))
            (p/'temb.f32').write_bytes(b'bad')
            with self.assertRaises(ValueError):self.m.read_snapshot(p,'temb')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):self.m.save_snapshot(Path(d),'temb',np.array([float('nan')]))

    def test_shape_mismatch_cannot_be_reported_as_matching(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            self.m.save_snapshot(a,'temb',np.ones((2,3),dtype=np.float32))
            self.m.save_snapshot(b,'temb',np.ones((3,2),dtype=np.float32))
            with self.assertRaises(ValueError):self.m.compare_boundary(a,b,'temb')

    def test_comparison_is_explicitly_sample_scoped(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.mkdir();b.mkdir()
            self.m.save_snapshot(a,'temb',np.array([1.,2.],dtype=np.float32))
            self.m.save_snapshot(b,'temb',np.array([1.,3.],dtype=np.float32))
            result=self.m.compare_boundary(a,b,'temb')
            self.assertEqual(result['scope'],'saved-boundary-samples')
            self.assertEqual(result['full_shape'],[2])
            self.assertAlmostEqual(result['relative_l2'],1/np.sqrt(5))

    def test_failed_preflight_retains_failed_status(self):
        with tempfile.TemporaryDirectory() as d:
            rc=self.m.main(['--run-dir',d,'--work-home',d,'--local-target',d+'/local','--model-path',d+'/model'])
            self.assertEqual(rc,1)
            report=json.loads((Path(d)/'boundary-parity.json').read_text())
            self.assertEqual(report['result'],'FAILED');self.assertEqual(report['completed_workers'],[])
