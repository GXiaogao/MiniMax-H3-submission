import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/diagnose_minimax_h3_adaln_probe.py'

class AdaLNProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(),'AdaLN probe missing')
        spec=importlib.util.spec_from_file_location('adaln_probe_test',SCRIPT)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_gather_oracle_preserves_time_rows_and_rank_columns(self):
        parts=[np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]])]
        self.assertEqual(self.m.gather_expected(parts).tolist(),[[1,2,5,6],[3,4,7,8]])
        with self.assertRaises(ValueError):self.m.gather_expected([parts[0],np.zeros((1,2))])

    def test_shard_ranges_cover_width_without_overlap(self):
        self.assertEqual(self.m.shard_ranges(16,8),[(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,16)])
        with self.assertRaises(ValueError):self.m.shard_ranges(15,8)

    def test_marker_layout_is_exact_bf16_safe_and_rank_distinct(self):
        x=self.m.marker_parts(2,16,8)
        self.assertEqual(x[0].tolist(),[[0,1],[8,9]])
        self.assertEqual(x[1].tolist(),[[16,17],[24,25]])
        self.assertTrue(all(np.array_equal(p,p.astype(np.float16)) for p in x))

    def test_compiler_failure_retains_log_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            with self.assertRaises(RuntimeError):
                self.m.compile_shim(folder,folder,compiler='missing-compiler-for-adaln-test')
            self.assertTrue((folder/'compile.log').is_file())

    def test_main_preflight_failure_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.m.main(['--run-dir',d,'--work-home',d,'--model-path',d]),1)
            import json
            self.assertEqual(json.loads((Path(d)/'adaln-probe.json').read_text())['result'],'FAILED')
