"""Host-only tests of reporting/child isolation, never GPU correctness tests."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h3_check', ROOT / 'scripts/verify_h3_toolchain.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class DriverTests(unittest.TestCase):
    def run_stub(self, body, stages=('environment', 'jit-add'), timeout=10):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stub = root / 'stub.py'
            stub.write_text('import sys\nstage=sys.argv[sys.argv.index("--stage")+1]\n' + body)
            with patch.object(check, '__file__', str(stub)), contextlib.redirect_stdout(io.StringIO()):
                rc = check.run_all(root, root, root, timeout, stages=stages)
            return rc, json.loads((root / 'summary.json').read_text())

    def test_zero_exit_without_completion_is_failure(self):
        rc, report = self.run_stub('print("no result")\n')
        self.assertEqual(rc, 1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['stages'][1]['status'], 'NOT_RUN')

    def test_crash_after_completion_marker_is_failure(self):
        rc, report = self.run_stub('print(f"STAGE PASSED: {stage}", flush=True)\nsys.exit(139 if stage=="jit-add" else 0)\n')
        self.assertEqual(rc, 1)
        self.assertEqual(report['stages'][1]['exit_code'], 139)
        self.assertFalse(report['stages'][1]['passed'])

    def test_jit_success_does_not_claim_aot(self):
        rc, report = self.run_stub('print(f"STAGE PASSED: {stage}")\n', stages=check.JIT_STAGES)
        self.assertEqual(rc, 0)
        self.assertTrue(report['passed'])
        self.assertEqual(report['scope'], 'jit')
        self.assertEqual(report['aot_validated'], False)

    def test_child_timeout_is_failure(self):
        rc, report = self.run_stub('import time\ntime.sleep(10)\n', timeout=0.2)
        self.assertEqual(rc, 1)
        self.assertEqual(report['stages'][0]['exit_code'], 124)

    def test_caches_are_under_run_artifacts(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ):
            root = Path(name).resolve()
            check.configure_caches(root)
            self.assertTrue(Path(os.environ['HOME']).is_relative_to(root))
            self.assertTrue(Path(os.environ['TRITON_CACHE_DIR']).is_relative_to(root))
            self.assertTrue(Path(os.environ['HOME']).is_dir())

    def test_known_hip_aot_mismatch_detected_without_compiling(self):
        aot = ROOT / 'repos/ninetoothed-v0.26.0/src/ninetoothed/aot.py'
        with tempfile.TemporaryDirectory() as name:
            triton = Path(name)
            template = triton / 'tools/extra/hip/compile.cpp'
            template.parent.mkdir(parents=True)
            template.write_text('// HIP template uses .cpp\n')
            issues = check.aot_compatibility_issues(aot, triton)
        self.assertTrue(any('.cpp' in item and '.c' in item for item in issues))
        self.assertTrue(any('cuCtxGetId' in item for item in issues))
        self.assertTrue(any('stride' in item for item in issues))


if __name__ == '__main__':
    unittest.main()
