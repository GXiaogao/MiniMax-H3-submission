"""Protect measurement units, completion gates and same-input timing boundaries."""
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/h3-gated-benchmark-v1/protocol.py'


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SOURCE.exists():
            spec = importlib.util.spec_from_file_location('gated_perf_protocol', SOURCE)
            cls.api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.api)

    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'Performance protocol implementation is missing')

    def test_percentiles_do_not_drop_slow_outlier(self):
        result = self.api.statistics_us([1, 2, 3, 4, 100])
        self.assertEqual(result['p50_us'], 3)
        self.assertAlmostEqual(result['p95_us'], 80.8)
        self.assertEqual(result['samples'], 5)

    def test_invalid_measurements_cannot_be_successful(self):
        for samples in ([], [0], [-1], [math.nan], [math.inf]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.api.statistics_us(samples)

    def test_batch_milliseconds_are_normalized_to_microseconds_per_call(self):
        self.assertEqual(self.api.per_call_us(2.5, 50), 50)
        for total, count in ((0, 10), (1, 0), (math.inf, 1)):
            with self.assertRaises(ValueError):
                self.api.per_call_us(total, count)

    def test_ab_ba_order_balanced_and_rotates_with_case(self):
        self.assertEqual(self.api.path_order(0, 0), ('baseline', 'fused'))
        self.assertEqual(self.api.path_order(1, 0), ('fused', 'baseline'))
        self.assertEqual(self.api.path_order(0, 1), ('fused', 'baseline'))

    def test_synced_sample_excludes_initial_sync_and_includes_final_sync(self):
        # Fake clock/device is necessary on this CPU Windows host. This tests
        # our timing boundary, not device behaviour or any performance value.
        ticks = [0]
        def sync(): ticks[0] += 1000
        def clock(): return ticks[0]
        def call(): ticks[0] += 3000
        elapsed, _ = self.api.host_sample(call, sync, clock)
        self.assertEqual(elapsed, 4.0)

    def test_device_batch_waits_and_excludes_event_creation(self):
        trail = []
        class Event:
            def __init__(self): trail.append('create')
            def record(self): trail.append('record')
            def synchronize(self): trail.append('wait')
            def elapsed_time(self, other):
                self.assert_waited = trail[-1] == 'wait'
                if not self.assert_waited: raise AssertionError('timer not synchronized')
                return 0.4
        elapsed, _ = self.api.device_sample(lambda: trail.append('call'), lambda: trail.append('sync'), Event, 2)
        self.assertEqual(elapsed, 200.0)
        self.assertEqual(trail, ['create', 'create', 'sync', 'record', 'call', 'call', 'record', 'wait'])

    def test_slowdown_is_reported_instead_of_discarded(self):
        result = self.api.comparison([10, 20, 30], [20, 40, 60])
        self.assertEqual(result['p50_speedup'], 0.5)
        self.assertEqual(result['p50_latency_reduction_percent'], -100.0)

    def test_completion_rejects_partial_cases_and_missing_correctness(self):
        row = {'passed': True, 'correct_before': True, 'correct_after': True,
               'launch_counts_ok': True, 'samples': [{'index': i, 'path': p,
                 'host_us': 1., 'device_us': 2.} for i in range(2) for p in ('baseline','fused')]}
        self.assertTrue(self.api.complete([row], 1, 2))
        self.assertFalse(self.api.complete([row], 2, 2))
        row['samples'].pop()
        self.assertFalse(self.api.complete([row], 1, 2))
        row['correct_after'] = False
        self.assertFalse(self.api.complete([row], 1, 2))

    def test_launch_gate_requires_exact_path_counts(self):
        self.api.check_launches('baseline', 0, 100)
        self.api.check_launches('fused', 100, 100)
        for path, got in (('baseline', 1), ('fused', 0), ('fused', 99)):
            with self.assertRaises(AssertionError): self.api.check_launches(path, got, 100)

    def test_bundle_rejects_non_v3_and_changed_files(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'kernel').write_bytes(b'good')
            manifest = {'revision': self.api.V3_REVISION, 'semantics': self.api.V3_SEMANTICS,
                'source_hashes': {'gated_residual.py':'verified'},
                'files': {'kernel': hashlib.sha256(b'good').hexdigest()}}
            (root/'bundle.json').write_text(json.dumps(manifest))
            self.api.validate_bundle(root, 'verified', required_files={'kernel'})
            (root/'kernel').write_bytes(b'changed')
            with self.assertRaises(ValueError): self.api.validate_bundle(root, 'verified', required_files={'kernel'})
            manifest['revision'] = 'v2'
            (root/'bundle.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): self.api.validate_bundle(root, 'verified', required_files={'kernel'})

    def test_missing_active_bundle_archives_failure_without_claiming_measurement(self):
        with tempfile.TemporaryDirectory() as name:
            result = subprocess.run([sys.executable,str(SOURCE.with_name('benchmark.py'))],
                env=dict(os.environ,WORK_HOME=name),capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            archives=list((Path(name)/'logs').glob('h3_gated_benchmark_*.zip'))
            self.assertEqual(len(archives),1)
            with zipfile.ZipFile(archives[0]) as archive:
                run=json.loads(archive.read(next(p for p in archive.namelist() if p.endswith('/run.json'))))
                self.assertFalse(run['passed'])
                self.assertIn('FileNotFoundError',run['error'])
            self.assertNotIn('BENCHMARK PASSED',result.stdout)

    @unittest.skipIf(sys.platform=='linux' and sys.version_info[:2]==(3,11),'Use CPU unsupported host for this gate test')
    def test_unsupported_host_emits_failed_summary_not_fake_performance(self):
        with tempfile.TemporaryDirectory() as name:
            result=subprocess.run([sys.executable,str(SOURCE.with_name('benchmark.py')),'--worker',name],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            report=json.loads((Path(name)/'summary.json').read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['performance_tested'])
            self.assertEqual(report['cases'],[])

    def test_invalid_sample_count_stops_before_creating_experiment(self):
        with tempfile.TemporaryDirectory() as name:
            result=subprocess.run([sys.executable,str(SOURCE.with_name('benchmark.py')),'--samples','3'],
                env=dict(os.environ,WORK_HOME=name),capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertFalse((Path(name)/'logs').exists())


if __name__ == '__main__':
    unittest.main()
