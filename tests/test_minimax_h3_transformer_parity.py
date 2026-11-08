"""Catch altered capture inputs, misleading comparisons, and unsafe placement."""
import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'artifacts/infinifabric-build-v1/diagnose_minimax_h3_transformer_parity.py'


class TransformerParityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'transformer parity diagnostic missing')
        spec = importlib.util.spec_from_file_location('transformer_parity_test', MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_capture_preserves_signature_input_and_single_call_then_stops(self):
        inputs = np.array([[[1.25, -2.0]]], dtype=np.float32)
        positions = np.array([[0.125, -3.5, 4.]], dtype=np.float64)
        calls = []
        def forward(hidden_states, *, position_ids, return_dict=False):
            calls.append(1)
            self.assertIs(hidden_states, inputs)
            self.assertIs(position_ids, positions)
            return (hidden_states + 2, hidden_states - 3)
        owner = SimpleNamespace(forward=forward)
        capture = self.mod.FirstForwardCapture(self.root)
        capture.attach(owner)
        self.assertEqual(inspect.signature(owner.forward), inspect.signature(forward))
        with self.assertRaises(self.mod.CaptureComplete):
            owner.forward(inputs, position_ids=positions)
        self.assertEqual(calls, [1])
        saved, metadata = self.mod.read_bundle(self.root, 'first-inputs')
        np.testing.assert_array_equal(saved['hidden_states'], [[[1.25, -2.0]]])
        self.assertEqual(saved['position_ids'].dtype, np.float64)
        np.testing.assert_array_equal(saved['position_ids'], positions)
        output, _ = self.mod.read_bundle(self.root, 'first-native-outputs')
        np.testing.assert_array_equal(output['video'], [[[3.25, 0.]]])
        capture.close()
        self.assertIs(owner.forward, forward)

    def test_bundle_rejects_tampering_and_does_not_round_large_indices(self):
        self.mod.save_bundle(self.root, 'example', {'ids': np.array([2**54+1], dtype=np.int64)})
        arrays, metadata = self.mod.read_bundle(self.root, 'example')
        self.assertEqual(int(arrays['ids'][0]), 2**54+1)
        with (self.root/'example.npz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.mod.read_bundle(self.root, 'example')

    def test_metrics_use_full_arrays_and_do_not_invent_pass_threshold(self):
        result = self.mod.compare_arrays(np.array([1., 2.]), np.array([1., 4.]))
        self.assertEqual(result['elements'], 2)
        self.assertEqual(result['mae'], 1.)
        self.assertAlmostEqual(result['rmse'], math.sqrt(2))
        self.assertAlmostEqual(result['relative_l2'], 2/math.sqrt(5))
        self.assertAlmostEqual(result['cosine'], 9/math.sqrt(85))
        self.assertNotIn('passed', result)

    def test_metrics_refuse_nonfinite_and_shape_mismatch(self):
        for a,b in [(np.ones(2), np.ones((1,2))), (np.array([1.]),np.array([np.nan]))]:
            with self.assertRaises(ValueError): self.mod.compare_arrays(a,b)
        zero = self.mod.compare_arrays(np.zeros(3),np.ones(3))
        self.assertIsNone(zero['relative_l2'])
        self.assertIsNone(zero['cosine'])
        json.dumps(zero,allow_nan=False)

    def test_reference_placement_rejects_cpu_disk_and_unknown_gpu(self):
        self.mod.validate_device_map({'a':0,'b':'cuda:7'}, list(range(8)))
        for value in ('cpu','disk','cuda:8'):
            with self.assertRaises(ValueError): self.mod.validate_device_map({'x':value},list(range(8)))

    def test_real_worker_failure_prevents_reference_and_retains_stderr(self):
        worker = self.root/'worker.py'
        worker.write_text('import sys\nprint("worker failed",file=sys.stderr)\nsys.exit(37)\n')
        with self.assertRaisesRegex(RuntimeError, '37'):
            self.mod.run_worker([sys.executable,str(worker)],self.root/'worker.log',{})
        self.assertIn('worker failed',(self.root/'worker.log').read_text())

    def test_actual_forward_error_is_not_mistaken_for_capture_complete(self):
        def forward(hidden_states): raise RuntimeError('device kernel failed')
        owner=SimpleNamespace(forward=forward)
        capture=self.mod.FirstForwardCapture(self.root)
        capture.attach(owner)
        with self.assertRaisesRegex(RuntimeError,'device kernel failed'): owner.forward(np.ones(2))
        self.assertFalse((self.root/'first-native-outputs.npz').exists())
        capture.close()

    def test_failed_preflight_does_not_claim_workers_have_run(self):
        rc=self.mod.main(['--run-dir',str(self.root),'--work-home',str(self.root/'absent'),
                          '--local-target',str(self.root/'local'), '--model-path',str(self.root/'model'),
                          '--first-frame',str(self.root/'input.jpg')])
        self.assertEqual(rc,1)
        report=json.loads((self.root/'transformer-parity.json').read_text())
        self.assertEqual(report.get('completed_workers'),[])
        self.assertNotIn('executed_transformer_forwards',report)


if __name__ == '__main__':
    unittest.main()
