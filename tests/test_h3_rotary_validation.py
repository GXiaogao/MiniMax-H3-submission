"""CPU oracle and failure-report tests; these do not execute Hygon kernels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_h3_rotary_mha.py'


class RotaryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.is_file():
            raise AssertionError('Missing H3 rotary/MHA validation script')
        spec = importlib.util.spec_from_file_location('h3_rotary_validation', SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_continuous_cache_axis_order_and_fractional_positions(self):
        positions = np.array([[0, np.pi / 2, np.pi]], dtype=np.float32)
        cache = self.module.continuous_cache(positions, 1)
        np.testing.assert_allclose(cache, [[1, 0, -1, 0, 1, 0]], atol=2e-7)
        scaled = self.module.continuous_cache(np.array([[1, 2, 3]], np.float32), 2)
        angles = np.array([[1, .01, 2, .02, 3, .03]], np.float32)
        np.testing.assert_allclose(scaled[:, :6], np.cos(angles), atol=1e-7)
        np.testing.assert_allclose(scaled[:, 6:], np.sin(angles), atol=1e-7)

    def test_neox_pairing_and_exact_tail(self):
        x = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 8)
        cache = np.array([[1, 0, -1, 0, 1, 0]], np.float32)
        out = self.module.rotary_reference(x, cache, np.array([0]), 1)
        np.testing.assert_array_equal(out, [[[1, -5, -3, 4, 2, -6, 7, 8]]])
        np.testing.assert_array_equal(x, np.arange(1, 9).reshape(1, 1, 8))

    def test_cache_rows_can_be_reordered(self):
        x = np.arange(1, 17, dtype=np.float32).reshape(2, 1, 8)
        cache = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], np.float32)
        out = self.module.rotary_reference(x, cache, np.array([1, 0]), 1)
        np.testing.assert_array_equal(out[0], [[-4, -5, -6, 1, 2, 3, 7, 8]])
        np.testing.assert_array_equal(out[1], x[1])

    def test_noncausal_attention_and_contiguous_gqa_groups(self):
        q = np.zeros((2, 4, 2), np.float32)
        k = np.zeros((2, 2, 2), np.float32)
        v = np.array([[[1, 3], [10, 30]], [[3, 5], [30, 50]]], np.float32)
        expected = np.array([[[2, 4], [2, 4], [20, 40], [20, 40]]] * 2, np.float32)
        np.testing.assert_array_equal(self.module.attention_reference(q, k, v), expected)

    def test_nonfinite_or_stale_results_are_not_accepted(self):
        with self.assertRaises(AssertionError):
            self.module.compare_arrays(np.array([np.nan]), np.array([1.]), .01, .01)
        with self.assertRaises(AssertionError):
            self.module.compare_arrays(np.array([1.]), np.array([2.]), .01, .01)
        metrics = self.module.compare_arrays(np.array([1.]), np.array([1.]), .01, .01)
        self.assertEqual(metrics['max_abs'], 0.)

    def test_failed_case_stops_and_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            report = {'cases': [], 'status': 'running'}
            visited = []
            def run(case, record):
                visited.append(case['name'])
                record['repeats'] = [{'mrope_q': {'max_abs': .001}}]
                if case['name'] == 'bad':
                    raise AssertionError('wrong output')
            cases = [{'name': n} for n in ('good', 'bad', 'never')]
            with self.assertRaises(AssertionError):
                self.module.run_cases(cases, run, report, path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(visited, ['good', 'bad'])
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual([r['status'] for r in saved['cases']], ['passed', 'failed'])
            self.assertEqual(saved['cases'][-1]['repeats'][0]['mrope_q']['max_abs'], .001)

    def test_invalid_report_cannot_replace_valid_running_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            self.module.write_report(path, {'status': 'running'})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                self.module.write_report(path, {'status': 'passed', 'max_abs': float('nan')})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(path.read_text())['status'], 'running')


if __name__ == '__main__':
    unittest.main()
