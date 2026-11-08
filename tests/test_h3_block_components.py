"""Host checks for the independent H3 component reference; no GPU kernels."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class BlockComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = SCRIPTS / 'verify_h3_block_components.py'
        if not script.is_file():
            raise AssertionError('Missing H3 block component verifier')
        sys.path.insert(0, str(SCRIPTS))
        spec = importlib.util.spec_from_file_location('h3_components', script)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_rms_uses_uncentered_second_moment_and_epsilon(self):
        x = np.array([[3., 4.], [0., 0.]], np.float32)
        w = np.array([2., 1.], np.float32)
        expected = np.array([[6., 4.], [0., 0.]]) / np.sqrt([[12.5 + 1e-5], [1e-5]])
        np.testing.assert_allclose(self.m.rms_reference(x, w), expected, atol=1e-6)

    def test_adaln_preserves_three_separate_rounding_boundaries(self):
        q = lambda a: np.asarray(a, np.float16).astype(np.float32)
        # The half-ULP addition rounds back to 1 before subtracting 1.
        x = q([1., 1.234, -2.125])
        s = q([2**-11, -.31, .125])
        shift = q([-1., .14, .333])
        expected = q(q(x + q(x * s)) + shift)
        np.testing.assert_array_equal(self.m.adaln_reference(x, s, shift, q), expected)
        self.assertTrue(np.any(expected != q(x * (1 + s) + shift)))

    def test_modality_rows_and_six_chunks_are_not_interchanged(self):
        projection = np.arange(2 * 18 * 2, dtype=np.float32).reshape(2, 36)
        time = np.array([1, 0, 1, 0], np.int64)
        tags = np.array([2, 0, 1, 2], np.int64)
        chunks = self.m.modulation_reference(projection, time, tags, 2)
        table = projection.reshape(6, 12)[[5, 0, 4, 2]][None]
        self.assertEqual(len(chunks), 6)
        for i, chunk in enumerate(chunks):
            np.testing.assert_array_equal(chunk, table[..., i*2:(i+1)*2])

    def test_swiglu_activates_second_half(self):
        up = np.array([2., -3., 1.], np.float32)
        gate = np.array([-1., 2., 0.], np.float32)
        expected = up * gate / (1 + np.exp(-gate))
        np.testing.assert_allclose(self.m.swiglu_reference(up, gate), expected, atol=1e-7)
        self.assertFalse(np.allclose(expected, self.m.swiglu_reference(gate, up)))

    def test_linear_bias_is_added_before_output_rounding(self):
        q = lambda a: np.asarray(a, np.float16).astype(np.float32)
        x = q([[1.3, -2.2]])
        w = q([[.4, .7], [1.2, -.3]])
        b = q([.3, -.2])
        np.testing.assert_array_equal(self.m.linear_reference(x, w, b, q), q(x @ w.T + b))

    def test_zero_gates_keep_both_residuals_exact_and_inputs_unchanged(self):
        case = next(c for c in self.m.make_cases() if c['scenario'] == 'zero_gates')
        data = self.m.make_inputs(case, np.random.default_rng(1), 0)
        copies = {k: v.copy() for k, v in data.items()}
        ref = self.m.chain_reference(data, lambda a: np.asarray(a, np.float32))
        np.testing.assert_array_equal(ref['attention_residual'], data['hidden'])
        np.testing.assert_array_equal(ref['mlp_residual'], data['hidden'])
        for name in data:
            np.testing.assert_array_equal(data[name], copies[name])

    def test_cases_cover_all_dtypes_and_inputs_change(self):
        cases = self.m.make_cases()
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({c['name'] for c in cases}), 15)
        for dtype in ('float16', 'bfloat16', 'float32'):
            self.assertEqual(sum(c['dtype'] == dtype for c in cases), 5)
        case = next(c for c in cases if c['scenario'] == 'mixed_modalities')
        rng = np.random.default_rng(7)
        a, b = [self.m.make_inputs(case, rng, i) for i in range(2)]
        self.assertFalse(np.array_equal(a['hidden'], b['hidden']))
        self.assertFalse(np.array_equal(a['tags'], b['tags']))
        self.assertEqual(set(a['tags']), {0, 1, 2})
        self.assertEqual(set(a['time_indices']), {0, 1})


if __name__ == '__main__':
    unittest.main()
