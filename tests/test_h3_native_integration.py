"""CPU-only checks for synthetic native H3 integration fixtures and reference."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SCRIPTS / 'verify_h3_native_integration.py'
        if not path.is_file():
            raise AssertionError('Missing native H3 integration verifier')
        sys.path.insert(0, str(SCRIPTS))
        spec = importlib.util.spec_from_file_location('h3_native', path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def setUp(self):
        self.c = self.m.make_config('float32')
        self.q = lambda x: np.asarray(x, np.float32)
        self.w = self.m.make_weights(self.c, self.q)

    def test_explicit_small_config_and_mixed_precision_weight_layout(self):
        self.assertEqual(self.c['num_layers'], 1)
        self.assertEqual(self.c['token_refiner_num_layers'], 1)
        self.assertLess(self.c['hidden_size'], 256)
        self.assertEqual(self.c['attention_head_dim'], 64)
        self.assertEqual(len(self.w), 42)
        self.assertEqual(self.w['proj_in.weight'].shape, (64, 16))
        self.assertEqual(self.w['transformer_blocks.0.attn.to_q.weight'].shape, (128, 64))
        self.assertTrue(self.m.preserve_fp32('time_embedder.linear_1.weight'))
        self.assertFalse(self.m.preserve_fp32('norm_out.linear.weight'))

    def test_diffusers_packs_up_before_gate_without_mutation(self):
        before = {k: v.copy() for k, v in self.w.items()}
        packed = self.m.checkpoint_layout(self.w, self.c, 'diffusers')
        p = 'transformer_blocks.0.ff.net.0.proj.'
        np.testing.assert_array_equal(packed[p+'weight'][:128], self.w[p+'up_proj.weight'])
        np.testing.assert_array_equal(packed[p+'weight'][128:], self.w[p+'gate_proj.weight'])
        self.assertNotIn(p+'up_proj.weight', packed)
        for k in before:
            np.testing.assert_array_equal(self.w[k], before[k])

    def test_legacy_qkv_is_head_major_and_mlp_is_gate_then_up(self):
        packed = self.m.checkpoint_layout(self.w, self.c, 'legacy')
        qkv = packed['blocks.0.attn.qkv_proj.weight'].reshape(2, 3, 64, 64)
        for slot, name in enumerate(('q', 'k', 'v')):
            np.testing.assert_array_equal(qkv[:, slot].reshape(128, 64), self.w[f'transformer_blocks.0.attn.to_{name}.weight'])
        mlp = packed['blocks.0.mlp.fc1.weight']
        np.testing.assert_array_equal(mlp[:128], self.w['transformer_blocks.0.ff.net.0.proj.gate_proj.weight'])
        np.testing.assert_array_equal(mlp[128:], self.w['transformer_blocks.0.ff.net.0.proj.up_proj.weight'])
        self.assertIn('video_patch_proj.weight', packed)
        self.assertIn('token_refiner.blocks.0.attn.qkv_proj.weight', packed)

    def test_timestep_has_cos_then_sin_and_expected_frequencies(self):
        result = self.m.timestep_reference(np.array([0, np.pi/2], np.float32), 4)
        np.testing.assert_allclose(result[0], [1, 1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(result[1], [0, np.cos(np.pi/200), 1, np.sin(np.pi/200)], atol=1e-7)

    def test_inputs_cover_modalities_and_change_sequence_and_values(self):
        inputs = [self.m.make_model_inputs(self.c, r, self.q) for r in range(3)]
        self.assertEqual([x['position_ids'].size for x in inputs], [9, 13, 9])
        for x in inputs:
            all_indices = np.concatenate([x[k+'_indices'] for k in ('video', 'audio', 'text')])
            np.testing.assert_array_equal(np.sort(all_indices), x['position_ids'])
            for tag, kind in enumerate(('video', 'audio', 'text')):
                np.testing.assert_array_equal(x['token_tags'][x[kind+'_indices']], tag)
        self.assertFalse(np.array_equal(inputs[0]['video_hidden_states'], inputs[2]['video_hidden_states']))

    def test_zero_weights_have_known_output_bias(self):
        weights = {k: np.zeros_like(v) for k, v in self.w.items()}
        weights['proj_out.bias'][:] = .25
        weights['audio_proj_out.bias'][:] = -.5
        x = self.m.make_model_inputs(self.c, 0, self.q)
        out = self.m.model_reference(self.c, weights, x, self.q)
        self.assertEqual(out['video'].shape, (1, 3, 16))
        self.assertEqual(out['audio'].shape, (1, 2, 3))
        np.testing.assert_array_equal(out['video'], .25)
        np.testing.assert_array_equal(out['audio'], -.5)

    def test_nonzero_reference_responds_to_new_inputs_without_mutation(self):
        a = self.m.make_model_inputs(self.c, 0, self.q)
        b = self.m.make_model_inputs(self.c, 2, self.q)
        saved = {k: v.copy() for k, v in a.items()}
        first = self.m.model_reference(self.c, self.w, a, self.q)
        second = self.m.model_reference(self.c, self.w, b, self.q)
        self.assertTrue(np.isfinite(first['video']).all())
        self.assertFalse(np.allclose(first['video'], second['video'], atol=.01))
        for k in a:
            np.testing.assert_array_equal(a[k], saved[k])


if __name__ == '__main__':
    unittest.main()
