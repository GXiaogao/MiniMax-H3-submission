"""CPU contracts for the tiny Encoder -> native H3 validation harness."""
from pathlib import Path
import hashlib
import sys
import unittest
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_h3_encoder_bridge as bridge
import qwen3_vl_reference as qwen
import verify_h3_native_integration as h3


class EncoderBridgeTests(unittest.TestCase):
    def setUp(self):
        self.q = lambda a: np.asarray(a, np.float32)
        self.ec, self.hc = bridge.make_configs('float32')

    def test_full_encoder_width_is_used_without_projection_or_truncation(self):
        self.assertEqual(self.ec['text_config']['hidden_size'], 64)
        self.assertEqual(self.hc['text_dim'], 64)
        self.assertEqual(self.hc['num_layers'], 1)
        self.assertEqual(qwen.LAYERS, 2)
        self.assertEqual(h3.make_weights(self.hc, self.q)['context_embedder.weight'].shape, (64, 64))

    def test_variable_sequences_and_modality_packing_preserve_features(self):
        sequences = []
        for i, case in enumerate(bridge.make_cases(self.ec, self.q)):
            n = case['input_ids'].size
            sequences.append(n)
            features = np.arange(n*64, dtype=np.float32).reshape(1, n, 64)
            x = bridge.make_bridge_inputs(self.hc, features, i, self.q)
            np.testing.assert_array_equal(x['encoder_hidden_states'], features)
            self.assertFalse(np.shares_memory(x['encoder_hidden_states'], features))
            merged = np.concatenate([x[k+'_indices'] for k in ('video', 'audio', 'text')])
            np.testing.assert_array_equal(np.sort(merged), x['position_ids'])
            for tag, k in enumerate(('video', 'audio', 'text')):
                idx = x[k+'_indices']
                self.assertEqual(idx.size, x[{'video':'video_hidden_states', 'audio':'audio_hidden_states', 'text':'encoder_hidden_states'}[k]].shape[1])
                np.testing.assert_array_equal(x['token_tags'][idx], tag)
            self.assertTrue(np.all(x['timestep_indices'] < x['timestep'].size))
        self.assertEqual(sequences, [7, 11, 9, 5])

    def test_invalid_features_are_rejected_before_native_calls(self):
        for a in (np.zeros((7,64),np.float32), np.zeros((2,7,64),np.float32),
                  np.zeros((1,0,64),np.float32), np.zeros((1,7,12),np.float32),
                  np.zeros((1,7,64),np.int64), np.full((1,7,64),np.nan,np.float32)):
            with self.subTest(shape=a.shape, dtype=str(a.dtype)):
                with self.assertRaises(ValueError):
                    bridge.make_bridge_inputs(self.hc, a, 0, self.q)

    def test_reference_outputs_depend_on_condition_and_leave_inputs_unchanged(self):
        ew = qwen.make_weights(self.ec, self.q)
        hw = h3.make_weights(self.hc, self.q)
        case = bridge.make_cases(self.ec, self.q)[1]
        condition = qwen.encoder_reference(self.ec, ew, case, self.q)
        x = bridge.make_bridge_inputs(self.hc, condition, 1, self.q)
        before = {k:v.copy() for k,v in x.items()}
        out = h3.model_reference(self.hc, hw, x, self.q)
        changed = dict(x, encoder_hidden_states=-condition)
        other = h3.model_reference(self.hc, hw, changed, self.q)
        for kind in ('video','audio'):
            self.assertTrue(np.isfinite(out[kind]).all())
            self.assertGreater(float(np.max(np.abs(out[kind]-other[kind]))), 1e-5)
        for k in x:
            np.testing.assert_array_equal(x[k], before[k])

    def test_zero_backbone_reference_has_known_output_bias(self):
        weights = {k:np.zeros_like(v) for k,v in h3.make_weights(self.hc,self.q).items()}
        weights['proj_out.bias'][:] = .25
        weights['audio_proj_out.bias'][:] = -.5
        x = bridge.make_bridge_inputs(self.hc, np.ones((1,9,64),np.float32), 2, self.q)
        out = h3.model_reference(self.hc,weights,x,self.q)
        np.testing.assert_array_equal(out['video'], .25)
        np.testing.assert_array_equal(out['audio'], -.5)

    def test_each_dtype_detects_zero_or_inverted_condition_at_actual_tolerance(self):
        def quantizer(dtype):
            def q(a):
                a = np.asarray(a, np.float32)
                if dtype == 'float32':
                    return a.copy()
                if dtype == 'float16':
                    return a.astype(np.float16).astype(np.float32)
                # Round finite FP32 to BF16, nearest-even, without needing Torch.
                u = a.copy().view(np.uint32)
                u = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
                return u.view(np.float32)
            return q
        for dtype in qwen.DTYPES:
            q = quantizer(dtype)
            ec, hc = bridge.make_configs(dtype)
            ew = qwen.make_weights(ec, q)
            hw = bridge.make_backbone_weights(hc, q)
            for i, case in enumerate(bridge.make_cases(ec, q)):
                condition = qwen.encoder_reference(ec, ew, case, q)
                x = bridge.make_bridge_inputs(hc, condition, i, q)
                expected = h3.model_reference(hc, hw, x, q)
                for label, wrong in (('zero',np.zeros_like(condition)), ('inverted',-condition)):
                    actual = h3.model_reference(hc, hw, dict(x,encoder_hidden_states=wrong), q)
                    for kind in ('video','audio'):
                        with self.subTest(dtype=dtype, case=case['name'], mutation=label, output=kind):
                            with self.assertRaises(AssertionError):
                                bridge.compare_arrays(actual[kind],expected[kind],*h3.TOLERANCES[dtype])

    def test_release_checksum_manifest_is_linux_compatible_and_exact(self):
        archive = ROOT / 'artifacts' / 'h3-encoder-bridge-v1.zip'
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as package:
            prefix = 'h3-encoder-bridge-v1/'
            checksum_bytes = package.read(prefix+'SHA256SUMS')
            self.assertNotIn(b'\r', checksum_bytes)
            self.assertFalse(checksum_bytes.startswith(b'\xef\xbb\xbf'))
            lines = checksum_bytes.decode('ascii').splitlines()
            self.assertEqual(len(lines), 8)
            for line in lines:
                digest, name = line.split('  ', 1)
                self.assertNotIn('/', name)
                self.assertEqual(hashlib.sha256(package.read(prefix+name)).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
