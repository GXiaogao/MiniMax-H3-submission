import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class QwenEncoderReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'scripts/qwen3_vl_reference.py'
        if not path.exists():
            raise AssertionError('Missing Qwen3-VL encoder reference implementation')
        import qwen3_vl_reference
        cls.r = qwen3_vl_reference

    def test_config_and_parameter_shapes(self):
        c = self.r.make_config('float32')
        w = self.r.make_weights(c, lambda a: a)
        self.assertEqual(c['text_config']['num_hidden_layers'], 3)
        self.assertEqual(w['model.visual.patch_embed.proj.weight'].shape, (64, 3, 2, 2, 2))
        self.assertEqual(w['model.visual.deepstack_merger_list.0.norm.weight'].shape, (256,))
        self.assertEqual(w['model.visual.merger.norm.weight'].shape, (64,))
        self.assertEqual(w['model.language_model.layers.0.self_attn.k_proj.weight'].shape, (64, 64))
        self.assertNotIn('model.language_model.norm.weight', w)
        self.assertNotIn('lm_head.weight', w)
        self.assertFalse(any('.layers.2.' in k for k in w))

    def test_non_square_patch_order(self):
        rows, cols = self.r.patch_positions((1, 2, 4), 2)
        np.testing.assert_array_equal(rows, [0, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(cols, [0, 1, 0, 1, 2, 3, 2, 3])

    def test_bilinear_known_plane_and_endpoints(self):
        table = np.array([[[0.], [2.]], [[10.], [12.]]], np.float32)
        actual = self.r.interpolate_positions(table, (1, 2, 4), 2)
        expected = [0, 2/3, 10, 10+2/3, 4/3, 2, 10+4/3, 12]
        np.testing.assert_allclose(actual[:, 0], expected, atol=2e-6)

    def test_request_metadata_known_image_and_video(self):
        c = self.r.make_config('float32')
        image = self.r.make_case(c, 1, lambda a: a)
        self.assertEqual(image['input_ids'].shape, (1, 11))
        expected = np.array([[2,2,2,2,2,2], [2,2,2,3,3,3], [2,3,4,2,3,4]])
        np.testing.assert_array_equal(image['positions'][:, 2:8], expected)
        np.testing.assert_array_equal(image['positions'][:, 8:], [[5,6,7]]*3)
        video = self.r.make_case(c, 0, lambda a: a, video=True)
        self.assertEqual(video['media'][0]['bounds'], [(2,4), (5,7)])
        np.testing.assert_array_equal(video['positions'][:, 2:4], [[2,2],[2,2],[2,3]])
        np.testing.assert_array_equal(video['positions'][:, 5:7], [[5,5],[5,5],[5,6]])

    def test_causal_gqa_reference_has_no_future_leak(self):
        q = np.zeros((3, 2, 4), np.float32)
        k = np.zeros((3, 1, 4), np.float32)
        v = np.broadcast_to(np.array([2,4,9], np.float32)[:,None,None], (3,1,4)).copy()
        out = self.r.attention(q, k, v, causal=True)
        np.testing.assert_allclose(out[:,0,0], [2,3,5])
        np.testing.assert_array_equal(out[:,0], out[:,1])

    def test_interleaved_rotary_axes_and_identity(self):
        x = np.arange(24, dtype=np.float32).reshape(1,2,12)/20
        z = self.r.text_rotary(x, np.zeros((3,1),np.int64), [2,2,2], 10000., lambda a:a)
        np.testing.assert_array_equal(x,z)
        positions = np.array([[0],[1],[2]], np.int64)
        out = self.r.text_rotary(x, positions, [2,2,2], 10000., lambda a:a)
        np.testing.assert_array_equal(out[..., [0,3,6,9]], x[..., [0,3,6,9]])
        self.assertGreater(np.max(np.abs(out-x)), .01)

    def test_video_visual_reference_is_frame_isolated(self):
        c = self.r.make_config('float32')
        w = self.r.make_weights(c, lambda a:a)
        case = self.r.make_case(c, 0, lambda a:a, video=True)
        media = case['media'][0]
        pooled, deep = self.r.vision_reference(c, w, media, lambda a:a)
        altered = dict(media, pixels=media['pixels'].copy())
        altered['pixels'][8:] += 1
        p2, d2 = self.r.vision_reference(c, w, altered, lambda a:a)
        np.testing.assert_array_equal(pooled[:2], p2[:2])
        for a,b in zip(deep,d2):
            np.testing.assert_array_equal(a[:2], b[:2])
        self.assertGreater(np.max(np.abs(pooled[2:]-p2[2:])), 1e-5)

    def test_full_reference_zero_weights_and_changed_inputs(self):
        c = self.r.make_config('float32')
        w = self.r.make_weights(c, lambda a:a)
        x = self.r.make_case(c, 2, lambda a:a)
        zero = {k: np.zeros_like(v) for k,v in w.items()}
        np.testing.assert_array_equal(self.r.encoder_reference(c,zero,x,lambda a:a), np.zeros((1,10,64)))
        y = self.r.encoder_reference(c,w,x,lambda a:a)
        changed = dict(x, input_ids=x['input_ids'].copy())
        changed['input_ids'][0,-1] = (int(changed['input_ids'][0,-1])+1)%64
        y2 = self.r.encoder_reference(c,w,changed,lambda a:a)
        self.assertEqual(y.shape,(1,10,64))
        self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(y[:,:-1], y2[:,:-1])
        self.assertGreater(np.max(np.abs(y[:,-1]-y2[:,-1])), 1e-5)


if __name__ == '__main__':
    unittest.main()
