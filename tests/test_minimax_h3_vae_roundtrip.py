"""Behavior checks for the isolated diagnostic, without GPU/model loading."""
import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'artifacts/infinifabric-build-v1/diagnose_minimax_h3_vae_roundtrip.py'


def load_module():
    spec = importlib.util.spec_from_file_location('vae_roundtrip_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoundtripTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_first_keyframe_resize_retains_top_and_bottom_like_fl2va(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            pixels = np.zeros((8, 4, 3), dtype=np.uint8)
            pixels[2:6, :, 0] = 255
            Image.fromarray(pixels).save(path)
            actual = np.asarray(self.module.prepare_image(path, width=4, height=4))
            self.assertEqual(actual.shape, (4, 4, 3))
            self.assertLess(int(actual[0, 0, 0]), 64)
            self.assertLess(int(actual[-1, 0, 0]), 64)
            self.assertGreater(int(actual[1, 0, 0]), 200)
            self.assertGreater(int(actual[2, 0, 0]), 200)
            self.assertEqual(int(actual[:, :, 1:].max()), 0)

    def test_metrics_known_error_and_exact_copy(self):
        expected = np.zeros((2, 2, 3), dtype=np.float32)
        result = self.module.reconstruction_metrics(expected, expected + 0.1)
        self.assertAlmostEqual(result['mae'], 0.1, places=6)
        self.assertAlmostEqual(result['mse'], 0.01, places=6)
        self.assertAlmostEqual(result['psnr_db'], 20, places=5)
        exact = self.module.reconstruction_metrics(expected, expected)
        self.assertTrue(exact['exact_match'])
        self.assertIsNone(exact['psnr_db'])
        json.dumps(exact, allow_nan=False)

    def test_metrics_reject_invalid_values_and_shapes(self):
        pixels = np.zeros((2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.module.reconstruction_metrics(pixels, pixels + np.nan)
        with self.assertRaises(ValueError):
            self.module.reconstruction_metrics(pixels, pixels[:1])

    def test_parameter_audit_detects_required_float32_violation(self):
        def parameter(dtype):
            return SimpleNamespace(dtype=dtype, device='cuda:0', numel=lambda: 4)
        model = SimpleNamespace(
            _keep_in_fp32_modules=['encoder', 'norm1'],
            named_parameters=lambda: iter([
                ('encoder.conv.weight', parameter('torch.bfloat16')),
                ('decoder.blocks.0.norm1.weight', parameter('torch.float32')),
                ('decoder.proj.weight', parameter('torch.bfloat16')),
            ]),
        )
        result = self.module.audit_parameters(model)
        self.assertEqual(result['fp32_violations'], ['encoder.conv.weight'])
        self.assertEqual(result['dtype_elements']['torch.bfloat16'], 8)

    def run_main(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'result'
            args = ['--run-dir', str(run_dir), '--local-target', str(Path(directory) / 'local'),
                    '--model-path', directory, '--first-frame', str(Path(directory) / 'image.jpg'),
                    '--work-home', directory]
            def fake_diagnostic(arguments, report, checkpoint):
                report['last_stage'] = 'vae-decode'
                report['parameter_audit'] = {'fp32_violations': []}
                checkpoint()
                if failure:
                    raise RuntimeError('injected decode failure')
                report['reconstruction_frames'] = 22
            with patch.object(self.module, 'run_diagnostic', side_effect=fake_diagnostic), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = self.module.main(args)
            result = json.loads((run_dir / 'vae-roundtrip.json').read_text(encoding='utf-8'))
            return code, result

    def test_success_is_only_diagnostic_completion(self):
        code, result = self.run_main(False)
        self.assertEqual(code, 0)
        self.assertEqual(result['result'], 'DIAGNOSTIC_COMPLETE')
        self.assertEqual(result['quality_status'], 'UNASSESSED')
        self.assertEqual(result['validation_scope'], 'isolated-vae-roundtrip-not-generation')

    def test_failure_preserves_stage_and_partial_evidence(self):
        code, result = self.run_main(True)
        self.assertEqual(code, 1)
        self.assertEqual(result['result'], 'FAILED')
        self.assertEqual(result['last_stage'], 'vae-decode')
        self.assertEqual(result['parameter_audit']['fp32_violations'], [])
        self.assertIn('injected decode failure', result['error'])


if __name__ == '__main__':
    unittest.main()
