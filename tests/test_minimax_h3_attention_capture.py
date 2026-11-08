import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'


class AttentionCaptureTests(unittest.TestCase):
    def setUp(self):
        path = FOLDER / 'diagnose_minimax_h3_attention.py'
        self.assertTrue(path.exists(), 'attention capture missing')
        spec = importlib.util.spec_from_file_location('attention_capture_test', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_head_shards_slice_each_token_before_flattening(self):
        x = np.arange(1 * 228 * 56 * 128, dtype=np.float32).reshape(1, 228, 56, 128)
        for rank in (0, 3, 7):
            expected = x[:, :, rank * 7:(rank + 1) * 7, :]
            actual = self.m.reference_shard(x.reshape(1, 228, -1), 'q_raw', 8, rank)
            np.testing.assert_array_equal(actual, expected.reshape(1, 228, 896))
            normalized = self.m.reference_shard(x, 'q_norm', 8, rank)
            np.testing.assert_array_equal(normalized, expected)
        np.testing.assert_array_equal(self.m.reference_shard(x, 'mha', 1, 0), x.reshape(1, 228, 7168))

    def test_replicated_stages_are_not_sliced_and_bad_layout_rejected(self):
        x = np.ones((1, 228, 5376), dtype=np.float32)
        np.testing.assert_array_equal(self.m.reference_shard(x, 'projected', 8, 7), x)
        with self.assertRaises(ValueError):
            self.m.reference_shard(x, 'q_raw', 8, 0)
        for tp, rank in ((2, 0), (8, 8), (1, -1)):
            with self.assertRaises(ValueError):
                self.m.reference_shard(x, 'input', tp, rank)

    def test_rank_roundtrip_and_metadata_or_data_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            x = np.arange(228 * 896, dtype=np.float32).reshape(1, 228, 896)
            self.m.save_rank(folder, 'q_raw', 8, 3, x, 'bfloat16')
            samples, meta = self.m.read_rank(folder, 'q_raw', 8, 3)
            np.testing.assert_array_equal(samples, x.ravel()[self.m.sample_indices(x.size, 65536)])
            self.assertEqual(meta['rank'], 3)
            self.assertEqual(meta['tp_size'], 8)
            path = folder / 'refiner0_q_raw.rank3.json'
            original = path.read_text()
            changed = json.loads(original)
            changed['rank'] = 2
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.m.read_rank(folder, 'q_raw', 8, 3)
            path.write_text(original)
            (folder / 'refiner0_q_raw.rank3.f32').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                self.m.read_rank(folder, 'q_raw', 8, 3)

    def test_shape_nonfinite_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            x = np.zeros((1, 228, 7, 128), dtype=np.float32)
            self.m.save_rank(folder, 'q_norm', 8, 0, x, 'float32')
            with self.assertRaises(FileExistsError):
                self.m.save_rank(folder, 'q_norm', 8, 0, x, 'float32')
            with self.assertRaises(ValueError):
                self.m.save_rank(folder, 'k_norm', 8, 0, x.reshape(1, 228, 896), 'float32')
            x[:] = np.nan
            with self.assertRaises(ValueError):
                self.m.save_rank(folder, 'v_raw', 8, 0, x.reshape(1, 228, 896), 'float32')

    def test_correctly_aligned_native_and_reference_shard_compare_exact(self):
        with tempfile.TemporaryDirectory() as d:
            native = Path(d) / 'native'
            reference = Path(d) / 'ref'
            native.mkdir()
            reference.mkdir()
            x = np.arange(228 * 7168, dtype=np.float32).reshape(1, 228, 7168)
            shard = self.m.reference_shard(x, 'mha', 8, 6)
            self.m.save_rank(native, 'mha', 8, 6, shard, 'float32')
            self.m.save_rank(reference, 'mha', 8, 6, shard, 'float32')
            result = self.m.compare_rank(reference, native, 'mha', 8, 6)
            self.assertEqual(result['relative_l2'], 0)
            self.assertEqual(result['scope'], 'saved-rank-head-aligned-samples')

    def test_missing_rank_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                self.m.validate_rank_set(Path(d), 8)

    def test_reference_hooks_capture_mha_before_projection_and_slice_q_heads(self):
        from tests.test_minimax_h3_condition_capture import Node
        model = type('Model', (), {})()
        model.context_embedder = Node()
        model.time_embedder = Node()
        model.token_refiner = type('Refiner', (), {})()
        blocks = []
        for _ in range(2):
            block = Node()
            block.norm1, block.norm2, block.attn, block.ff = Node(), Node(), Node(), Node()
            blocks.append(block)
        model.token_refiner.refiner_blocks = blocks
        model.token_refiner.final_norm = Node()
        main = Node()
        main.adaln_proj = type('Projection', (), {'linear': Node()})()
        main.attn, main.norm2, main.ff = Node(), Node(), Node()
        model.transformer_blocks = [main]
        attn = blocks[0].attn
        values = np.arange(228 * 7168, dtype=np.float32).reshape(1, 228, 7168)
        attn.to_q, attn.to_k, attn.to_v = Node(lambda x: values), Node(), Node()
        attn.norm_q, attn.norm_k = Node(), Node()
        attn.to_out = [Node(lambda x: x[:, :, :5376] + 100), Node()]
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            handles = self.m.attach_reference(model, folder)
            attn.to_q(np.zeros((1, 228, 5376), dtype=np.float32))
            attn.to_out[0](values)
            q, _ = self.m.read_rank(folder / 'tp8', 'q_raw', 8, 4)
            mha, _ = self.m.read_rank(folder / 'tp8', 'mha', 8, 4)
            projected, _ = self.m.read_rank(folder / 'tp8', 'projected', 8, 4)
            expected = self.m.reference_shard(values, 'q_raw', 8, 4).ravel()
            idx = self.m.sample_indices(expected.size, 65536)
            np.testing.assert_array_equal(q, expected[idx])
            np.testing.assert_array_equal(mha, q)
            full_projected = (values[:, :, :5376] + 100).ravel()
            np.testing.assert_array_equal(projected, full_projected[self.m.sample_indices(full_projected.size, 65536)])
            for handle in handles:
                handle.remove()
            self.assertFalse(attn.to_out[0].pre)

    def test_wrong_binary_rejected_before_reading_source_or_loading_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            ext = work / self.m.EXTENSION_RELATIVE
            ext.parent.mkdir(parents=True)
            ext.write_bytes(b'wrong')
            run = work / self.m.BUILD_LOG_RELATIVE
            run.mkdir(parents=True)
            (run / 'artifacts.sha256').write_text(self.m.base.sha256(ext) + '  ' + str(ext) + '\n')
            (work / 'state').mkdir()
            marker = {'build_version': '1.0-isolated-h3-attention-trace', 'cxx11_abi': '1', 'extension': str(ext), 'run_dir': str(run), 'artifacts_sha256': str(run / 'artifacts.sha256')}
            (work / 'state/infinilm-attention.success').write_text(''.join(f'{k}={v}\n' for k, v in marker.items()))
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.m.validate_diagnostic(work)

    def test_valid_build_identity_and_sources_checked_after_pinned_binary(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            ext = work / self.m.EXTENSION_RELATIVE
            ext.parent.mkdir(parents=True)
            ext.write_bytes(b'fixture, not actual compiled binary')
            run = work / self.m.BUILD_LOG_RELATIVE
            run.mkdir(parents=True)
            (run / 'artifacts.sha256').write_text(self.m.EXPECTED_EXTENSION + '  ' + str(ext) + '\n')
            spec = importlib.util.spec_from_file_location('fixture_attention_patcher', FOLDER / 'prepare_minimax_h3_attention_trace.py')
            patcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patcher)
            cpp = ROOT / 'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'
            text, meta = patcher.instrument_text(cpp.read_bytes())
            (run / 'minimax_h3_transformer.cpp').write_text(text, encoding='utf-8', newline='\n')
            (run / 'minimax_h3_transformer.cpp.trace.json').write_text(json.dumps(meta))
            official = work / 'Projects/InfiniLM/python/infinilm'
            official.mkdir(parents=True)
            (official / '__init__.py').write_text('# same')
            (ext.parents[1] / '__init__.py').write_text('# same')
            (work / 'state').mkdir()
            marker = {'build_version': '1.0-isolated-h3-attention-trace', 'cxx11_abi': '1', 'extension': str(ext), 'run_dir': str(run), 'artifacts_sha256': str(run / 'artifacts.sha256')}
            (work / 'state/infinilm-attention.success').write_text(''.join(f'{k}={v}\n' for k, v in marker.items()))
            real_hash = self.m.base.sha256
            with patch.object(self.m.base, 'sha256', side_effect=lambda path: self.m.EXPECTED_EXTENSION if Path(path).resolve() == ext.resolve() else real_hash(path)):
                extension, _, evidence = self.m.validate_diagnostic(work)
                self.assertEqual(extension, ext.resolve())
                self.assertEqual(evidence['python_files_checked'], 1)
                (ext.parents[1] / '__init__.py').write_text('# drift')
                with self.assertRaisesRegex(ValueError, 'Python copy'):
                    self.m.validate_diagnostic(work)

    def test_failed_preflight_retains_failed_report(self):
        with tempfile.TemporaryDirectory() as d:
            rc = self.m.main(['--run-dir', d, '--work-home', d, '--local-target', d + '/local', '--model-path', d + '/model'])
            self.assertEqual(rc, 1)
            report = json.loads((Path(d) / 'attention-parity.json').read_text())
            self.assertEqual(report['result'], 'FAILED')
            self.assertEqual(report['completed_workers'], [])
