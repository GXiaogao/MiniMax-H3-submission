import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'
CPP = ROOT / 'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'


class ConditionPatchTests(unittest.TestCase):
    def setUp(self):
        path = FOLDER / 'prepare_minimax_h3_condition_trace.py'
        self.assertTrue(path.exists(), 'condition patcher missing')
        spec = importlib.util.spec_from_file_location('condition_test', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_pinned_source_and_repeat_patch_rejected(self):
        data = CPP.read_bytes()
        text, meta = self.m.instrument_text(data)
        self.assertEqual(CPP.read_bytes(), data)
        self.assertEqual(len(meta['boundaries']), 24)
        self.assertEqual(len(set(meta['boundaries'])), 24)
        self.assertIn('condition_input', meta['boundaries'])
        self.assertIn('condition_final', meta['boundaries'])
        for data in (data + b'\n// drift', text.encode()):
            with self.assertRaises(ValueError):
                self.m.instrument_text(data)

    def test_each_refiner_boundary_uses_current_block_index(self):
        text, meta = self.m.instrument_text(CPP.read_bytes())
        block = text.split('MiniMaxH3TokenRefinerBlock::forward(', 1)[1].split('MiniMaxH3TokenRefiner::MiniMaxH3TokenRefiner', 1)[0]
        for label in ('norm1', 'attention_output', 'after_attention', 'norm2', 'mlp_output', 'after_mlp'):
            self.assertEqual(block.count(f'save_condition("{label}"'), 1)
            for i in (0, 1):
                self.assertIn(f'condition_{i}_{label}', meta['boundaries'])
        self.assertIn('boundary_trace::condition_block = trace_block++;', text)
        self.assertIn('transformer_blocks_.size()!=1', text)

    def test_patch_backup_and_manifest_match_actual_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'h3.cpp'
            original = CPP.read_bytes()
            path.write_bytes(original)
            meta = self.m.patch_file(path)
            self.assertEqual(Path(meta['backup']).read_bytes(), original)
            self.assertEqual(meta['patched_sha256'], self.m.sha(path.read_bytes()))
            self.assertEqual(meta['helper_sha256'], self.m.sha((path.parent / 'h3_boundary_trace.hpp').read_bytes()))
            self.assertTrue(path.with_suffix('.cpp.trace.json').exists())

    def test_builder_cannot_publish_over_official_or_previous_boundary(self):
        data = (FOLDER / 'build_infinilm_extension.sh').read_bytes()
        text, meta = self.m.builder_text(data)
        self.assertEqual((FOLDER / 'build_infinilm_extension.sh').read_bytes(), data)
        self.assertEqual(meta['destination_root'], '${WORK_HOME}/diagnostics/h3-condition/${H3_BOUNDARY_BUILD_ID}/InfiniLM')
        self.assertEqual(meta['success_marker'], '${STATE_DIR}/infinilm-condition.success')
        self.assertIn('BUILD_VERSION="1.0-isolated-h3-condition-trace"', text)
        self.assertNotIn('SUCCESS_MARKER="${STATE_DIR}/infinilm-extension.success"', text)
        self.assertNotIn('SUCCESS_MARKER="${STATE_DIR}/infinilm-boundary.success"', text)
        with self.assertRaises(ValueError):
            self.m.builder_text(b'unsupported')

    def test_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'build.sh'
            target.write_text('preserve')
            with self.assertRaises(FileExistsError):
                self.m.write_builder(FOLDER / 'build_infinilm_extension.sh', target)
            self.assertEqual(target.read_text(), 'preserve')
