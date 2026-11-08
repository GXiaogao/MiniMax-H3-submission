import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'
CPP = ROOT / 'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'


class AttentionPatchTests(unittest.TestCase):
    def setUp(self):
        path = FOLDER / 'prepare_minimax_h3_attention_trace.py'
        self.assertTrue(path.exists(), 'attention patcher missing')
        spec = importlib.util.spec_from_file_location('attention_patch_test', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_source_pin_and_instrumentation_manifest(self):
        data = CPP.read_bytes()
        text, meta = self.m.instrument_text(data)
        self.assertEqual(CPP.read_bytes(), data)
        self.assertEqual(len(meta['boundaries']), 24)
        self.assertEqual(meta['attention_boundaries'], ['input', 'q_raw', 'k_raw', 'v_raw', 'q_norm', 'k_norm', 'mha', 'projected'])
        self.assertTrue(meta['all_tp_ranks'])
        for invalid in (data + b'//drift', text.encode()):
            with self.assertRaises(ValueError):
                self.m.instrument_text(invalid)

    def test_scope_is_only_around_refiner_attention_and_retains_real_projection(self):
        text, _ = self.m.instrument_text(CPP.read_bytes())
        attention = text.split('MiniMaxH3Attention::forward(', 1)[1].split('void MiniMaxH3Attention::process_weights', 1)[0]
        for stage in self.m.ATTENTION_BOUNDARIES:
            self.assertEqual(attention.count(f'attention_trace::save("{stage}"'), 1)
        self.assertLess(attention.index('save("q_raw"'), attention.index('norm_q_->forward'))
        self.assertLess(attention.index('save("q_norm"'), attention.index('infinicore::op::mha'))
        self.assertLess(attention.index('save("mha"'), attention.index('linear_forward(out_proj_'))
        self.assertEqual(attention.count('linear_forward(out_proj_, output)'), 1)
        self.assertNotIn('Scope', attention)
        refiner = text.split('MiniMaxH3TokenRefinerBlock::forward(', 1)[1].split('MiniMaxH3TokenRefiner::MiniMaxH3TokenRefiner', 1)[0]
        self.assertIn('attention_trace::Scope scope(boundary_trace::condition_block == 0);', refiner)
        self.assertEqual(text.count('attention_trace::Scope scope('), 1)
        self.assertIn('transformer_blocks_.size()!=1', text)

    def test_patch_keeps_original_and_both_header_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'h3.cpp'
            data = CPP.read_bytes()
            path.write_bytes(data)
            meta = self.m.patch_file(path)
            self.assertEqual(Path(meta['backup']).read_bytes(), data)
            self.assertEqual(meta['patched_sha256'], self.m.sha(path.read_bytes()))
            for name, digest in meta['helper_hashes'].items():
                self.assertEqual(self.m.sha((path.parent / name).read_bytes()), digest)
            self.assertEqual(len(meta['helper_hashes']), 2)
            saved = json.loads(path.with_suffix('.cpp.trace.json').read_text())
            self.assertEqual(saved['attention_boundaries'], self.m.ATTENTION_BOUNDARIES)

    def test_builder_preserves_previous_destinations_and_pins_original(self):
        data = (FOLDER / 'build_infinilm_extension.sh').read_bytes()
        text, meta = self.m.builder_text(data)
        self.assertEqual(meta['destination_root'], '${WORK_HOME}/diagnostics/h3-attention/${H3_BOUNDARY_BUILD_ID}/InfiniLM')
        self.assertEqual(meta['success_marker'], '${STATE_DIR}/infinilm-attention.success')
        self.assertIn('h3_attention_trace.hpp', text)
        self.assertIn('BUILD_VERSION="1.0-isolated-h3-attention-trace"', text)
        for old in ('infinilm-extension.success', 'infinilm-boundary.success', 'infinilm-condition.success'):
            self.assertNotIn('SUCCESS_MARKER="${STATE_DIR}/' + old + '"', text)
        with self.assertRaises(ValueError):
            self.m.builder_text(b'unknown')

    def test_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'build.sh'
            target.write_text('keep')
            with self.assertRaises(FileExistsError):
                self.m.write_builder(FOLDER / 'build_infinilm_extension.sh', target)
            self.assertEqual(target.read_text(), 'keep')
