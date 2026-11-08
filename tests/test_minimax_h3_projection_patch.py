import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'
BASE=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM'
CPP=BASE/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'
QUANT=BASE/'csrc/layers/quantization/none_quantization.cpp'


class ProjectionPatchTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'prepare_minimax_h3_projection_trace.py'
        self.assertTrue(path.exists(),'projection patcher missing')
        spec=importlib.util.spec_from_file_location('projection_patch_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_pinned_sources_scope_and_no_original_modification(self):
        original=CPP.read_bytes();text,meta=self.m.instrument_text(original)
        self.assertEqual(CPP.read_bytes(),original)
        self.assertIn('projection_trace::Scope scope(attention_trace::active)',text)
        self.assertEqual(text.count('projection_trace::Scope scope('),1)
        self.assertIn('return linear_forward(out_proj_, output);',text)
        self.assertEqual(len(meta['projection_boundaries']),5)
        for bad in (original+b'//drift',text.encode()):
            with self.assertRaises(ValueError):self.m.instrument_text(bad)
        with self.assertRaises(ValueError):self.m.instrument_quant(QUANT.read_bytes()+b'//drift')

    def test_original_projection_precedes_control_and_is_returned(self):
        text=self.m.instrument_quant(QUANT.read_bytes())
        allreduce=text.split('NoneQuantization::forward_allreduce(',1)[1].split('std::vector<SplitParam>',1)[0]
        self.assertLess(allreduce.index('infinicore::op::linear_allreduce('),allreduce.index('inspect_control('))
        self.assertIn('return original;',allreduce)
        self.assertIn('linear_allreduce_packed',allreduce)
        self.assertIn('../../models/minimax_h3/h3_projection_trace.hpp',text)
        self.assertIn('inspect_serial(',text)

    def test_both_originals_and_headers_preserved_with_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            h=Path(d)/'minimax_h3_transformer.cpp';q=Path(d)/'none_quantization.cpp'
            h.write_bytes(CPP.read_bytes());q.write_bytes(QUANT.read_bytes())
            meta=self.m.patch_files(h,q)
            self.assertEqual(Path(meta['backup']).read_bytes(),CPP.read_bytes())
            self.assertEqual(Path(meta['quant_backup']).read_bytes(),QUANT.read_bytes())
            self.assertEqual(meta['quant_patched_sha256'],self.m.sha(q.read_bytes()))
            self.assertEqual(len(meta['helper_hashes']),3)
            with self.assertRaises(ValueError):self.m.patch_files(h,q)

    def test_bad_quant_source_does_not_partially_patch_h3(self):
        with tempfile.TemporaryDirectory() as d:
            h=Path(d)/'h3.cpp';q=Path(d)/'quant.cpp';data=CPP.read_bytes()
            h.write_bytes(data);q.write_bytes(b'unknown')
            with self.assertRaises(ValueError):self.m.patch_files(h,q)
            self.assertEqual(h.read_bytes(),data)
            self.assertFalse(h.with_suffix('.cpp.before-projection-trace').exists())

    def test_builder_isolates_publication_and_includes_quant_source(self):
        text,meta=self.m.builder_text((FOLDER/'build_infinilm_extension.sh').read_bytes())
        self.assertEqual(meta['success_marker'],'${STATE_DIR}/infinilm-projection.success')
        self.assertIn('/diagnostics/h3-projection/',text)
        self.assertIn('--quant-source',text)
        self.assertIn('h3_projection_trace.hpp',text)
        self.assertIn('none_quantization.cpp',text)
        self.assertNotIn('SUCCESS_MARKER="${STATE_DIR}/infinilm-extension.success"',text)
