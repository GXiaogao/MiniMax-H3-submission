import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'
CPP=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'

class BoundaryPatchTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'prepare_minimax_h3_boundary_trace.py'
        self.assertTrue(path.exists(),'boundary patcher missing')
        spec=importlib.util.spec_from_file_location('boundary_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_build_variant_has_separate_publication_and_retains_original(self):
        source=(FOLDER/'build_infinilm_extension.sh').read_bytes()
        text,meta=self.m.builder_text(source)
        self.assertEqual((FOLDER/'build_infinilm_extension.sh').read_bytes(),source)
        self.assertEqual(meta['destination_root'],'${WORK_HOME}/diagnostics/h3-boundary/${H3_BOUNDARY_BUILD_ID}/InfiniLM')
        self.assertEqual(meta['success_marker'],'${STATE_DIR}/infinilm-boundary.success')
        self.assertNotIn('SUCCESS_MARKER="${STATE_DIR}/infinilm-extension.success"',text)
        self.assertNotIn('PERSISTENT_PROJECT_DIR="${WORK_HOME}/Projects/InfiniLM"',text)

    def test_source_drift_and_double_patch_rejected(self):
        original=CPP.read_bytes()
        text,meta=self.m.instrument_text(original)
        self.assertEqual(meta['sampling'],'linear-integer-indices-at-most-65536')
        self.assertEqual(len(meta['boundaries']),9)
        self.assertEqual(CPP.read_bytes(),original)
        for data in (original+b'\n//drift',text.encode()):
            with self.assertRaises(ValueError):self.m.instrument_text(data)
        with self.assertRaises(ValueError):self.m.builder_text(b'unsupported')

    def test_file_patch_keeps_backup_and_json_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'h3.cpp';original=CPP.read_bytes();path.write_bytes(original)
            meta=self.m.patch_file(path)
            self.assertEqual(Path(meta['backup']).read_bytes(),original)
            self.assertTrue(path.with_suffix('.cpp.trace.json').exists())
            self.assertNotEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):self.m.patch_file(path)

    def test_builder_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'build.sh';target.write_text('keep')
            with self.assertRaises(FileExistsError):self.m.write_builder(FOLDER/'build_infinilm_extension.sh',target)
            self.assertEqual(target.read_text(),'keep')
