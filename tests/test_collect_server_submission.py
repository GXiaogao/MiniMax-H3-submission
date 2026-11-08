import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]

class CollectionTests(unittest.TestCase):
    def test_collects_source_not_weights_and_preserves_input(self):
        spec = importlib.util.spec_from_file_location('collector', ROOT/'scripts/collect_server_submission.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/'work'; src=work/'Projects/InfiniLM'; src.mkdir(parents=True)
            (src/'model.cpp').write_text('int value = 1;')
            (src/'LICENSE').write_text('fixture license')
            (src/'weights.safetensors').write_bytes(b'not source')
            (src/'library.so').write_bytes(b'not source')
            (work/'state').mkdir(); (work/'state/infinilm-extension.success').write_text('fixture=1')
            before={str(p):p.read_bytes() for p in work.rglob('*') if p.is_file()}
            out=Path(tmp)/'output'
            report=mod.collect(work,out,probe=False)
            self.assertEqual(before,{str(p):p.read_bytes() for p in work.rglob('*') if p.is_file()})
            self.assertTrue((out/'sources/InfiniLM/model.cpp').exists())
            self.assertTrue((out/'sources/InfiniLM/LICENSE').exists())
            self.assertFalse((out/'sources/InfiniLM/weights.safetensors').exists())
            self.assertFalse((out/'sources/InfiniLM/library.so').exists())
            self.assertIn('InfiniCore',report['missing_sources'])
            self.assertEqual(json.loads((out/'report.json').read_text())['scope'],'collection-not-build-or-inference-validation')
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('sources/InfiniLM/model.cpp',z.namelist())
    def test_output_inside_source_is_rejected(self):
        spec=importlib.util.spec_from_file_location('collector',ROOT/'scripts/collect_server_submission.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            w=Path(tmp)
            with self.assertRaises(ValueError): mod.collect(w,w/'Projects/InfiniLM/output',probe=False)

if __name__=='__main__': unittest.main()
