"""Off/on archives contain both numerical outputs and media-kind text metadata."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class GatedComparisonTests(unittest.TestCase):
    def setUp(self):
        package = ROOT / 'scripts/h3-gated-integration-v1'
        sys.path.insert(0, str(package))
        try:
            spec = importlib.util.spec_from_file_location('gated_comparison_test', package/'verify_integration.py')
            self.api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.api)
        finally:
            sys.path.pop(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for prefix, count in (('h3', 18), ('bridge', 12)):
            for mode in ('off', 'on'):
                folder = self.root/(prefix+'_'+mode)
                folder.mkdir()
                for i in range(count):
                    self.save(folder/f'{i}.npz')

    def save(self, path, *, kind='image', actual=None):
        np.savez(path, actual=np.array([0.5, -2.0], dtype=np.float32) if actual is None else actual,
                 media_0_kind=np.array(kind), input_ids=np.array([1, 2], dtype=np.int64))

    def test_numeric_and_text_fields_are_compared_exactly(self):
        self.assertEqual(len(self.api.compare_runs(self.root)), 30)

    def test_changed_media_metadata_is_rejected(self):
        self.save(self.root/'bridge_on/0.npz', kind='video')
        with self.assertRaises(AssertionError):
            self.api.compare_runs(self.root)

    def test_nonfinite_output_is_rejected_even_when_both_match(self):
        for mode in ('off', 'on'):
            self.save(self.root/f'h3_{mode}/0.npz', actual=np.array([np.inf, 0.0], dtype=np.float32))
        with self.assertRaises(AssertionError):
            self.api.compare_runs(self.root)


if __name__ == '__main__':
    unittest.main()
