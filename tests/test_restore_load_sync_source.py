import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class RestoreTests(unittest.TestCase):
    def test_real_archive_matches_recorded_server_source_hashes(self):
        spec=importlib.util.spec_from_file_location('restore',ROOT/'scripts/restore_load_sync_source.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            result=mod.restore(ROOT,Path(d)/'candidate')
            self.assertEqual(result['result'],'SOURCE_HASHES_MATCH_SERVER_TRACE')
            self.assertEqual(len(result['verified_hashes']),11)
            self.assertFalse(result['binary_rebuild_verified'])
            rank=Path(d)/'candidate/csrc/engine/rank_worker.cpp'
            self.assertIn('weight_load_sync::stage',rank.read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):mod.restore(ROOT,Path(d)/'candidate')

if __name__=='__main__':unittest.main()
