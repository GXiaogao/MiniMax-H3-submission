import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class CleanBuildTests(unittest.TestCase):
    def test_builder_keeps_old_markers_and_uses_new_candidate(self):
        spec=importlib.util.spec_from_file_location('clean',ROOT/'artifacts/infinifabric-build-v1/prepare_load_sync_clean_rebuild.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        text=mod.builder_text((ROOT/'artifacts/infinifabric-build-v1/build_infinilm_extension.sh').read_bytes())
        self.assertIn('/diagnostics/h3-load-sync-rebuild/',text)
        self.assertIn('readonly SUCCESS_MARKER="${RUN_DIR}/infinilm-load-sync-rebuild.success"',text)
        self.assertNotIn('readonly SUCCESS_MARKER="${STATE_DIR}/infinilm-load-sync.success"',text)
        self.assertIn('source-hash-verification.json',text)
        self.assertLess(text.index('run_stage "verify-reconstructed-source"'),text.index('run_stage "xmake-config"'))

if __name__=='__main__':unittest.main()
