"""Host-only safeguards for the read-only HIP diagnostic, not device tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/diagnose_h3_triton_hip.py'


def module():
    if not SCRIPT.is_file():
        raise AssertionError('Diagnostic is not implemented yet')
    spec = importlib.util.spec_from_file_location('hip_diag', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class DiagnosticTests(unittest.TestCase):
    def test_invalid_architecture_bytes_are_preserved(self):
        diag = module()
        self.assertEqual(diag.arch_value(b'gfx906\x00garbage')['text'], 'gfx906')
        result = diag.arch_value(b'\x93\x00')
        self.assertEqual(result['hex'], '93')
        self.assertIsNone(result['text'])
        self.assertFalse(result['valid_ascii'])
        self.assertFalse(diag.arch_value(b'gfx906')['nul_terminated'])

    def test_layout_mismatch_requires_success_and_independent_device(self):
        diag = module()
        old = {'arch': diag.arch_value(b'gfx906\0'), 'warp': 64, 'multiprocessors': 60}
        new = {'arch': diag.arch_value(b'\x93\0'), 'warp': 0, 'multiprocessors': 0}
        row = {'rc': 0, 'guard_ok': True, 'R0000': old, 'R0600': new}
        self.assertEqual(diag.layout_evidence(row, 'gfx906:sramecc+'), 'R0000_matches_only')
        for changes in ({'rc': 1}, {'guard_ok': False}, {'R0600': old}):
            self.assertNotEqual(diag.layout_evidence({**row, **changes}, 'gfx906'), 'R0000_matches_only')
        self.assertNotEqual(diag.layout_evidence(row, ''), 'R0000_matches_only')

    @unittest.skipIf(sys.platform == 'linux', 'Non-Linux failure path')
    def test_unsupported_host_still_produces_evidence_without_pass(self):
        module()
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(SCRIPT), '--log-dir', folder],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            reports = list(Path(folder).glob('h3_hip_abi_*/report.json'))
            self.assertEqual(len(reports), 1)
            data = json.loads(reports[0].read_text())
            self.assertFalse(data['toolchain_validated'])
            self.assertFalse(data['collection_complete'])
            self.assertIn('Linux', data['error'])
            self.assertEqual(len(list(Path(folder).glob('*.zip'))), 1)


if __name__ == '__main__':
    unittest.main()
