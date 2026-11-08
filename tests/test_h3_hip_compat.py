"""Replay real server ABI evidence and check reversible, scoped source repair."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'logs/2026-10-26-hip-abi-server/evidence'
SCRIPT = ROOT / 'scripts/repair_h3_triton_hip.py'


def module():
    if not SCRIPT.is_file():
        raise AssertionError('HIP compatibility repair is not implemented')
    spec = importlib.util.spec_from_file_location('repair', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class RepairTests(unittest.TestCase):
    def test_postcheck_works_with_torch29_hip_properties_without_sharedmem_attribute(self):
        repair = module()
        report = json.loads((EVIDENCE / 'report.json').read_text())
        props = types.SimpleNamespace(gcnArchName=report['torch']['arch'], warp_size=64,
                                      multi_processor_count=64)
        actual = {'arch': props.gcnArchName, 'warpSize': 64, 'multiprocessor_count': 64,
                  'max_shared_mem': 65536}
        target = types.SimpleNamespace(backend='hip', arch='gfx906', warp_size=64)
        torch = types.SimpleNamespace(__version__='2.9.0', __file__=report['torch']['path'],
            version=types.SimpleNamespace(hip='6.3.26093'),
            cuda=types.SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: props))
        triton = types.SimpleNamespace(runtime=types.SimpleNamespace(driver=types.SimpleNamespace(
            active=types.SimpleNamespace(get_current_target=lambda: target,
            utils=types.SimpleNamespace(get_device_properties=lambda _: actual)))))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / 'driver.c'; native.write_bytes(b'test source')
            before = root / 'before.json'; before.write_text(json.dumps(report))
            after = root / 'after.json'
            driver = types.SimpleNamespace(__file__=str(root / 'driver.py'),
                _get_path_to_hip_runtime_dylib=lambda: report['selected_hip_library']['path'])
            modules = {'torch': torch, 'triton': triton,
                       'triton.backends.amd': types.SimpleNamespace(driver=driver)}
            original_read = Path.read_text
            def read(path, *args, **kwargs):
                return '' if path.as_posix() == '/proc/self/maps' else original_read(path, *args, **kwargs)
            with patch.dict('sys.modules', modules), patch('sys.argv',
                    ['postcheck.py', str(before), repair.file_hash(native), str(after)]), \
                    patch.object(Path, 'read_text', read):
                exec(repair.POSTCHECK, {})
            self.assertTrue(json.loads(after.read_text())['passed'])

    def test_accepts_real_evidence_and_rejects_ambiguous_or_changed_runtime(self):
        repair = module()
        report = json.loads((EVIDENCE / 'report.json').read_text())
        repair.validate_probe(report)
        for key in ('version', 'layout', 'query', 'library', 'target', 'failed'):
            bad = copy.deepcopy(report)
            if key == 'version': bad['versions']['hipDriverGetVersion']['value'] += 1
            if key == 'layout': bad['queries']['hipGetProcAddress:hipGetDeviceProperties']['layout_evidence'] = 'ambiguous'
            if key == 'query': bad['get_proc_address']['rc'] = 1
            if key == 'library': bad['selected_hip_library']['realpath'] = '/tmp/other.so'
            if key == 'target': bad['torch']['arch'] = 'gfx1100'
            if key == 'failed': bad['collection_complete'] = False
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                repair.validate_probe(bad)

    def test_patch_is_exact_and_preserves_compute_and_launch(self):
        repair = module()
        original = (EVIDENCE / 'source-evidence/driver.c').read_bytes()
        patched = repair.patched_source(original)
        self.assertIn(b'#include <hip/hip_deprecated.h>', patched)
        self.assertIn(b'hipDeviceProp_tR0000 *prop', patched)
        self.assertIn(b'hipDeviceProp_tR0000 props = storage.legacy;', patched)
        self.assertEqual(original.split(b'static PyObject *loadBinary', 1)[1],
                         patched.split(b'static PyObject *loadBinary', 1)[1])
        with self.assertRaises(RuntimeError):
            repair.patched_source(original + b'\n')

    def test_failed_validation_rolls_back_and_retains_backup(self):
        repair = module()
        original = (EVIDENCE / 'source-evidence/driver.c').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = root / 'venv/driver.c'; driver.parent.mkdir()
            driver.write_bytes(original)
            logs = root / 'logs'; logs.mkdir()
            def fail(): raise RuntimeError('post-check failed')
            with self.assertRaisesRegex(RuntimeError, 'post-check failed'):
                repair.transaction(driver, original, logs, fail)
            self.assertEqual(driver.read_bytes(), original)
            self.assertEqual((logs / 'driver.c.before').read_bytes(), original)
            self.assertTrue((logs / 'driver.patch').is_file())

    def test_write_failure_preserves_source_and_reports_original_error(self):
        repair = module()
        original = (EVIDENCE / 'source-evidence/driver.c').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = root / 'driver.c'; driver.write_bytes(original)
            with patch.object(repair, 'write_atomic', side_effect=PermissionError('permission denied')):
                with self.assertRaisesRegex(PermissionError, 'permission denied'):
                    repair.transaction(driver, original, root, lambda: {'passed': True})
            self.assertEqual(driver.read_bytes(), original)

    def test_second_run_does_not_overwrite_original_backup(self):
        repair = module()
        original = (EVIDENCE / 'source-evidence/driver.c').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = root / 'driver.c'; driver.write_bytes(original)
            first = root / 'first'; first.mkdir()
            second = root / 'second'; second.mkdir()
            repair.transaction(driver, original, first, lambda: {'passed': True})
            repair.transaction(driver, original, second, lambda: {'passed': True})
            self.assertEqual((first / 'driver.c.before').read_bytes(), original)
            self.assertEqual(driver.read_bytes(), repair.patched_source(original))
            self.assertEqual((second / 'driver.c.original').read_bytes(), original)

    def test_unknown_source_and_target_outside_candidate_are_rejected(self):
        repair = module()
        original = (EVIDENCE / 'source-evidence/driver.c').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = root / 'driver.c'; driver.write_bytes(b'not the expected source')
            with self.assertRaises(RuntimeError):
                repair.transaction(driver, original, root, lambda: None)
            self.assertEqual(driver.read_bytes(), b'not the expected source')
            with self.assertRaises(RuntimeError):
                repair.require_inside(driver, root / 'candidate')


if __name__ == '__main__':
    unittest.main()
