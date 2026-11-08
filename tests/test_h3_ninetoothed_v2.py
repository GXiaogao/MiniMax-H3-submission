"""Host tests for offline integrity and fail-closed stage reporting, not DCU tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'scripts/h3-ninetoothed-v2'


def module(name):
    path = PACKAGE / (name + '.py')
    if not path.is_file():
        raise AssertionError(f'Implementation missing: {path.name}')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class IntegrityTests(unittest.TestCase):
    def test_wheel_name_metadata_and_digest_are_verified(self):
        common = module('common')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wheel = root / 'ninetoothed-0.26.0-py3-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('ninetoothed-0.26.0.dist-info/METADATA',
                                 'Name: ninetoothed\nVersion: 0.26.0\n')
            record = {'name': 'ninetoothed', 'version': '0.26.0',
                      'file': wheel.name, 'sha256': common.sha256(wheel)}
            common.check_wheel(root, record)
            for changes in ({'file': '../x.whl'}, {'file': r'..\x.whl'},
                            {'sha256': '0' * 64}, {'version': '0.27.0'}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    common.check_wheel(root, {**record, **changes})

    def test_manifest_requires_exact_set_and_published_hash(self):
        common = module('common')
        items = [{'name': name, 'version': version, 'file': f'{name}.whl',
                  'sha256': '1' * 64, 'publisher_sha256': '1' * 64}
                 for name, version in common.EXPECTED.items()]
        good = {'schema': 2, 'candidate': common.CANDIDATE, 'wheels': items}
        self.assertEqual(len(common.validate_manifest(good)), 5)
        for entries in (items[:-1], items + [items[0]],
                        [{**items[0], 'publisher_sha256': '2' * 64}] + items[1:]):
            with self.assertRaises(ValueError):
                common.validate_manifest({**good, 'wheels': entries})


class ReportTests(unittest.TestCase):
    def test_stage_needs_matching_identity_count_and_clean_exit(self):
        common = module('common')
        good = {'stage': 'jit-add', 'run_id': 'r1', 'passed': True,
                'cases': [{'passed': True}] * 9}
        self.assertTrue(common.stage_passed(0, good, 'jit-add', 'r1', 9))
        for rc, report in ((139, good), (0, None),
                           (0, {**good, 'run_id': 'old'}),
                           (0, {**good, 'cases': []}),
                           (0, {**good, 'cases': [None] * 9}),
                           (0, {**good, 'passed': False}),
                           (0, {**good, 'cases': [{'passed': False}] * 9})):
            self.assertFalse(common.stage_passed(rc, report, 'jit-add', 'r1', 9))

    def test_child_timeout_and_output_preserved(self):
        common = module('common')
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'child.log'
            with log.open('w') as stream:
                rc = common.run_child([sys.executable, '-c',
                                       'import time; print("started", flush=True); time.sleep(30)'],
                                      stream, .5)
            self.assertEqual(rc, 124)
            self.assertIn('started', log.read_text())

    def test_generated_launch_option_only_not_compute_source_is_changed(self):
        common = module('common')
        source = 'def compute(a):\n    return a * a + a\ndef launch(a):\n    return compute[(1,)](a, num_warps=4)\n'
        result = common.disable_fusion_source(source, 'compute')
        import ast
        old = ast.parse(source)
        new = ast.parse(result)
        self.assertEqual(ast.dump(old.body[0]), ast.dump(new.body[0]))
        call = new.body[1].body[0].value
        self.assertIn(('enable_fp_fusion', False), [(x.arg, x.value.value) for x in call.keywords])
        with self.assertRaises(ValueError):
            common.disable_fusion_source('def launch():\n    return 1\n', 'compute')

    def test_cpu_machine_exits_nonzero_and_preserves_report(self):
        script = PACKAGE / 'verify.py'
        self.assertTrue(script.is_file(), 'Verifier is not implemented')
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(script), '--logs', folder],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            reports = list(Path(folder).glob('h3_ninetoothed_v2_*/summary.json'))
            self.assertEqual(len(reports), 1, result.stdout + result.stderr)
            report = json.loads(reports[0].read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['infinicore_integrated'])
            self.assertEqual(report['stages'][0]['status'], 'FAIL')
            self.assertTrue(all(x['status'] == 'NOT_RUN' for x in report['stages'][1:]))


if __name__ == '__main__':
    unittest.main()
