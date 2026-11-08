import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'


class NineToothedInstallTests(unittest.TestCase):
    def setUp(self):
        f = FOLDER / 'install_ninetoothed_gfx936.py'
        self.assertTrue(f.exists(), 'NineToothed installer missing')
        s = importlib.util.spec_from_file_location('nt_install_test', f)
        self.m = importlib.util.module_from_spec(s)
        s.loader.exec_module(self.m)

    def test_all_four_existing_local_wheels_verified(self):
        wheel_dir = Path('E:/projects/minimax-h3-match/相关代码文件/wheel')
        if not wheel_dir.is_dir():self.skipTest('local wheel evidence unavailable')
        wheels = self.m.verify_wheels(wheel_dir)
        self.assertEqual(len(wheels), 4)
        self.assertTrue(all(p.suffix == '.whl' for p in wheels))

    def test_missing_or_corrupt_wheel_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.m.verify_wheels(Path(d))
            filename = next(iter(self.m.WHEELS))
            (Path(d) / filename).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):self.m.verify_wheels(Path(d))

    def test_cache_patch_only_changes_isolated_location_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'generation.py'
            source = 'import pathlib\nCACHE_DIR = pathlib.Path.home() / ".ninetoothed"\nCACHE_DIR.mkdir(exist_ok=True)\n'
            f.write_text(source)
            evidence = self.m.patch_cache(f)
            self.assertNotIn('Path.home()', f.read_text())
            self.assertIn('NINETOOTHED_CACHE_DIR', f.read_text())
            self.assertEqual(evidence['patched_sha256'], self.m.sha(f))
            self.assertEqual(f.with_suffix('.py.before-cache-patch').read_text(), source)
            with self.assertRaises(ValueError):self.m.patch_cache(f)

    def test_candidate_overwrite_refused_before_unpack(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'candidate'
            root.mkdir()
            (root / 'keep').write_text('original')
            with self.assertRaises(FileExistsError):self.m.install_candidate([], root)
            self.assertEqual((root / 'keep').read_text(), 'original')

    def test_worker_environment_isolated_without_redefining_home(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = self.m.worker_environment(root/'candidate', root/'vendor', root/'cache',
                {'HOME': '/unchanged', 'PYTHONPATH': '/untrusted', 'PYTHONHOME': '/other', 'HSA_OVERRIDE_GFX_VERSION': '9.0.0'})
            self.assertEqual(env['HOME'], '/unchanged')
            self.assertNotIn('PYTHONHOME', env)
            self.assertNotIn('/untrusted', env['PYTHONPATH'])
            self.assertEqual(env['HSA_OVERRIDE_GFX_VERSION'], '9.0.0')
            self.assertEqual(env['PYTHONNOUSERSITE'], '1')
            self.assertTrue(Path(env['NINETOOTHED_CACHE_DIR']).is_dir())

    def test_failed_preflight_does_not_publish_success_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rc = self.m.main(['--work-home', d, '--run-dir', d+'/logs', '--local-root', d+'/local', '--candidate', d+'/candidate'])
            self.assertEqual(rc, 1)
            report = json.loads((root/'logs/ninetoothed-install.json').read_text())
            self.assertEqual(report['result'], 'FAILED')
            self.assertFalse((root/'state/ninetoothed-gfx936.json').exists())

    def test_jit_report_requires_exact_nine_cases_and_matching_baseline(self):
        baseline = {'torch_version': 'unchanged'}
        cases = [{'name': f'{dtype}-n{n}-stride{stride}', 'passed': True}
                 for dtype in ('float32', 'float16', 'bfloat16') for n, stride in ((1, 1), (513, 1), (513, 2))]
        valid = {'result': 'JIT_VERIFIED', 'arch': 'gfx936', 'baseline': baseline, 'cases': cases,
                 'isolated_numpy': {'version': '1.26.4'}}
        self.m.validate_probe(valid, baseline)
        for bad in ({**valid, 'cases': cases[:-1]}, {**valid, 'arch': 'gfx906'},
                    {**valid, 'cases': [{'name': str(i), 'passed': True} for i in range(9)]},
                    {**valid, 'isolated_numpy': {'version': '1.24.0'}}):
            with self.assertRaises(ValueError):self.m.validate_probe(bad, baseline)

    def test_numpy_minimum_includes_patch_version(self):
        path = FOLDER / 'ninetoothed_gfx936_probe.py'
        s = importlib.util.spec_from_file_location('nt_probe_test', path)
        probe = importlib.util.module_from_spec(s);s.loader.exec_module(probe)
        self.assertFalse(probe.numpy_compatible('1.26.3'))
        self.assertTrue(probe.numpy_compatible('1.26.4'))
        self.assertTrue(probe.numpy_compatible('2.2.6'))

    def test_private_numpy_can_differ_while_torch_identity_is_preserved(self):
        path = FOLDER / 'ninetoothed_gfx936_probe.py'
        s = importlib.util.spec_from_file_location('nt_numpy_identity_test', path)
        probe = importlib.util.module_from_spec(s);s.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as d:
            candidate = Path(d)
            baseline = {'torch_version': '2.9.0', 'torch_file': '/system/torch.py', 'hip': '6.3',
                        'numpy_version': '1.24.0', 'numpy_file': '/system/numpy.py'}
            actual = {**baseline, 'numpy_version': '1.26.4',
                      'numpy_file': str(candidate/'site-packages/numpy/__init__.py')}
            result = probe.validate_runtime_identity(actual, baseline, candidate)
            self.assertEqual(result['version'], '1.26.4')
            for bad in ({**actual, 'torch_file': '/changed/torch.py'},
                        {**actual, 'numpy_file': '/system/numpy.py'},
                        {**actual, 'numpy_version': '1.26.3'}):
                with self.assertRaises(RuntimeError):probe.validate_runtime_identity(bad, baseline, candidate)

    def test_publication_lifecycle_probe_failure_postcheck_failure_and_success(self):
        baseline = {'torch_version': '2.9.0', 'torch_file': '/system/torch', 'numpy_version': '2.2.6'}
        cases = [{'name': f'{dtype}-n{n}-stride{stride}', 'passed': True}
                 for dtype in ('float32', 'float16', 'bfloat16') for n, stride in ((1, 1), (513, 1), (513, 2))]
        for scenario in ('probe-failure', 'postcheck-failure', 'success'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as d, ExitStack() as stack:
                root = Path(d)
                state = root / 'state/ninetoothed-gfx936.json'
                state.parent.mkdir()
                state.write_text('previous verified state')
                def fake_probe(command, env):
                    if scenario == 'probe-failure':raise RuntimeError('probe failed')
                    self.m.write_json(root / 'logs/ninetoothed-probe.json',
                        {'result': 'JIT_VERIFIED', 'arch': 'gfx936', 'baseline': baseline, 'cases': cases,
                         'isolated_numpy': {'version': '1.26.4'}})
                stack.enter_context(patch.object(self.m.sys, 'platform', 'linux'))
                stack.enter_context(patch.object(self.m.sys, 'version_info', (3, 11)))
                stack.enter_context(patch.object(self.m, 'verify_wheels', return_value=[]))
                stack.enter_context(patch.object(self.m, 'sha', return_value=self.m.TRITON_SHA))
                stack.enter_context(patch.object(self.m, 'install_candidate', return_value={'scope': 'test-cache'}))
                stack.enter_context(patch.object(self.m, 'patch_triton_cache', return_value={'scope': 'test-rename'}))
                stack.enter_context(patch.object(self.m.shutil, 'copyfile'))
                stack.enter_context(patch.object(self.m, 'extractor', return_value=Mock()))
                after = {**baseline, 'torch_file': '/changed/torch'} if scenario == 'postcheck-failure' else baseline
                stack.enter_context(patch.object(self.m.subprocess, 'check_output', side_effect=[json.dumps(baseline), json.dumps(after)]))
                stack.enter_context(patch.object(self.m, 'run_with_heartbeat', side_effect=fake_probe))
                rc = self.m.main(['--work-home', d, '--run-dir', d+'/logs', '--local-root', d+'/local', '--candidate', d+'/candidate'])
                report = json.loads((root/'logs/ninetoothed-install.json').read_text())
                if scenario == 'success':
                    self.assertEqual(rc, 0)
                    self.assertTrue(report['state_published'])
                    self.assertEqual(json.loads(state.read_text())['scope'], 'standalone-add-jit-not-infinicore-integration')
                else:
                    self.assertEqual(rc, 1)
                    self.assertFalse(report['state_published'])
                    self.assertEqual(state.read_text(), 'previous verified state')
