"""Exercise patch deployment with real temporary Git repositories; no GPU claims."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mha_patch', ROOT / 'scripts/apply_h3_mha_patch.py')


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Local test')
        (self.repo / 'operator.cc').write_bytes(b'old\n')
        self.git('add', 'operator.cc')
        self.git('commit', '-qm', 'baseline')
        baseline = self.git('rev-parse', 'HEAD').strip()
        self.package = self.root / 'package'
        self.package.mkdir()
        (self.package / 'h3_mha_infiniop.patch').write_text(
            'diff --git a/operator.cc b/operator.cc\n'
            '--- a/operator.cc\n+++ b/operator.cc\n@@ -1 +1 @@\n-old\n+new\n', encoding='utf-8')
        digest = lambda x: hashlib.sha256(x).hexdigest()
        patch_bytes = (self.package / 'h3_mha_infiniop.patch').read_bytes().replace(b'\r\n', b'\n')
        (self.package / 'h3_mha_patch_manifest.json').write_text(json.dumps({
            'baseline_commit': baseline, 'patch_sha256': digest(patch_bytes),
            'files': {'operator.cc': {'before': digest(b'old\n'), 'after': digest(b'new\n')}}
        }))

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.excludesFile=',
                                        '-C', str(self.repo), *args], text=True, stderr=subprocess.STDOUT)

    def apply(self):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            return module.apply(self.repo, self.package, self.root / 'logs')

    def test_applies_once_and_second_run_is_noop(self):
        self.assertEqual(self.apply(), 'applied')
        self.assertEqual((self.repo / 'operator.cc').read_bytes(), b'new\n')
        self.assertEqual(self.apply(), 'already-applied')

    def test_preserves_unrelated_work(self):
        (self.repo / 'notes.txt').write_text('my work')
        self.apply()
        self.assertEqual((self.repo / 'notes.txt').read_text(), 'my work')

    def test_rejects_modified_target_without_overwriting(self):
        (self.repo / 'operator.cc').write_text('my edit\n')
        with self.assertRaisesRegex(RuntimeError, 'modified|mismatch'):
            self.apply()
        self.assertEqual((self.repo / 'operator.cc').read_text(), 'my edit\n')

    def test_rejects_different_commit(self):
        self.git('commit', '--allow-empty', '-qm', 'other')
        with self.assertRaisesRegex(RuntimeError, 'commit'):
            self.apply()
        self.assertEqual((self.repo / 'operator.cc').read_bytes(), b'old\n')

    def test_rejects_corrupt_patch(self):
        with (self.package / 'h3_mha_infiniop.patch').open('a') as handle:
            handle.write('corruption\n')
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            self.apply()
        self.assertEqual((self.repo / 'operator.cc').read_bytes(), b'old\n')

    def prepare_real_singleton_upgrade(self):
        # Recreate the deployed v1 state: tracked baseline + original patch,
        # including the backend file which is untracked on the server.
        relative = Path('src/infinicore/ops/multi_head_attention/mha.cc')
        (self.repo / relative).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'repos/InfiniCore' / relative, self.repo / relative)
        self.git('add', str(relative))
        self.git('commit', '-qm', 'source baseline')
        self.git('apply', str(ROOT / 'artifacts/h3-mha-infiniop-v1/h3_mha_infiniop.patch'))
        package = ROOT / 'artifacts/h3-mha-singleton-stride-v1'
        shutil.copyfile(package/'h3_mha_infiniop.patch', self.package/'h3_mha_infiniop.patch')
        manifest = json.loads((package/'h3_mha_patch_manifest.json').read_text(encoding='utf-8'))
        # Only the fixture commit differs from the server; use the real patch
        # and file fingerprints without relaxing target-content checks.
        manifest['baseline_commit'] = self.git('rev-parse','HEAD').strip()
        (self.package/'h3_mha_patch_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return relative

    def test_real_singleton_upgrade_handles_untracked_backend_and_reentry(self):
        self.prepare_real_singleton_upgrade()
        self.assertEqual(self.apply(), 'applied')
        relative = 'src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc'
        expected = (ROOT/'worktrees/InfiniCore-h3-mha'/relative).read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual((self.repo/relative).read_bytes(),expected)
        self.assertEqual(self.apply(), 'already-applied')

    def test_real_singleton_upgrade_refuses_changed_dispatcher(self):
        relative = self.prepare_real_singleton_upgrade()
        (self.repo/relative).write_bytes(b'local modification\n')
        with self.assertRaisesRegex(RuntimeError,'modified'):
            self.apply()
        self.assertEqual((self.repo/relative).read_bytes(),b'local modification\n')


if __name__ == '__main__':
    unittest.main()
