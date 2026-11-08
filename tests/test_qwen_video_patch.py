"""Exercise the real video-boundary patch in disposable Git repositories."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = ['csrc/models/qwen3_5/qwen3_5_vision.cpp',
         'csrc/models/qwen3_5/qwen3_5_vision.hpp']


class VideoPatchTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('video_patch', ROOT/'scripts/apply_qwen_video_patch.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root/'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Local test')
        for name in FILES:
            target = self.repo/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/'repos/InfiniLM'/name).read_bytes().replace(b'\r\n', b'\n'))
        self.git('add', '.')
        self.git('commit', '-qm', 'baseline')
        self.package = self.root/'package'
        shutil.copytree(ROOT/'artifacts/qwen3-vl-video-boundary-v1', self.package)
        manifest = json.loads((self.package/'qwen_video_patch_manifest.json').read_text())
        # Only the temporary fixture commit differs, target fingerprints stay exact.
        manifest['baseline_commit'] = self.git('rev-parse', 'HEAD').strip()
        (self.package/'qwen_video_patch_manifest.json').write_text(json.dumps(manifest))

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.filemode=false',
                                        '-c', 'core.excludesFile=', '-C', str(self.repo), *args],
                                       text=True, stderr=subprocess.STDOUT)

    def apply(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.module.apply(self.repo, self.package, self.root/'logs')

    def test_real_patch_applies_once_and_preserves_unrelated_files(self):
        (self.repo/'notes.txt').write_text('user work')
        self.assertEqual(self.apply(), 'applied')
        self.assertEqual(self.apply(), 'already-applied')
        self.assertEqual((self.repo/'notes.txt').read_text(), 'user work')
        for name in FILES:
            expected = (ROOT/'worktrees/InfiniLM-qwen-video'/name).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual((self.repo/name).read_bytes(), expected)

    def test_changed_target_is_preserved(self):
        path = self.repo/FILES[0]
        path.write_text('user changes\n')
        with self.assertRaisesRegex(RuntimeError, 'modified'):
            self.apply()
        self.assertEqual(path.read_text(), 'user changes\n')

    def test_partial_patch_is_rejected(self):
        (self.repo/FILES[0]).write_bytes((ROOT/'worktrees/InfiniLM-qwen-video'/FILES[0]).read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'partial'):
            self.apply()

    def test_wrong_commit_is_rejected(self):
        self.git('commit', '--allow-empty', '-qm', 'other version')
        with self.assertRaisesRegex(RuntimeError, 'commit mismatch'):
            self.apply()

    def test_corrupt_patch_is_rejected(self):
        with (self.package/'qwen_video_boundaries.patch').open('a') as handle:
            handle.write('corrupt\n')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            self.apply()


if __name__ == '__main__':
    unittest.main()
