import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'artifacts/infinifabric-build-v1/diagnose_minimax_h3_vae_roundtrip.sh'
BASH = os.environ.get('TEST_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required')
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LAUNCHER.is_file())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = os.environ.copy()
        for key in ('IFAB_AFS_HOME', 'WORK_HOME', 'LOG_DIR', 'PYTHON'):
            self.env.pop(key, None)

    def run_bash(self, script, *args):
        return subprocess.run([BASH, '-c', 'export PATH="/usr/bin:/bin:$PATH"\n' + script,
                               'test', *map(str, args)], env=self.env, cwd=self.directory,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)

    def test_rejects_page_overrides_before_writes(self):
        result = self.run_bash('bash "$1" --steps 1', LAUNCHER.as_posix())
        self.assertEqual(result.returncode, 2)
        self.assertIn('accepts no program arguments', result.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_afs_is_explicit(self):
        result = self.run_bash('bash "$1"', LAUNCHER.as_posix())
        self.assertEqual(result.returncode, 1)
        self.assertIn('IFAB_AFS_HOME', result.stderr)

    def test_python_failure_retains_stderr_and_exit_status(self):
        project = self.directory / 'project'
        project.mkdir()
        shutil.copyfile(LAUNCHER, project / LAUNCHER.name)
        for name in ('diagnose_minimax_h3_vae_roundtrip.py', 'run_minimax_h3_fl2va_native_smoke.py',
                     'install_infinilm_private_runtime.py', 'prepare_minimax_h3_hygon_runtime.py'):
            (project / name).touch()
        work = self.directory / 'afs/minimax-h3'
        work.mkdir(parents=True)
        (work / 'setup_infinifabric_env.sh').write_text('# test fixture\n', encoding='utf-8')
        runner = project / 'python-stub.sh'
        runner.write_text('#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then printf "/torch/lib\\n"; exit 0; fi\nprintf "injected model load failure\\n" >&2\nexit 47\n',
                          encoding='utf-8', newline='\n')
        runner.chmod(0o755)
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(), PYTHON=runner.as_posix())
        result = self.run_bash('''if command -v cygpath >/dev/null; then
    export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"
fi
bash "$1"''', (project / LAUNCHER.name).as_posix())
        self.assertEqual(result.returncode, 47, result.stdout + result.stderr)
        logs = list((work / 'logs').rglob('run.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('injected model load failure', logs[0].read_text(encoding='utf-8'))
        self.assertIn('exit_code=47', (logs[0].parent / 'summary.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
