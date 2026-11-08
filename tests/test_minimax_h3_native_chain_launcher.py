"""Execute the minimal chain launcher and inherited fixed workload boundary."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / 'artifacts/infinifabric-build-v1'
BASH = os.environ.get('TEST_BASH') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash required')
class ChainLauncherTests(unittest.TestCase):
    def test_runner_selection_fixed_arguments_and_error_propagation(self):
        launcher = FILES / 'diagnose_minimax_h3_native_chain.sh'
        self.assertTrue(launcher.is_file(), 'chain launcher missing')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ('diagnose_minimax_h3_native_chain.sh', 'run_minimax_h3_fl2va_native_quality_probe.sh'):
                shutil.copyfile(FILES / name, tmp / name)
            # Stub only the full GPU execution boundary, retain launcher source selection.
            (tmp / 'run_minimax_h3_fl2va_native_quality_probe.sh').write_text(
                'main() { printf "helper=%s\\nlabel=%s\\n" "$NATIVE_PROBE_HELPER_NAME" "$NATIVE_PROBE_LABEL"; return 39; }\n')
            result = subprocess.run([BASH, '-c', 'export PATH="/usr/bin:/bin:$PATH"; bash "$1"',
                                     'test', (tmp / launcher.name).as_posix()], capture_output=True, text=True)
            self.assertEqual(result.returncode, 39, result.stderr)
            self.assertIn('helper=diagnose_minimax_h3_native_chain.py', result.stdout)
            self.assertIn('label=native-chain-trace', result.stdout)

    def test_extra_program_arguments_rejected(self):
        launcher = FILES / 'diagnose_minimax_h3_native_chain.sh'
        self.assertTrue(launcher.is_file(), 'chain launcher missing')
        result = subprocess.run([BASH, '-c', 'export PATH="/usr/bin:/bin:$PATH"; bash "$1" --steps=1',
                                 'test', launcher.as_posix()], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('accepts no program arguments', result.stderr)


if __name__ == '__main__':
    unittest.main()
