"""Exercise restore decisions without installing into the developer's Python."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("TEST_BASH") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required")
class RestoreFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        for directory in ("wheels", "packages", "scripts", "logs", "bin", "Projects/InfiniCore", "Projects/InfiniLM"):
            (self.home / directory).mkdir(parents=True)
        for project in ("InfiniCore", "InfiniLM"):
            (self.home / "Projects" / project / "xmake.lua").touch()
        shutil.copy(ROOT / "scripts/infinilm_runtime_requirements.txt", self.home / "scripts")
        probe = ROOT / "scripts/runtime_probe.py"
        if probe.exists():
            shutil.copy(probe, self.home / "scripts")
        shutil.copy(ROOT / "scripts/restore_runtime.sh", self.home / "scripts")
        self.write_tool("python", r'''#!/usr/bin/env bash
case "$1" in
    *runtime_probe.py)
        case "$2" in
            dependencies) [[ "${CASE:-}" != missing_dependency && "${CASE:-}" != install_failure || -f "$WORK_HOME/installed" ]] || exit 10 ;;
            dependency-imports) [[ "${CASE:-}" != broken_import ]] || { echo 'ImportError: test ABI mismatch'; exit 1; } ;;
            registration) [[ "${CASE:-}:$3" != missing_registration:infinicore || -f "$WORK_HOME/installed" ]] || exit 10 ;;
        esac
        exit 0 ;;
    -m)
        printf 'python %s\n' "$*" >> "$WORK_HOME/mutations"
        if [[ "${CASE:-}" == install_failure ]]; then echo 'offline wheel unavailable'; exit 37; fi
        touch "$WORK_HOME/installed"
        exit 0 ;;
    -) cat >/dev/null; exit 0 ;;
    -c) exit 0 ;;
esac
echo "unexpected Python invocation: $*" >&2
exit 99
''')
        self.write_tool("xmake", r'''#!/usr/bin/env bash
case "$*" in
    g) printf 'network = private\npkg_searchdirs = %s/packages\n' "$WORK_HOME"; exit 0 ;;
    'repo --list') printf '%s/.xmake/repositories/xmake-repo\n' "$WORK_HOME"; exit 0 ;;
esac
printf 'xmake %s\n' "$*" >> "$WORK_HOME/mutations"
mkdir -p "$INFINI_ROOT/include/infinicore"
touch "$INFINI_ROOT/include/infinicore/tensor.hpp"
''')

    def write_tool(self, name, text):
        path = self.home / "bin" / name
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def header(self):
        path = self.home / "infini/include/infinicore/tensor.hpp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def run_restore(self, case="", setup=False):
        env = dict(os.environ, WORK_HOME=self.home.as_posix(),
                   INFINI_ROOT=(self.home / "infini").as_posix(),
                   LOG_DIR=(self.home / "logs").as_posix(), CASE=case,
                   DTK_ENV_FILE=(self.home / "dtk_env.sh").as_posix(), MINIMAX_AUTO_RESTORE="1")
        command = '''if command -v cygpath >/dev/null; then
            WORK_HOME="$(cygpath -u "$WORK_HOME")"
            INFINI_ROOT="$(cygpath -u "$INFINI_ROOT")"
            LOG_DIR="$(cygpath -u "$LOG_DIR")"
        fi
        export PATH="$WORK_HOME/bin:$PATH"
        '''
        command += 'source "$1" && source "$1"' if setup else 'bash "$1"'
        script = "setup_env.sh" if setup else "restore_runtime.sh"
        return subprocess.run([BASH, "-c", command, "test", (ROOT / "scripts" / script).as_posix()],
                              env=env, text=True, capture_output=True)

    def mutations(self):
        path = self.home / "mutations"
        return path.read_text().splitlines() if path.exists() else []

    def test_missing_native_install_does_not_reinstall_existing_python_package(self):
        result = self.run_restore()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mutations(), ["xmake install"])

    def test_ready_runtime_has_no_install_side_effects_on_repeated_entry(self):
        self.header()
        for _ in range(2):
            result = self.run_restore()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mutations(), [])

    def test_broken_import_reports_error_without_blind_reinstallation(self):
        self.header()
        result = self.run_restore("broken_import")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ABI mismatch", result.stdout + result.stderr)
        self.assertEqual(self.mutations(), [])

    def test_missing_registration_only_runs_offline_editable_install(self):
        self.header()
        result = self.run_restore("missing_registration")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        changes = self.mutations()
        self.assertEqual(len(changes), 1, changes)
        self.assertIn("--no-index", changes[0])
        self.assertIn("--no-deps", changes[0])
        self.assertNotIn("xmake", changes[0])

    def test_dependency_repair_uses_constraints_and_is_not_repeated(self):
        self.header()
        for _ in range(2):
            result = self.run_restore("missing_dependency")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        changes = self.mutations()
        self.assertEqual(len(changes), 1, changes)
        self.assertIn("--no-index", changes[0])
        self.assertIn(" -c ", changes[0])
        self.assertNotIn("--force-reinstall", changes[0])

    def test_install_failure_stops_and_preserves_error_details(self):
        self.header()
        result = self.run_restore("install_failure")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertIn("offline wheel unavailable", result.stdout + result.stderr)
        self.assertNotIn("Runtime is ready", result.stdout)
        self.assertEqual(len(self.mutations()), 1)
        self.assertIn("offline wheel unavailable", (self.home / "logs/setup_runtime_details.log").read_text())

    def test_setup_reentry_does_not_write_xmake_settings_or_reinstall(self):
        self.header()
        (self.home / "dtk_env.sh").write_text("true\n")
        for relative in ("packages/b/boost/xmake.lua", "packages/p/pybind11/xmake.lua"):
            path = self.home / ".xmake/repositories/xmake-repo" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        result = self.run_restore(setup=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.mutations(), [])
        self.assertIn("exit code: 0", (self.home / "logs/setup_runtime_restore.log").read_text())


if __name__ == "__main__":
    unittest.main()
