"""Execute the Bash entry and its real fixed-argument runner boundary."""
import os
import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/infinifabric-build-v1"
LAUNCHER = ARTIFACTS / "run_minimax_h3_fl2va_native_quality_probe.sh"
BASH = os.environ.get("TEST_BASH") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required")
class QualityProbeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LAUNCHER.is_file(), "quality-probe launcher is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        for key in ("IFAB_AFS_HOME", "WORK_HOME", "LOG_DIR", "PYTHON"):
            self.env.pop(key, None)

    def bash(self, command, *args):
        command = 'export PATH="/usr/bin:/bin:$PATH"\n' + command
        return subprocess.run([BASH, "-c", command, "test", *map(str, args)],
                              env=self.env, cwd=self.root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)

    def test_page_arguments_are_rejected_before_any_file_is_created(self):
        result = self.bash('bash "$1" --steps=1', LAUNCHER.as_posix())
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("accepts no program arguments", result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_afs_stops_without_creating_work_directories(self):
        result = self.bash('bash "$1"', LAUNCHER.as_posix())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IFAB_AFS_HOME", result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_real_runner_boundary_uses_patched_entry_and_fixed_workload(self):
        capture = self.root / "arguments.bin"
        runner = self.root / "python-stub.sh"
        runner.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE"\nexit "${RUNNER_RC:-0}"\n',
                          encoding="utf-8", newline="\n")
        runner.chmod(0o755)
        self.env["CAPTURE"] = capture.as_posix()
        command = '''source "$1"
PYTHON="$2"
HELPER="/helpers/run_minimax_h3_fl2va_native_smoke.py"
RUNTIME_HELPER="/helpers/install_infinilm_private_runtime.py"
HYGON_RUNTIME_HELPER="/helpers/prepare_minimax_h3_hygon_runtime.py"
STATE_MANIFEST="/state/private.json"
HYGON_RUNTIME_MANIFEST="/state/hygon.json"
LOCAL_TARGET="/job with spaces/site-packages"
RUN_DIR="/logs/probe"
PATCHED_EXAMPLE="/job with spaces/example/minimax_h3.py"
MODEL_PATH="/public/home/aclsylqidf/shared/MiniMax-H3"
FIRST_FRAME="/afs/inputs/fl2va/Golde33443.jpg"
OUTPUT_PATH="/afs/outputs/probe.mp4"
invoke_native_helper
'''
        result = self.bash(command, LAUNCHER.as_posix(), runner.as_posix())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = capture.read_bytes().decode().rstrip("\0").split("\0")
        self.assertEqual(actual, [
            "-u", "/helpers/run_minimax_h3_fl2va_native_smoke.py",
            "--runtime-helper", "/helpers/install_infinilm_private_runtime.py",
            "--hygon-runtime-helper", "/helpers/prepare_minimax_h3_hygon_runtime.py",
            "--hygon-runtime-manifest", "/state/hygon.json",
            "--state-manifest", "/state/private.json",
            "--local-target", "/job with spaces/site-packages", "--run-dir", "/logs/probe",
            "--example", "/job with spaces/example/minimax_h3.py",
            "--model-path", "/public/home/aclsylqidf/shared/MiniMax-H3",
            "--first-frame", "/afs/inputs/fl2va/Golde33443.jpg",
            "--output", "/afs/outputs/probe.mp4", "--run-profile=formal",
            "--text=A dog falls asleep", "--gpus=0,1,2,3,4,5,6,7",
            "--num-frames=124", "--steps=10", "--seed=42", "--height=384", "--width=576",
        ])
        self.env["RUNNER_RC"] = "37"
        failure = self.bash(command, LAUNCHER.as_posix(), runner.as_posix())
        self.assertEqual(failure.returncode, 37, failure.stdout + failure.stderr)

    def test_missing_required_helper_retains_failure_summary(self):
        copied = self.root / "project" / LAUNCHER.name
        copied.parent.mkdir()
        shutil.copyfile(LAUNCHER, copied)
        work = self.root / "afs" / "minimax-h3"
        work.mkdir(parents=True)
        (work / "setup_infinifabric_env.sh").write_text("# fixture environment\n", encoding="utf-8")
        self.env["IFAB_AFS_HOME"] = work.parent.as_posix()
        self.env["WORK_HOME"] = work.as_posix()
        result = self.bash('''if command -v cygpath >/dev/null; then
    export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"
    export WORK_HOME="$(cygpath -u "$WORK_HOME")"
fi
bash "$1"''', copied.as_posix())
        self.assertNotEqual(result.returncode, 0)
        summaries = list((work / "logs").rglob("summary.txt"))
        self.assertEqual(len(summaries), 1, result.stdout + result.stderr)
        summary = summaries[0].read_text(encoding="utf-8")
        self.assertIn("exit_code=1", summary)
        self.assertIn("last_stage=preflight", summary)
        self.assertIn("quality_status=UNASSESSED", summary)
        self.assertIn("required file is missing", result.stdout + result.stderr)

    def test_torch_preflight_retains_traceback_and_failure_status(self):
        runner = self.root / "python-stub.sh"
        runner.write_text('#!/usr/bin/env bash\nprintf "ImportError: shared library unavailable\\n" >&2\nexit 41\n',
                          encoding="utf-8", newline="\n")
        runner.chmod(0o755)
        result = self.bash('''source "$1"
PYTHON="$2"
RUN_DIR="$PWD"
RUN_LOG="$RUN_DIR/run.log"
probe_torch_library
''', LAUNCHER.as_posix(), runner.as_posix())
        self.assertEqual(result.returncode, 41, result.stdout + result.stderr)
        for filename in ("run.log", "torch-preflight.log"):
            self.assertIn("ImportError: shared library unavailable",
                          (self.root / filename).read_text(encoding="utf-8"))

    def run_until_patch_boundary(self, source, failing_checksum=False):
        project = self.root / "project"
        project.mkdir()
        shutil.copyfile(LAUNCHER, project / LAUNCHER.name)
        shutil.copyfile(ARTIFACTS / "prepare_minimax_h3_vae_dtype_probe.py",
                        project / "prepare_minimax_h3_vae_dtype_probe.py")
        for name in ("run_minimax_h3_fl2va_native_smoke.py",
                     "install_infinilm_private_runtime.py", "prepare_minimax_h3_hygon_runtime.py"):
            (project / name).touch()
        work = self.root / "afs/minimax-h3"
        for directory in ("state", "Projects/InfiniLM/examples", "inputs/fl2va"):
            (work / directory).mkdir(parents=True)
        (work / "setup_infinifabric_env.sh").write_text("# fixture\n", encoding="utf-8")
        example = work / "Projects/InfiniLM/examples/minimax_h3.py"
        example.write_text(source, encoding="utf-8", newline="\n")
        (work / "inputs/fl2va/Golde33443.jpg").touch()
        for name in ("infinilm-runtime-private.json", "minimax-h3-hygon-runtime.json"):
            (work / "state" / name).write_text("{}", encoding="utf-8")
        extension = work / "existing-extension.so"
        extension.write_bytes(b"fixture extension")
        (work / "state/infinilm-extension.success").write_text(
            "build_version=1.5-hygon-fl2va-no-cpu-runtime\nextension=" + extension.as_posix() + "\n",
            encoding="utf-8", newline="\n")
        self.env.update(IFAB_AFS_HOME=work.parent.as_posix(), WORK_HOME=work.as_posix(),
                        PYTHON=Path(sys.executable).as_posix())
        checksum_failure = '''sha256sum() { printf 'checksum read failure\n' >&2; return 73; }
export -f sha256sum
''' if failing_checksum else ""
        result = self.bash(checksum_failure + '''if command -v cygpath >/dev/null; then
    export IFAB_AFS_HOME="$(cygpath -u "$IFAB_AFS_HOME")"
    export WORK_HOME="$(cygpath -u "$WORK_HOME")"
fi
bash "$1"''', (project / LAUNCHER.name).as_posix())
        summaries = list((work / "logs").rglob("summary.txt"))
        self.assertEqual(len(summaries), 1, result.stdout + result.stderr)
        self.assertEqual(example.read_text(encoding="utf-8"), source)
        return result, summaries[0].parent

    def test_checksum_failure_stops_before_patching_and_retains_failure_summary(self):
        source = ("pipe.vae.to(device=execution_device, dtype=DTYPE)\n"
                  "pipe.audio_vae.to(device=execution_device, dtype=DTYPE)\n")
        result, evidence = self.run_until_patch_boundary(source, failing_checksum=True)
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        self.assertIn("last_stage=preflight", (evidence / "summary.txt").read_text(encoding="utf-8"))
        self.assertIn("checksum read failure", result.stderr)
        self.assertFalse((evidence / "vae-dtype-patch.json").exists())

    def test_patch_provenance_persists_before_any_gpu_loading(self):
        source = ("def load(pipe, execution_device, DTYPE):\n"
                  "    pipe.vae.to(device=execution_device, dtype=DTYPE)\n"
                  "    pipe.audio_vae.to(device=execution_device, dtype=DTYPE)\n")
        result, evidence = self.run_until_patch_boundary(source)
        self.assertNotEqual(result.returncode, 0)  # No runtime installed in the fixture.
        summary = (evidence / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("last_stage=runtime-preflight", summary)
        report = json.loads((evidence / "vae-dtype-patch.json").read_text(encoding="utf-8"))
        self.assertEqual(report["source_sha256"], hashlib.sha256(source.encode()).hexdigest())
        saved = evidence / "minimax_h3.vae-probe.py"
        self.assertEqual(report["destination_sha256"], hashlib.sha256(saved.read_bytes()).hexdigest())
        self.assertIn('"patch_profile": "preserve-vae-fp32-modules-v1"', result.stdout)
        self.assertIn("infinilm_extension_sha256=", result.stdout)

    def test_source_drift_stops_before_runtime_and_preserves_patch_error(self):
        result, evidence = self.run_until_patch_boundary("# incompatible upstream source\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("last_stage=prepare-vae-dtype-probe",
                      (evidence / "summary.txt").read_text(encoding="utf-8"))
        self.assertIn("exactly once", (evidence / "vae-dtype-patch.log").read_text(encoding="utf-8"))
        self.assertFalse((evidence / "minimax_h3.vae-probe.py").exists())


if __name__ == "__main__":
    unittest.main()
