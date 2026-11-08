import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-xmake-diagnostic-v1"
    / "diagnose_xmake_offline.sh"
)


class XmakeDiagnosticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT_PATH.is_file():
            raise AssertionError(f"diagnostic script is missing: {SCRIPT_PATH}")
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_portable_lf_only_bash(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("exec > >(", self.text)
        self.assertNotIn("/dev/fd", self.text)

    def test_checks_locked_offline_archives_and_recipes(self):
        self.assertIn("boost-1.92.0-cmake.tar.xz", self.text)
        self.assertIn("pybind11-v3.1.0.zip", self.text)
        self.assertIn(
            "9bed76128d4e46755dbe818487788c6fceb6f72b378f4daa49b7e1e600d9088d",
            self.text,
        )
        self.assertIn(
            "affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536",
            self.text,
        )
        self.assertIn("packages/b/boost/xmake.lua", self.text)
        self.assertIn("packages/p/pybind11/xmake.lua", self.text)

    def test_reasserts_private_xmake_configuration(self):
        self.assertIn("xmake g", self.text)
        self.assertIn("--network=private", self.text)
        self.assertIn('--pkg_searchdirs="${PKG_DIR}"', self.text)

    def test_environment_can_assign_work_home_before_it_is_frozen(self):
        source_pos = self.text.index('source "${ENV_FILE}"')
        prefix = self.text[:source_pos]
        self.assertNotIn("readonly WORK_HOME=", prefix)
        self.assertIn('BOOTSTRAP_WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"', prefix)
        self.assertIn(
            '[[ "${WORK_HOME:-}" != "${BOOTSTRAP_WORK_HOME}" ]]',
            self.text[source_pos:],
        )

    def test_uses_job_workspace_for_xmake_temporary_files(self):
        self.assertNotIn(
            'XMAKE_DIAG_TMP="/dev/shm/',
            self.text,
        )
        self.assertIn(
            'XMAKE_DIAG_TMP="${PWD}/.xmake-diagnostic-${UID}-$$"',
            self.text,
        )

    def test_localizes_mutable_xmake_state_and_project_copy(self):
        self.assertIn(
            'PERSISTENT_XMAKE_STORAGE="${XMAKE_STORAGE_DIR}"',
            self.text,
        )
        self.assertIn(
            'export XMAKE_GLOBALDIR="${XMAKE_DIAG_TMP}/global"',
            self.text,
        )
        self.assertIn(
            'export XMAKE_STORAGE_DIR="${XMAKE_GLOBALDIR}/.xmake"',
            self.text,
        )
        self.assertIn(
            'cp -a "${PERSISTENT_XMAKE_STORAGE}/repositories"',
            self.text,
        )
        self.assertIn('LOCAL_PROJECT_DIR="${XMAKE_DIAG_TMP}/InfiniCore"', self.text)
        self.assertIn('cp -a "${PROJECT_DIR}" "${LOCAL_PROJECT_DIR}"', self.text)
        self.assertIn('cd -- "${LOCAL_PROJECT_DIR}"', self.text)

    def test_only_prunes_unused_native_dependencies_in_local_copy(self):
        self.assertIn('patch_local_native_project()', self.text)
        self.assertIn('add_requires("boost", {configs = {stacktrace = true}})', self.text)
        self.assertIn('add_requires("pybind11")', self.text)
        self.assertIn('sed -i', self.text)
        self.assertIn('"${LOCAL_PROJECT_DIR}/xmake.lua"', self.text)

    def test_configures_the_dtk_cuda_compatibility_root(self):
        self.assertIn('HYGON_CUDA_ROOT="/opt/dtk/cuda/cuda"', self.text)
        self.assertIn('export CUDA_HOME="${HYGON_CUDA_ROOT}"', self.text)
        self.assertIn('export CUDA_PATH="${HYGON_CUDA_ROOT}"', self.text)
        self.assertIn('export HYGON_CUDA_HOME="${HYGON_CUDA_ROOT}"', self.text)
        self.assertIn('export HYGON_ARCH="gfx936"', self.text)
        self.assertIn('check_hygon_cuda_root()', self.text)
        self.assertIn('"${HYGON_CUDA_ROOT}/bin/nvcc"', self.text)
        self.assertIn('--cuda="${HYGON_CUDA_ROOT}"', self.text)

    def test_collects_bounded_diagnostics_and_verbose_config(self):
        self.assertIn("timeout 60s", self.text)
        self.assertIn("xmake --version", self.text)
        self.assertIn("xmake l os.programdir", self.text)
        self.assertIn("xmake require --info boost", self.text)
        self.assertIn("xmake require --info pybind11", self.text)
        self.assertIn("xmake f -y -vD -m release", self.text)

    def test_never_builds_installs_or_cleans(self):
        for forbidden in (
            "xmake build",
            "xmake install",
            "xmake clean",
            "rm -rf",
            "sudo",
            "pip install",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.text)

    def test_writes_persistent_stage_logs_and_summary(self):
        self.assertIn("${LOG_DIR}/infinifabric-xmake-diagnostic-", self.text)
        self.assertIn("run.log", self.text)
        self.assertIn("summary.txt", self.text)
        self.assertIn("xmake-config-debug.log", self.text)
        self.assertIn("DIAGNOSTIC COMPLETE", self.text)


if __name__ == "__main__":
    unittest.main()
