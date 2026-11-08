import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "build_infinicore_python.sh"
)


class InfiniCorePythonBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT_PATH.is_file():
            raise AssertionError(f"build script is missing: {SCRIPT_PATH}")
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_lf_only_strict_bash_without_dev_fd(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("/dev/fd", self.text)

    def test_requires_verified_native_stage_and_fixed_source(self):
        self.assertIn("infinicore-native.success", self.text)
        self.assertIn("InfiniCore-infinifabric-v3.tar.gz", self.text)
        self.assertIn(
            "37414edbc25545063462a3cbe47811ca0a672378058d78b761ec36e6fc3f8811",
            self.text,
        )
        self.assertIn("sha256sum --check", self.text)

    def test_uses_job_local_xmake_state_and_offline_pybind(self):
        self.assertIn('LOCAL_ROOT="${PWD}/.infinicore-python-${UID}-$$"', self.text)
        self.assertIn('export XMAKE_GLOBALDIR="${LOCAL_ROOT}/global"', self.text)
        self.assertIn('xmake-repo-cache.tar.gz', self.text)
        self.assertIn('pybind11-v3.1.0.zip', self.text)
        self.assertIn('xmake g --network=private --pkg_searchdirs="${PKG_DIR}"', self.text)

    def test_extracts_verified_pybind_headers_and_bypasses_package_download(self):
        self.assertIn('-m zipfile -e "${PYBIND_ARCHIVE}"', self.text)
        self.assertIn('pybind11-src/pybind11-3.1.0', self.text)
        self.assertIn('LOCAL_PYBIND_INCLUDE="${LOCAL_PYBIND_ROOT}/include"', self.text)
        self.assertIn('pybind11/pybind11.h', self.text)
        self.assertIn('add_requires("pybind11")', self.text)
        self.assertIn('add_packages("pybind11")', self.text)
        self.assertIn('add_includedirs(\\"${LOCAL_PYBIND_INCLUDE}\\"', self.text)

    def test_injects_active_python_development_headers(self):
        self.assertIn('sysconfig.get_paths()["include"]', self.text)
        self.assertIn('PYTHON_INCLUDE_DIR', self.text)
        self.assertIn('${PYTHON_INCLUDE_DIR}/Python.h', self.text)
        self.assertIn('\\"${PYTHON_INCLUDE_DIR}\\"', self.text)

    def test_links_verified_native_install_without_rebuilding_it(self):
        self.assertIn('add_deps("infinicore_cpp_api")', self.text)
        self.assertIn('add_links("infinicore_cpp_api")', self.text)
        self.assertIn('xmake build -j "${BUILD_JOBS}" _infinicore', self.text)
        self.assertNotIn('xmake build -j "${BUILD_JOBS}" infinicore_cpp_api', self.text)
        self.assertNotIn("xmake install infinicore_cpp_api", self.text)

    def test_reuses_same_hygon_release_configuration(self):
        for token in (
            "--hygon-dcu=y",
            '--hygon-arch="${HYGON_ARCH}"',
            '--cuda="${HYGON_CUDA_ROOT}"',
            "--cpu=n",
            "--omp=n",
            "--aten=y",
            "--ccl=y",
            "--graph=n",
            "--infiniops=n",
            "--ninetoothed=n",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_validates_then_atomically_promotes_python_extension(self):
        self.assertIn("EXT_SUFFIX", self.text)
        self.assertIn("_infinicore${EXT_SUFFIX}", self.text)
        self.assertIn("ldd", self.text)
        self.assertIn("not found", self.text)
        self.assertIn("mv -f", self.text)
        self.assertIn("import torch, infinicore", self.text)
        self.assertIn("infinicore-python.success", self.text)
        copy_index = self.text.index('mv -f -- "${EXTENSION_TMP}"')
        marker_index = self.text.index("infinicore-python.success")
        self.assertLess(copy_index, marker_index)

    def test_requires_readable_native_inputs_without_changing_their_modes(self):
        self.assertIn('verify_native_inputs_readable()', self.text)
        self.assertIn('[[ -r "${native_path}" ]]', self.text)
        self.assertIn('run_stage "native-input-readability" verify_native_inputs_readable', self.text)
        self.assertNotIn('chmod o+r -- "${PERSISTENT_INFINI_ROOT}', self.text)

    def test_publishes_python_extension_for_future_job_uids(self):
        self.assertIn('run_stage "built-extension-readability" chmod o+r -- "${BUILT_EXTENSION}"', self.text)
        self.assertIn('if [[ -r "${DESTINATION_EXTENSION}" ]]', self.text)
        self.assertIn('previous_extension_backup=SKIPPED_UNREADABLE', self.text)
        self.assertIn('[[ -r "${DESTINATION_EXTENSION}" ]]', self.text)
        chmod_index = self.text.index('run_stage "built-extension-readability"')
        promote_index = self.text.index('CURRENT_STAGE="extension-promotion"')
        self.assertLess(chmod_index, promote_index)
        self.assertNotIn('chmod 777', self.text)

    def test_avoids_shared_python_mutation_and_destructive_operations(self):
        for token in ("pip install", "sudo", "/usr/local", "rm -rf", "rm -r"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.text)

    def test_records_failure_stage_and_final_result(self):
        self.assertIn("CURRENT_STAGE", self.text)
        self.assertIn("exit_code", self.text)
        self.assertIn("InfiniCore Python binding: PASSED", self.text)
        self.assertIn("RESULT:", self.text)


if __name__ == "__main__":
    unittest.main()
