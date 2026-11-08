import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "validate_infinicore_native.sh"
)


class InfiniCoreNativeValidationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT_PATH.is_file():
            raise AssertionError(f"validation script is missing: {SCRIPT_PATH}")
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_lf_only_strict_bash_without_dev_fd(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("/dev/fd", self.text)

    def test_reuses_installed_artifacts_without_building(self):
        self.assertIn("libinfinicore_cpp_api.so", self.text)
        self.assertIn("infinicore/tensor.hpp", self.text)
        self.assertNotIn("xmake build", self.text)
        self.assertNotIn("xmake install", self.text)
        self.assertNotIn("InfiniCore-infinifabric-v3.tar.gz", self.text)

    def test_resolves_and_checks_torch_runtime_libraries(self):
        self.assertIn("import torch", self.text)
        for library in (
            "libtorch.so",
            "libtorch_cpu.so",
            "libc10.so",
            "libc10_hip.so",
        ):
            with self.subTest(library=library):
                self.assertIn(library, self.text)
        self.assertIn('export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:', self.text)
        self.assertIn("ldd", self.text)
        self.assertIn("not found", self.text)

    def test_logs_hashes_and_writes_atomic_success_marker_last(self):
        self.assertIn("artifacts.sha256", self.text)
        self.assertIn("infinicore-native.success", self.text)
        self.assertIn("mv -f", self.text)
        hashes = self.text.index("artifacts.sha256")
        marker = self.text.index("infinicore-native.success")
        self.assertLess(hashes, marker)

    def test_reports_pass_or_failure_stage(self):
        self.assertIn("CURRENT_STAGE", self.text)
        self.assertIn("exit_code", self.text)
        self.assertIn("InfiniCore native validation: PASSED", self.text)
        self.assertIn("RESULT:", self.text)


if __name__ == "__main__":
    unittest.main()
