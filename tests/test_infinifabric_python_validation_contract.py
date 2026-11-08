import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "validate_infinicore_python.sh"
)


class InfiniCorePythonValidationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT_PATH.is_file():
            raise AssertionError(f"validation script is missing: {SCRIPT_PATH}")
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_lf_only_strict_bash_without_build_commands(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("xmake", self.text)
        self.assertNotIn("pip install", self.text)

    def test_requires_native_success_and_existing_extension(self):
        self.assertIn("infinicore-native.success", self.text)
        self.assertIn("_infinicore${EXT_SUFFIX}", self.text)
        self.assertIn('[[ -f "${EXTENSION}" ]]', self.text)

    def test_resolves_runtime_dependencies_before_import(self):
        self.assertIn("TORCH_LIB_DIR", self.text)
        self.assertIn("libtorch.so", self.text)
        self.assertIn("libtorch_hip.so", self.text)
        self.assertIn('export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:', self.text)
        self.assertIn('run_stage "ldd" ldd "${EXTENSION}"', self.text)
        self.assertIn('grep -Fq "not found"', self.text)

    def test_imports_canonical_package_and_checks_hygon_eight_cards(self):
        self.assertIn("import torch, infinicore", self.text)
        self.assertIn("Device.Type.HYGON", self.text)
        self.assertIn("get_device_count", self.text)
        self.assertIn("hygon_count == 8", self.text)
        self.assertIn("torch.cuda.device_count() == 8", self.text)
        self.assertIn("current.type == hygon", self.text)

    def test_writes_marker_only_after_validation(self):
        import_index = self.text.index('run_stage "import-smoke"')
        marker_index = self.text.index('CURRENT_STAGE="success-marker"')
        self.assertLess(import_index, marker_index)
        self.assertIn("infinicore-python.success", self.text)
        self.assertIn("mv -f", self.text)
        self.assertIn("InfiniCore Python validation: PASSED", self.text)


if __name__ == "__main__":
    unittest.main()
