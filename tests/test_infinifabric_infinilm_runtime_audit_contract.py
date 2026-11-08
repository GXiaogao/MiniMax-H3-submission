import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "artifacts" / "infinifabric-build-v1" / "audit_infinilm_runtime.sh"
HELPER_PATH = ROOT / "artifacts" / "infinifabric-build-v1" / "audit_infinilm_runtime.py"


class InfiniLMRuntimeAuditContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script_raw = SCRIPT_PATH.read_bytes()
        cls.script = cls.script_raw.decode("utf-8")
        cls.helper_raw = HELPER_PATH.read_bytes()
        cls.helper = cls.helper_raw.decode("utf-8")

    def test_files_are_lf_only_and_script_avoids_dev_fd(self):
        self.assertTrue(self.script_raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertTrue(self.helper_raw.startswith(b"#!/usr/bin/env python3\n"))
        self.assertNotIn(b"\r\n", self.script_raw)
        self.assertNotIn(b"\r\n", self.helper_raw)
        self.assertNotIn("/dev/fd", self.script)

    def test_requires_completed_native_python_and_extension_stages(self):
        self.assertIn("infinicore-python.success", self.script)
        self.assertIn("infinilm-extension.success", self.script)
        self.assertIn("test -f", self.script)

    def test_uses_persistent_sources_and_dynamic_libraries(self):
        self.assertIn("Projects/InfiniCore/python", self.script)
        self.assertIn('INFINILM_PROJECT="${WORK_HOME}/Projects/InfiniLM"', self.script)
        self.assertIn('INFINILM_PYTHON="${INFINILM_PROJECT}/python"', self.script)
        self.assertIn('pathlib.Path(torch.__file__).resolve().parent / "lib"', self.script)
        self.assertIn("LD_LIBRARY_PATH", self.script)
        self.assertIn("PYTHONPATH", self.script)

    def test_audits_static_imports_and_canonical_runtime_import(self):
        self.assertIn("ast.parse", self.helper)
        self.assertIn("importlib.util.find_spec", self.helper)
        self.assertIn('"infinilm"', self.helper)
        self.assertIn("from infinilm.multimodal.minimax_h3 import", self.helper)
        self.assertIn("traceback", self.helper)

    def test_records_machine_readable_results_without_installing(self):
        self.assertIn("runtime-audit.json", self.script)
        self.assertIn("missing-imports.txt", self.helper)
        self.assertIn("DIAGNOSTIC COLLECTED", self.script)
        self.assertIn("canonical_import=", self.helper)
        for token in ("pip install", "sudo", "rm -rf", "rm -r"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.script)
                self.assertNotIn(token, self.helper)

    def test_runtime_success_marker_is_only_written_after_full_imports(self):
        self.assertIn("infinilm-runtime-import.success", self.script)
        self.assertIn("runtime_ready", self.helper)
        self.assertIn('if [[ "${AUDIT_RC}" -eq 0 ]]', self.script)


if __name__ == "__main__":
    unittest.main()
