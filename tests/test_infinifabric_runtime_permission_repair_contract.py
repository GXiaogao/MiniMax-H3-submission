import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "repair_infinicore_runtime_permissions.sh"
)


class RuntimePermissionRepairContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_lf_only_strict_bash_without_dev_fd(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("/dev/fd", self.text)

    def test_requires_both_prior_success_markers_and_extensions(self):
        self.assertIn("infinicore-python.success", self.text)
        self.assertIn("infinilm-extension.success", self.text)
        self.assertIn("_infinicore${EXT_SUFFIX}", self.text)
        self.assertIn("_infinilm${EXT_SUFFIX}", self.text)

    def test_records_modes_before_and_after_group_read_repair(self):
        self.assertIn("permissions-before", self.text)
        self.assertIn("permissions-after", self.text)
        self.assertIn("stat -Lc", self.text)
        self.assertIn("chmod g+r", self.text)
        self.assertIn("chgrp", self.text)
        self.assertIn("nogroup", self.text)
        self.assertIn("already_shared_group_read", self.text)
        self.assertNotIn("chmod -R", self.text)
        self.assertNotIn("chmod 777", self.text)

    def test_validates_canonical_infinicore_and_direct_infinilm_extension(self):
        self.assertIn("import infinicore", self.text)
        self.assertIn("import _infinilm", self.text)
        self.assertIn("hygon_count", self.text)
        self.assertIn("torch_device_count", self.text)
        self.assertIn("InferEngine", self.text)
        self.assertIn("DistConfig", self.text)

    def test_does_not_install_or_delete_content(self):
        for token in ("pip install", "sudo", "rm -r", "rm -f", "find "):
            with self.subTest(token=token):
                self.assertNotIn(token, self.text)

    def test_writes_separate_repair_marker_after_validation(self):
        self.assertIn("runtime-permissions.success", self.text)
        self.assertIn("Permission repair validation: PASSED", self.text)
        self.assertIn("RESULT:", self.text)


if __name__ == "__main__":
    unittest.main()
