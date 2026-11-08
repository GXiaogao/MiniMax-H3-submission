import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "audit_minimax_h3_hygon_runtime.sh"
HELPER = ARTIFACT_DIR / "audit_minimax_h3_hygon_runtime.py"


class MiniMaxH3HygonRuntimeAuditContractTests(unittest.TestCase):
    def test_shell_uses_job_local_private_runtime_and_never_installs(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("infinilm-runtime-private.json", text)
        self.assertIn('LOCAL_TARGET="${PWD}/', text)
        self.assertIn('"$@"', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("curl ", text)
        self.assertNotIn("wget ", text)

    def test_helper_checks_pod_guide_prerequisites_without_loading_weights(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "torchvision",
            "torchvision::nms",
            "torchvision.transforms",
            "einops",
            "triton",
            "flash_attn",
            "FA_SO_PATH",
            "scaled_dot_product_attention",
            "enable_flash_sdp",
            "AutoProcessor",
            "local_files_only=True",
            "rtol=0.02",
            "atol=0.02",
        ):
            self.assertIn(token, text)
        for forbidden in (
            "pip install",
            "http://",
            "https://",
            "MiniMaxH3ConditionEncoder(",
            "MiniMaxH3TransformerRunner(",
        ):
            self.assertNotIn(forbidden, text)

    def test_helper_stages_hygon_runtime_in_a_separate_overlay(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn(
            'hygon_target = local_target.parent / "hygon-site-packages"', text
        )
        self.assertIn(
            "installer.install_wheels_direct(wheels, local_target)", text
        )
        self.assertIn(
            "installer,\n            hygon_target,\n", text
        )
        self.assertNotIn(
            "installer,\n            local_target,\n            local_target.parent / \"hygon-reconstructed\"",
            text,
        )

    def test_readiness_requires_every_critical_probe(self):
        namespace = runpy.run_path(str(HELPER))
        evaluate_readiness = namespace["evaluate_readiness"]
        ready_report = {
            "hardware": {"torch_cuda_available": True, "torch_device_count": 8},
            "packages": {
                "torchvision": {"import_ok": True},
                "einops": {"import_ok": True},
                "triton": {"import_ok": True},
                "flash-attn": {"import_ok": True},
            },
            "torchvision": {"transforms_ok": True, "nms_registered": True},
            "processor": {"ready": True},
            "flash_attention": {"shared_object_ready": True},
            "sdpa": {"flash_forward_ok": True, "allclose": True},
        }
        readiness = evaluate_readiness(ready_report)
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["missing"], [])

        ready_report["torchvision"]["nms_registered"] = False
        readiness = evaluate_readiness(ready_report)
        self.assertFalse(readiness["ready"])
        self.assertIn("torchvision::nms", readiness["missing"])

    def test_audit_is_diagnostic_when_a_package_is_absent(self):
        namespace = runpy.run_path(str(HELPER))
        package_probe = namespace["package_probe"]
        result = package_probe("a-distribution-that-does-not-exist", "missing_module_xyz")
        self.assertFalse(result["import_ok"])
        self.assertIsNone(result["version"])
        self.assertIn("not found", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
