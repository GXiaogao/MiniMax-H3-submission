import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "diagnose_infinilm_full_runtime.sh"
HELPER = ARTIFACT_DIR / "diagnose_infinilm_full_runtime.py"
MANIFEST = ARTIFACT_DIR / "infinilm-local-wheel-inventory.json"


class FullRuntimeDiagnosticContractTests(unittest.TestCase):
    def test_shell_is_read_only_lf_bash_and_loads_persistent_environment(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("diagnose_infinilm_full_runtime.py", text)
        self.assertIn("infinilm-local-wheel-inventory.json", text)
        self.assertIn('tee -a "${RUN_LOG}"', text)
        for forbidden in ("pip install", "sudo", "rm -rf", "/dev/fd"):
            self.assertNotIn(forbidden, text)

    def test_helper_scans_source_imports_and_independent_runtime_surfaces(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("ast.parse", text)
        self.assertIn("source_missing_imports", text)
        for surface in (
            "infinicore-native",
            "infinilm-core",
            "minimax-h3-adapter",
            "accelerate-api",
            "diffusers-minimax-h3-api",
            "transformers-qwen3vl-api",
            "video-runtime",
            "image-runtime",
            "mooncake-optional",
        ):
            with self.subTest(surface=surface):
                self.assertIn(surface, text)

    def test_helper_checks_wheel_compatibility_dependencies_and_protected_stack(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("parse_wheel_filename", text)
        self.assertIn("sys_tags", text)
        self.assertIn("Requirement", text)
        self.assertIn("candidate_dependency_issues", text)
        self.assertIn("protected_stack_conflicts", text)
        for dist in (
            "torch",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "safetensors",
            "numpy",
        ):
            self.assertIn(f'"{dist}"', text)
        self.assertNotIn("subprocess.run([sys.executable, \"-m\", \"pip\", \"install\"", text)

    def test_manifest_captures_local_candidates_and_mooncake_gap(self):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        wheels = data["wheels"]
        names = {item["normalized_name"] for item in wheels}
        for name in (
            "accelerate",
            "av",
            "decord",
            "diffusers",
            "httpx",
            "janus",
            "msgspec",
            "opencv-python-headless",
            "scikit-image",
            "scipy",
            "xxhash",
        ):
            self.assertIn(name, names)
        self.assertNotIn("mooncake-transfer-engine", names)
        self.assertEqual(data["source_directory_name"], "wheels")
        self.assertTrue(all(len(item["sha256"]) == 64 for item in wheels))

    def test_helper_writes_machine_and_human_readable_results(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("full-runtime-diagnostic.json", text)
        self.assertIn("missing-runtime.txt", text)
        self.assertIn("runtime_ready", text)
        self.assertIn("diagnostic_complete", text)
        self.assertIn("conditional_missing", text)


if __name__ == "__main__":
    unittest.main()
