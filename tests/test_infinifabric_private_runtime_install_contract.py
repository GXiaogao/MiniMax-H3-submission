import unittest
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "install_infinilm_private_runtime.sh"
HELPER = ARTIFACT_DIR / "install_infinilm_private_runtime.py"


class PrivateRuntimeInstallContractTests(unittest.TestCase):
    def test_shell_is_portable_and_uses_dedicated_afs_wheel_directory(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("packages/infinilm-runtime-wheels", text)
        self.assertIn("install_infinilm_private_runtime.py", text)
        self.assertIn('tee -a "${RUN_LOG}"', text)
        self.assertNotIn("PERSISTENT_ROOT", text)
        self.assertIn("PYTHONUNBUFFERED=1", text)
        self.assertIn('"${PYTHON}" -u "${HELPER}"', text)
        for forbidden in ("sudo", "/dev/fd", "/usr/local/lib/python", "pip install --user"):
            self.assertNotIn(forbidden, text)

    def test_helper_pins_all_uploaded_wheels_and_checks_hashes(self):
        text = HELPER.read_text(encoding="utf-8")
        for filename in (
            "accelerate-1.15.0-py3-none-any.whl",
            "av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl",
            "decord-0.6.0-py3-none-manylinux2010_x86_64.whl",
            "diffusers-0.41.0-py3-none-any.whl",
            "httpx-0.28.1-py3-none-any.whl",
            "janus-2.0.0-py3-none-any.whl",
            "lazy_loader-0.4-py3-none-any.whl",
            "msgspec-0.22.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
            "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "scikit_image-0.26.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
            "scipy-1.11.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "xxhash-4.0.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        ):
            with self.subTest(filename=filename):
                self.assertIn(filename, text)
        self.assertIn("hashlib.sha256", text)
        self.assertIn("EXPECTED_WHEELS", text)

    def test_install_directly_unpacks_wheels_without_starting_pip(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "zipfile.ZipFile",
            "safe_extract_wheel",
            '"purelib"',
            '"platlib"',
            "local_target",
        ):
            self.assertIn(token, text)
        self.assertNotIn('"-m",\n        "pip"', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("https://", text)

    def test_protected_platform_stack_is_excluded_and_checked(self):
        text = HELPER.read_text(encoding="utf-8")
        for name in (
            "torch",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "safetensors",
            "numpy",
        ):
            self.assertIn(f'"{name}"', text)
        self.assertIn("protected_stack_contamination", text)

    def test_private_runtime_pins_the_compatible_huggingface_dependency_closure(self):
        namespace = runpy.run_path(str(HELPER))
        expected = namespace["EXPECTED_DISTRIBUTIONS"]
        protected = namespace["PROTECTED_STACK"]
        required_versions = {
            "annotated-doc": "0.0.5",
            "click": "8.4.2",
            "h11": "0.16.0",
            "hf-xet": "1.5.2",
            "httpcore": "1.0.9",
            "huggingface-hub": "1.32.0",
            "imageio": "2.34.2",
            "markdown-it-py": "4.2.0",
            "mdurl": "0.1.2",
            "pygments": "2.21.0",
            "regex": "2026.9.29",
            "rich": "15.0.0",
            "safetensors": "0.8.0",
            "shellingham": "1.5.4",
            "tifffile": "2024.8.30",
            "tokenizers": "0.23.3",
            "transformers": "5.20.2",
            "typer": "0.27.3",
        }
        for distribution, version in required_versions.items():
            with self.subTest(distribution=distribution):
                self.assertEqual(expected.get(distribution), version)
        self.assertEqual(protected, {"torch", "numpy"})

    def test_validation_precedes_atomic_wheel_bundle_marker_without_afs_tree_copy(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertLess(text.index("validate_runtime(local_target"), text.index("marker_temp.write_text"))
        self.assertNotIn("shutil.copytree", text)
        self.assertNotIn("publish_readable", text)
        self.assertNotIn("persistent_target", text)
        self.assertIn("infinilm-runtime-private.json", text)
        self.assertIn('"runtime_mode": "job-local-wheel-staging"', text)
        self.assertIn("os.replace", text)
        self.assertIn("infinilm-runtime-private.path", text)
        self.assertIn("runtime-install-result.json", text)
        self.assertIn("RESULT: PASSED", text)

    def test_helper_emits_unbuffered_stage_progress(self):
        text = HELPER.read_text(encoding="utf-8")
        self.assertIn("def stage_start(", text)
        self.assertIn("def stage_done(", text)
        for stage in (
            "wheel-verification",
            "local-wheel-unpack",
            "local-runtime-validation",
            "state-manifest",
            "state-marker",
        ):
            with self.subTest(stage=stage):
                self.assertIn(f'stage_start(result, "{stage}")', text)
        self.assertIn("flush=True", text)

    def test_validation_covers_minimax_h3_and_runtime_dependencies(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "MiniMaxH3DiffusersAdapter",
            "MiniMaxH3Transformer3DModel",
            "Qwen3VLForConditionalGeneration",
            "import janus",
            "import lazy_loader",
            "import av",
            "import decord",
            "import cv2",
            "import skimage",
            "import scipy",
            "import msgspec",
            "import xxhash",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
