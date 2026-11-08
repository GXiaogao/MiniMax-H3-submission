import io
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "prepare_minimax_h3_hygon_runtime.sh"
HELPER = ARTIFACT_DIR / "prepare_minimax_h3_hygon_runtime.py"
AUDIT_SHELL = ARTIFACT_DIR / "audit_minimax_h3_hygon_runtime.sh"
AUDIT_HELPER = ARTIFACT_DIR / "audit_minimax_h3_hygon_runtime.py"
PROBE_SHELL = ARTIFACT_DIR / "probe_minimax_h3_native_load.sh"
PROBE_HELPER = ARTIFACT_DIR / "probe_minimax_h3_native_load.py"


class HygonRuntimePrepareContractTests(unittest.TestCase):
    def test_shell_builds_locally_and_publishes_only_ready_state(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("minimax-h3-hygon-runtime-inputs", text)
        self.assertIn("minimax-h3-hygon-runtime-ready", text)
        self.assertIn("minimax-h3-hygon-runtime.json", text)
        self.assertIn('LOCAL_ROOT="${PWD}/', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("curl ", text)
        self.assertNotIn("wget ", text)

    def test_helper_pins_inputs_builds_torchvision_and_validates_hygon_stack(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1",
            "e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581",
            "f799cdd1d67a3edbcdc6af8fb416fe1b019b512fb426c0314302cd81518a0095",
            "2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9",
            "b6a85871a79d2e3b22d2d1b94ac2824226a63c6b741c88f7ae975f18b6778374",
            "09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759",
            "44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669",
            "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9",
            "setup.py",
            "bdist_wheel",
            "BUILD_VERSION",
            "FORCE_CUDA",
            "stage_runtime",
            "import pytest",
            "FA_SO_PATH",
            "torchvision::nms",
            "scaled_dot_product_attention",
        ):
            self.assertIn(token, text)
        for forbidden in ("pip install", "http://", "https://"):
            self.assertNotIn(forbidden, text)

    def test_safe_tar_extraction_rejects_parent_traversal(self):
        namespace = runpy.run_path(str(HELPER))
        safe_extract_tar = namespace["safe_extract_tar"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("../escape")
                payload = b"bad"
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
            with self.assertRaises(RuntimeError):
                safe_extract_tar(archive, root / "output")

    def test_tar_validation_accepts_internal_parent_symlink(self):
        namespace = runpy.run_path(str(HELPER))
        validate_tar_member = namespace["validate_tar_member"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            member = tarfile.TarInfo(
                "vision-0.24.0/gallery/assets/coco/images/000000000001.jpg"
            )
            member.type = tarfile.SYMTYPE
            member.linkname = "../../astronaut.jpg"

            validate_tar_member(root, member)

    def test_tar_validation_rejects_escaping_parent_symlink(self):
        namespace = runpy.run_path(str(HELPER))
        validate_tar_member = namespace["validate_tar_member"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            member = tarfile.TarInfo("vision-0.24.0/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"

            with self.assertRaises(RuntimeError):
                validate_tar_member(root, member)

    def test_audit_and_native_probe_stage_the_prepared_runtime(self):
        for path in (AUDIT_SHELL, PROBE_SHELL):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("minimax-h3-hygon-runtime", text)
                self.assertIn("HYGON_RUNTIME_HELPER", text)
        for path in (AUDIT_HELPER, PROBE_HELPER):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("hygon-runtime-manifest", text)
                self.assertIn("stage_runtime", text)


if __name__ == "__main__":
    unittest.main()
