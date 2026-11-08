import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "preflight_minimax_h3_assets.sh"
HELPER = ARTIFACT_DIR / "preflight_minimax_h3_assets.py"


class MiniMaxH3AssetsPreflightContractTests(unittest.TestCase):
    def test_shell_stages_runtime_in_job_workspace_and_forwards_arguments(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("infinilm-runtime-private.json", text)
        self.assertIn('LOCAL_TARGET="${PWD}/', text)
        self.assertIn('"$@"', text)
        self.assertNotIn("python-envs", text)
        self.assertNotIn("pip install", text)

    def test_helper_reuses_the_validated_direct_wheel_stager(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "install_infinilm_private_runtime.py",
            "verify_wheels",
            "install_wheels_direct",
            "check_target_contents",
            '"job-local-wheel-staging"',
        ):
            self.assertIn(token, text)
        for forbidden in ("pip install", "from_pretrained", "http://", "https://"):
            self.assertNotIn(forbidden, text)

    def test_model_layout_validation_covers_fl2va_and_optional_ref2va(self):
        namespace = runpy.run_path(str(HELPER))
        validate_model_layout = namespace["validate_model_layout"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MiniMax-H3"
            diffusers = root / "diffusers"
            for component in ("text_encoder", "transformer"):
                component_dir = diffusers / component
                component_dir.mkdir(parents=True, exist_ok=True)
                (component_dir / "model.safetensors").write_bytes(b"weights")
            (diffusers / "model_index.json").write_text(
                json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
                encoding="utf-8",
            )

            report = validate_model_layout(root)
            self.assertTrue(report["fl2va_ready"])
            self.assertFalse(report["ref2va_ready"])
            self.assertEqual(report["errors"], [])

            ref2va = root / "Ref2VA"
            for component in ("text_encoder", "transformer"):
                component_dir = ref2va / component
                component_dir.mkdir(parents=True, exist_ok=True)
                (component_dir / "model.safetensors").write_bytes(b"weights")
            (ref2va / "model_index.json").write_text(
                json.dumps({"_minimax_h3": {"partition": "ref2va"}}),
                encoding="utf-8",
            )
            report = validate_model_layout(root)
            self.assertTrue(report["ref2va_ready"])

    def test_weight_index_references_must_exist(self):
        namespace = runpy.run_path(str(HELPER))
        validate_model_layout = namespace["validate_model_layout"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MiniMax-H3"
            diffusers = root / "diffusers"
            for component in ("text_encoder", "transformer"):
                (diffusers / component).mkdir(parents=True, exist_ok=True)
            (diffusers / "model_index.json").write_text(
                json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
                encoding="utf-8",
            )
            (diffusers / "text_encoder" / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "missing-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            (diffusers / "transformer" / "model.safetensors").write_bytes(b"weights")

            report = validate_model_layout(root)
            self.assertFalse(report["fl2va_ready"])
            self.assertTrue(any("missing-00001-of-00001.safetensors" in item for item in report["errors"]))

    def test_unreadable_shared_candidate_is_reported_without_aborting_discovery(self):
        namespace = runpy.run_path(str(HELPER))
        safe_validate = namespace["safe_validate_model_layout"]
        original = safe_validate.__globals__["validate_model_layout"]
        try:
            safe_validate.__globals__["validate_model_layout"] = lambda _path: (_ for _ in ()).throw(
                PermissionError(13, "Permission denied", "/shared/private/model_index.json")
            )
            report = safe_validate(Path("/shared/private"))
        finally:
            safe_validate.__globals__["validate_model_layout"] = original
        self.assertFalse(report["fl2va_ready"])
        self.assertFalse(report["ref2va_ready"])
        self.assertTrue(any("Permission denied" in item for item in report["errors"]))

    def test_permission_denied_during_file_type_probe_is_skipped(self):
        namespace = runpy.run_path(str(HELPER))
        safe_is_file = namespace["safe_is_file"]
        with mock.patch.object(
            Path,
            "is_file",
            side_effect=PermissionError(13, "Permission denied", "/shared/private/model_index.json"),
        ):
            self.assertFalse(safe_is_file(Path("/shared/private/model_index.json")))


if __name__ == "__main__":
    unittest.main()
