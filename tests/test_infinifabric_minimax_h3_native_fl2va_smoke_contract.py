import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "run_minimax_h3_fl2va_native_smoke.sh"
HELPER = ARTIFACT_DIR / "run_minimax_h3_fl2va_native_smoke.py"
README = ARTIFACT_DIR / "README.md"


class MiniMaxH3NativeFl2vaSmokeContractTests(unittest.TestCase):
    def test_helper_profiles_select_distinct_evidence_names_and_labels(self):
        namespace = runpy.run_path(str(HELPER))
        self.assertIn("profile_result_filename", namespace)
        self.assertIn("profile_success_label", namespace)
        profile_result_filename = namespace["profile_result_filename"]
        profile_success_label = namespace["profile_success_label"]

        self.assertEqual(
            profile_result_filename("smoke"),
            "minimax-h3-native-fl2va-smoke.json",
        )
        self.assertEqual(
            profile_result_filename("formal"),
            "minimax-h3-native-fl2va-formal.json",
        )
        self.assertEqual(profile_success_label("smoke"), "NATIVE_FL2VA_SMOKE PASSED")
        self.assertEqual(profile_success_label("formal"), "NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED")

    def run_with_device_boundary_fakes(self, root, profile, media_error=None):
        """Exercise real runner reporting without GPU/wheel-install side effects."""
        namespace = runpy.run_path(str(HELPER))
        model = root / "model"
        for name in ("model_index.json", "text_encoder/config.json", "transformer/config.json"):
            path = model / "diffusers" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        image = root / "first.jpg"
        image.touch()
        example = root / "example.py"
        example.write_text("def main(): pass\n", encoding="utf-8")
        state = root / "state.json"
        state.write_text(json.dumps({"runtime_mode": "job-local-wheel-staging",
                                    "expected_wheels": [], "wheel_dir": str(root)}))
        args = SimpleNamespace(
            run_dir=root / "run", run_profile=profile, model_path=model,
            first_frame=image, output=root / "out.mp4", example=example,
            state_manifest=state, runtime_helper=root / "installer.py",
            hygon_runtime_helper=root / "hygon.py", hygon_runtime_manifest=state,
            local_target=root / "site-packages", gpus="0,1,2,3,4,5,6,7",
            text="A dog falls asleep", num_frames=124, steps=10, seed=42,
            height=384, width=576)
        installer = SimpleNamespace(EXPECTED_WHEELS=[], verify_wheels=lambda _: [],
                                    install_wheels_direct=lambda *_: None,
                                    check_target_contents=lambda _: [])
        hygon = SimpleNamespace(stage_runtime=lambda *_: {})
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 8,
                               synchronize=lambda _: None)
        native = SimpleNamespace(Device=SimpleNamespace(Type=SimpleNamespace(HYGON=1, CPU=2)),
                                 get_device_count=lambda kind: 8 if kind == 1 else 1)
        modules = {"torch": SimpleNamespace(cuda=cuda), "infinicore": SimpleNamespace(),
                   "infinicore.lib": SimpleNamespace(_infinicore=native)}
        def inspect(*_, **kwargs):
            if media_error:
                raise RuntimeError(media_error)
            return {"width": 576, "height": 384, "decoded_frame_count": 124,
                    "audio_stream_count": 1}
        replacements = {
            "faulthandler": SimpleNamespace(enable=lambda **_: None),
            "parse_args": lambda: args,
            "load_helper": lambda path, *_: installer if path == args.runtime_helper else hygon,
            "memory_snapshot": lambda *_: [],
            "invoke_official_example": lambda *_: None,
            "inspect_media": inspect,
        }
        output = io.StringIO()
        with patch.dict(namespace["main"].__globals__, replacements), \
                patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path[:]), \
                patch.dict(__import__("os").environ), \
                redirect_stdout(output), redirect_stderr(output):
            rc = namespace["main"]()
        path = args.run_dir / namespace["profile_result_filename"](profile)
        return rc, json.loads(path.read_text(encoding="utf-8")), output.getvalue()

    def test_formal_media_success_is_not_reported_as_quality_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result, output = self.run_with_device_boundary_fakes(Path(tmp), "formal")
        self.assertEqual(rc, 0)
        self.assertEqual(result["result"], "STRUCTURAL_PASSED")
        self.assertEqual(result.get("quality_status"), "UNASSESSED")
        self.assertEqual(result["validation_scope"], "media-structure-only")
        self.assertIn("NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED", output)
        self.assertNotIn("NATIVE_FL2VA_FORMAL PASSED", output)

    def test_media_failure_retains_failure_and_unassessed_quality_in_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result, output = self.run_with_device_boundary_fakes(
                Path(tmp), "formal", "unexpected decoded frame count: 123")
        self.assertEqual(rc, 1)
        self.assertEqual(result["result"], "FAILED")
        self.assertEqual(result["last_stage"], "media-validation")
        self.assertEqual(result.get("quality_status"), "UNASSESSED")
        self.assertIn("unexpected decoded frame count", result["error"])
        self.assertNotIn("STRUCTURAL_PASSED", output)

    def test_smoke_retains_its_label_but_records_quality_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, result, output = self.run_with_device_boundary_fakes(Path(tmp), "smoke")
        self.assertEqual(rc, 0)
        self.assertEqual(result["result"], "PASSED")
        self.assertEqual(result.get("quality_status"), "UNASSESSED")
        self.assertIn("NATIVE_FL2VA_SMOKE PASSED", output)

    def test_helper_run_profile_defaults_to_smoke_accepts_formal_and_rejects_unknown(self):
        namespace = runpy.run_path(str(HELPER))
        parse_args = namespace["parse_args"]
        required = [
            "runner.py",
            "--runtime-helper=/tmp/runtime.py",
            "--hygon-runtime-helper=/tmp/hygon.py",
            "--hygon-runtime-manifest=/tmp/hygon.json",
            "--state-manifest=/tmp/state.json",
            "--local-target=/tmp/site-packages",
            "--run-dir=/tmp/run",
            "--example=/tmp/example.py",
            "--first-frame=/tmp/frame.jpg",
            "--output=/tmp/output.mp4",
        ]

        with patch.object(sys, "argv", required):
            self.assertEqual(parse_args().run_profile, "smoke")
        with patch.object(sys, "argv", [*required, "--run-profile=formal"]):
            self.assertEqual(parse_args().run_profile, "formal")
        with patch.object(sys, "argv", [*required, "--run-profile=unknown"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_shell_uses_persistent_inputs_outputs_logs_and_bounded_runtime(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        for token in (
            "set -Eeuo pipefail",
            'source "${WORK_HOME}/setup_infinifabric_env.sh"',
            'FIRST_FRAME="${EFFECTIVE_WORK_HOME}/inputs/fl2va/Golde33443.jpg"',
            'OUTPUT_DIR="${EFFECTIVE_WORK_HOME}/outputs/fl2va-native-smoke"',
            'RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-run-minimax-h3-native-fl2va-smoke-${RUN_ID}"',
            'timeout --signal=TERM --kill-after=30s 25m',
            'tee -a "${RUN_LOG}"',
            '"$@"',
        ):
            self.assertIn(token, text)
        self.assertNotIn("pip install", text)

    def test_helper_builds_exact_official_native_smoke_arguments(self):
        namespace = runpy.run_path(str(HELPER))
        build_official_argv = namespace["build_official_argv"]
        args = SimpleNamespace(
            model_path=Path("/models/MiniMax-H3"),
            first_frame=Path("/inputs/Golde33443.jpg"),
            output=Path("/outputs/smoke.mp4"),
            text="A dog falls asleep",
            gpus="0,1,2,3,4,5,6,7",
            num_frames=124,
            steps=2,
            seed=42,
            height=384,
            width=576,
        )

        argv = build_official_argv(args)

        self.assertEqual(argv[0], "fl2va")
        for expected in (
            "--model-path=/models/MiniMax-H3",
            "--text=A dog falls asleep",
            "--first-frame=/inputs/Golde33443.jpg",
            "--gpus=0,1,2,3,4,5,6,7",
            "--num-frames=124",
            "--steps=2",
            "--seed=42",
            "--height=384",
            "--width=576",
            "--output=/outputs/smoke.mp4",
        ):
            self.assertIn(expected, argv)
        self.assertNotIn("--legacy", argv)
        self.assertFalse(any(item.startswith("--legacy-") for item in argv))

    def test_helper_requires_all_eight_unique_devices(self):
        namespace = runpy.run_path(str(HELPER))
        parse_devices = namespace["parse_devices"]
        validate_device_counts = namespace["validate_device_counts"]
        devices = parse_devices("0,1,2,3,4,5,6,7")
        self.assertEqual(devices, list(range(8)))
        validate_device_counts(devices, torch_count=8, native_count=8)
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            parse_devices("0,1,2,3")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_devices("0,1,2,3,4,5,6,6")
        with self.assertRaisesRegex(RuntimeError, "torch reports 7"):
            validate_device_counts(devices, torch_count=7, native_count=8)
        with self.assertRaisesRegex(RuntimeError, "InfiniCore reports 7"):
            validate_device_counts(devices, torch_count=8, native_count=7)

    def test_cpu_less_runtime_stages_checkpoint_tensors_on_first_dcu(self):
        namespace = runpy.run_path(str(HELPER))
        configure_weight_staging = namespace["configure_weight_staging"]

        class FakeTensor:
            def __init__(self, device_type):
                self.device = SimpleNamespace(type=device_type)
                self.moved_to = None

            def to(self, *, device):
                self.moved_to = device
                return FakeTensor("cuda")

        class FakeInfiniCore:
            def __init__(self):
                self.received = []

            def from_torch(self, tensor):
                self.received.append(tensor.device.type)
                return tensor.device.type

        module = FakeInfiniCore()
        report = configure_weight_staging(module, list(range(8)), native_cpu_count=0)
        tensor = FakeTensor("cpu")
        converted = module.from_torch(tensor)
        self.assertEqual(report["mode"], "hygon-gpu-overlay")
        self.assertEqual(report["target_device"], "cuda:0")
        self.assertEqual(tensor.moved_to, "cuda:0")
        self.assertEqual(module.received, ["cuda"])
        self.assertEqual(converted, "cuda")

    def test_official_example_loader_registers_module_before_dataclass_exec(self):
        namespace = runpy.run_path(str(HELPER))
        load_official_example = namespace["load_official_example"]
        with tempfile.TemporaryDirectory() as temporary:
            example = Path(temporary) / "example.py"
            example.write_text(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class ModelPaths:\n"
                "    root: str\n"
                "def main():\n"
                "    return None\n",
                encoding="utf-8",
            )
            try:
                module = load_official_example(example)
                self.assertIs(sys.modules.get(module.__name__), module)
                self.assertEqual(module.ModelPaths("/models").root, "/models")
            finally:
                sys.modules.pop("infinifabric_official_minimax_h3_example", None)

    def test_official_stage_tracing_prints_start_done_and_failed_markers(self):
        namespace = runpy.run_path(str(HELPER))
        install_official_stage_tracing = namespace["install_official_stage_tracing"]

        class FakeTimingReport:
            @contextmanager
            def measure(self, name, *, synchronize=False):
                del name, synchronize
                yield

        fake_module = SimpleNamespace(TimingReport=FakeTimingReport)
        install_official_stage_tracing(fake_module)
        output = io.StringIO()
        report = FakeTimingReport()
        with redirect_stdout(output):
            with report.measure("compute.condition_encoder", synchronize=True):
                pass
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with report.measure("compute.h3_transformer"):
                    raise RuntimeError("boom")

        rendered = output.getvalue()
        for expected in (
            "native_trace stage=compute.condition_encoder status=START synchronize=true",
            "native_trace stage=compute.condition_encoder status=DONE",
            "native_trace stage=compute.h3_transformer status=START synchronize=false",
            "native_trace stage=compute.h3_transformer status=FAILED exception=RuntimeError",
        ):
            self.assertIn(expected, rendered)

    def test_helper_has_progress_result_and_strict_media_validation_contract(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            "faulthandler.enable(all_threads=True)",
            "fatal_signal_trace=enabled",
            "install_official_stage_tracing(module)",
            "class StageHeartbeat",
            "stage=job-local-runtime-staging status=START",
            "stage=official-native-fl2va status=START",
            "stage=media-validation status=START",
            "decoded_frame_count",
            "expected_width=args.width",
            "expected_height=args.height",
            "expected_frame_count=args.num_frames",
            "audio_stream_count",
            "NATIVE_FL2VA_SMOKE PASSED",
            "minimax-h3-native-fl2va-smoke.json",
        ):
            self.assertIn(token, text)
        self.assertIn("import av", text)
        self.assertIn("module.main()", text)
        self.assertNotIn("subprocess.run", text)

    def test_shell_enables_fatal_signal_reporting_and_labels_segfaults(self):
        text = SHELL.read_text(encoding="utf-8")
        self.assertIn("export PYTHONFAULTHANDLER=1", text)
        self.assertIn('if [[ "${HELPER_RC}" -eq 139 ]]', text)
        self.assertIn("native process terminated with SIGSEGV", text)

    def test_shell_rejects_stale_infinilm_extension_before_model_loading(self):
        text = SHELL.read_text(encoding="utf-8")
        for token in (
            'SMOKE_VERSION="1.3-minimax-h3-native-extension-gate"',
            'EXPECTED_INFINILM_BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"',
            'INFINILM_SUCCESS_MARKER="${EFFECTIVE_WORK_HOME}/state/infinilm-extension.success"',
            'require_file "${INFINILM_SUCCESS_MARKER}"',
            'marker_build_version="$(read_marker_value build_version "${INFINILM_SUCCESS_MARKER}")"',
            'marker_extension="$(read_marker_value extension "${INFINILM_SUCCESS_MARKER}")"',
            'if [[ "${marker_build_version}" != "${EXPECTED_INFINILM_BUILD_VERSION}" ]]',
            'require_file "${marker_extension}"',
            'extension_sha256="$(sha256sum "${marker_extension}" | awk',
            'log_line "infinilm_extension_sha256=${extension_sha256}"',
        ):
            self.assertIn(token, text)
        self.assertLess(
            text.index('require_file "${INFINILM_SUCCESS_MARKER}"'),
            text.index('timeout --signal=TERM --kill-after=30s 25m'),
        )

    def test_readme_documents_upload_launch_success_and_scope(self):
        text = README.read_text(encoding="utf-8")
        for token in (
            "run_minimax_h3_fl2va_native_smoke.sh",
            "run_minimax_h3_fl2va_native_smoke.py",
            "inputs/fl2va/Golde33443.jpg",
            "NATIVE_FL2VA_SMOKE PASSED",
            "outputs/fl2va-native-smoke",
            "steps=2",
            "steps=10",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
