import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
SHELL = ARTIFACT_DIR / "probe_minimax_h3_native_load.sh"
HELPER = ARTIFACT_DIR / "probe_minimax_h3_native_load.py"


class MiniMaxH3NativeLoadProbeContractTests(unittest.TestCase):
    def test_shell_uses_job_local_runtime_and_persistent_logs(self):
        raw = SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn("infinilm-runtime-private.json", text)
        self.assertIn('LOCAL_TARGET="${PWD}/', text)
        self.assertIn('tee -a "${RUN_LOG}"', text)
        self.assertIn('"$@"', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("python-envs", text)

    def test_helper_defaults_to_eight_card_fl2va_encoder_load(self):
        text = HELPER.read_text(encoding="utf-8")
        for token in (
            'default="encoder"',
            'default="fl2va"',
            'default="0,1,2,3,4,5,6,7"',
            'default=Path("/public/home/aclsylqidf/shared/MiniMax-H3")',
            "MiniMaxH3ConditionEncoder",
            "MiniMaxH3TransformerRunner",
            "tp_device_ids=devices",
            "install_wheels_direct",
            "check_target_contents",
            "memory_before",
            "memory_after_load",
        ):
            self.assertIn(token, text)
        for forbidden in (".forward(", "generate(", "from_pretrained", "pip install"):
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

    def test_device_parser_rejects_duplicates_and_accepts_all_eight_cards(self):
        namespace = runpy.run_path(str(HELPER))
        parse_devices = namespace["parse_devices"]
        self.assertEqual(parse_devices("0,1,2,3,4,5,6,7"), list(range(8)))
        with self.assertRaises(ValueError):
            parse_devices("0,1,1")

    def test_partition_resolution_matches_official_model_layout(self):
        namespace = runpy.run_path(str(HELPER))
        resolve_partition = namespace["resolve_partition"]
        root = Path("/models/MiniMax-H3")
        self.assertEqual(resolve_partition(root, "fl2va"), root / "diffusers")
        self.assertEqual(resolve_partition(root, "ref2va"), root / "Ref2VA")

    def test_hygon_only_runtime_stages_cpu_weights_on_first_selected_gpu(self):
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
        report = configure_weight_staging(module, [3, 4], native_cpu_count=0)
        cpu_tensor = FakeTensor("cpu")
        converted = module.from_torch(cpu_tensor)

        self.assertEqual(report["mode"], "hygon-gpu-overlay")
        self.assertEqual(report["target_device"], "cuda:3")
        self.assertEqual(cpu_tensor.moved_to, "cuda:3")
        self.assertEqual(module.received, ["cuda"])
        self.assertEqual(converted, "cuda")

    def test_cpu_capable_runtime_keeps_native_weight_staging(self):
        namespace = runpy.run_path(str(HELPER))
        configure_weight_staging = namespace["configure_weight_staging"]

        class FakeInfiniCore:
            def __init__(self):
                self.received = []

            def from_torch(self, tensor):
                self.received.append(tensor.device.type)
                return tensor.device.type

        module = FakeInfiniCore()
        report = configure_weight_staging(module, [0], native_cpu_count=1)
        converted = module.from_torch(SimpleNamespace(device=SimpleNamespace(type="cpu")))

        self.assertEqual(report["mode"], "native-cpu")
        self.assertIsNone(report["target_device"])
        self.assertEqual(module.received, ["cpu"])
        self.assertEqual(converted, "cpu")


if __name__ == "__main__":
    unittest.main()
