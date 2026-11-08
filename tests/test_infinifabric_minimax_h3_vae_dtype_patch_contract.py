import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "artifacts/infinifabric-build-v1/prepare_minimax_h3_vae_dtype_probe.py"
SOURCE = b'''# original fixture; unrelated dtype must remain untouched
def move(pipe, execution_device, DTYPE):
    pipe.transformer.to(device=execution_device, dtype=DTYPE)
    pipe.vae.to(device=execution_device, dtype=DTYPE)
    pipe.audio_vae.to(device=execution_device, dtype=DTYPE)
'''
EXPECTED = b'''# original fixture; unrelated dtype must remain untouched
def move(pipe, execution_device, DTYPE):
    pipe.transformer.to(device=execution_device, dtype=DTYPE)
    pipe.vae.to(device=execution_device)
    pipe.audio_vae.to(device=execution_device)
'''


class VaeDtypePatchTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "VAE precision patch helper is missing")
        self.patch_example = runpy.run_path(str(HELPER))["patch_example"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.py"
        self.destination = self.root / "job-local/example/minimax_h3.py"
        self.source.write_bytes(SOURCE)

    def test_only_vae_casts_change_and_source_hash_still_matches(self):
        report = self.patch_example(self.source, self.destination)
        self.assertEqual(self.source.read_bytes(), SOURCE)
        self.assertEqual(self.destination.read_bytes(), EXPECTED)
        self.assertEqual(report["source_sha256"], hashlib.sha256(SOURCE).hexdigest())
        self.assertEqual(report["destination_sha256"], hashlib.sha256(EXPECTED).hexdigest())
        self.assertEqual(Path(report["source_path"]), self.source.resolve())
        self.assertEqual(Path(report["destination_path"]), self.destination.resolve())
        self.assertEqual(report["patch_profile"], "preserve-vae-fp32-modules-v1")

    def test_executed_copy_keeps_loaded_vae_precision_and_moves_all_components(self):
        self.patch_example(self.source, self.destination)
        class Component:
            def __init__(self):
                self.dtype = "float32"
                self.device = "cpu"
            def to(self, *, device, dtype=None):
                self.device = device
                if dtype is not None:
                    self.dtype = dtype
        pipe = SimpleNamespace(vae=Component(), audio_vae=Component(), transformer=Component())
        runpy.run_path(str(self.destination))["move"](pipe, "cuda:0", "bfloat16")
        self.assertEqual(pipe.vae.dtype, "float32")
        self.assertEqual(pipe.audio_vae.dtype, "float32")
        self.assertEqual(pipe.transformer.dtype, "bfloat16")
        self.assertEqual([pipe.vae.device, pipe.audio_vae.device, pipe.transformer.device],
                         ["cuda:0", "cuda:0", "cuda:0"])

    def test_missing_or_duplicated_cast_refuses_to_create_output(self):
        for name in (b"pipe.vae", b"pipe.audio_vae"):
            line = name + b".to(device=execution_device, dtype=DTYPE)"
            for altered in (SOURCE.replace(line, b"pass"), SOURCE + b"# " + line + b"\n"):
                with self.subTest(name=name, altered=altered):
                    self.source.write_bytes(altered)
                    with self.assertRaisesRegex(ValueError, "exactly once"):
                        self.patch_example(self.source, self.destination)
                    self.assertFalse(self.destination.exists())
                    self.assertEqual(self.source.read_bytes(), altered)

    def test_commented_out_cast_is_not_an_executable_patch_target(self):
        self.source.write_bytes(SOURCE.replace(b"    pipe.vae", b"    # pipe.vae"))
        with self.assertRaisesRegex(ValueError, "executable"):
            self.patch_example(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_same_file_and_existing_destination_cannot_overwrite_baseline(self):
        with self.assertRaisesRegex(ValueError, "different"):
            self.patch_example(self.source, self.source)
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"prior run evidence")
        with self.assertRaises(FileExistsError):
            self.patch_example(self.source, self.destination)
        self.assertEqual(self.source.read_bytes(), SOURCE)
        self.assertEqual(self.destination.read_bytes(), b"prior run evidence")

    def test_invalid_python_is_rejected_before_destination_write(self):
        self.source.write_bytes(SOURCE + b"this is invalid !!!\n")
        with self.assertRaises(SyntaxError):
            self.patch_example(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_actual_archived_example_changes_exactly_the_two_calls(self):
        archived = ROOT / "artifacts/infinifabric-migration-v3/stage/InfiniLM/examples/minimax_h3.py"
        before = archived.read_bytes()
        self.patch_example(archived, self.destination)
        changed = [(a, b) for a, b in zip(before.splitlines(), self.destination.read_bytes().splitlines())
                   if a != b]
        self.assertEqual(changed, [
            (b"        pipe.vae.to(device=execution_device, dtype=DTYPE)",
             b"        pipe.vae.to(device=execution_device)"),
            (b"        pipe.audio_vae.to(device=execution_device, dtype=DTYPE)",
             b"        pipe.audio_vae.to(device=execution_device)"),
        ])
        self.assertEqual(archived.read_bytes(), before)

    def test_cli_emits_metadata_and_failure_is_nonzero(self):
        cmd = [sys.executable, str(HELPER), "--source", str(self.source),
               "--destination", str(self.destination)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["destination_sha256"],
                         hashlib.sha256(self.destination.read_bytes()).hexdigest())
        again = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("ERROR", again.stderr)
        self.assertEqual(again.stdout, "")


if __name__ == "__main__":
    unittest.main()
