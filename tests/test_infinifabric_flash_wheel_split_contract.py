import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "artifacts"
    / "infinifabric-build-v1"
    / "split_flash_attention_wheel.ps1"
)


class FlashWheelSplitContractTests(unittest.TestCase):
    def test_splitter_streams_upload_sized_parts_and_records_full_hash(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "ChunkMiB = 100",
            "FileStream",
            "SHA256SUMS.flash-parts",
            "flash-wheel-manifest.json",
            "full_sha256",
            "part_sha256",
            ".part-{0:D3}",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("ReadAllBytes", text)
        self.assertNotIn("Remove-Item -Recurse", text)


if __name__ == "__main__":
    unittest.main()
