import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "artifacts"
    / "infinifabric-build-v1"
    / "download_infinilm_private_runtime_v1_4_wheels.ps1"
)


class WheelDownloadScriptContractTests(unittest.TestCase):
    def test_script_downloads_exact_additional_wheel_set_safely(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            r"E:\projects\minimax-h3-match\相关代码文件\new_wheels", text
        )
        self.assertIn(
            r"E:\projects\minimax-h3-match\相关代码文件\wheels", text
        )
        self.assertEqual(text.count("[pscustomobject]@{"), 18)
        for token in (
            "https://pypi.org/pypi/",
            "Invoke-RestMethod",
            "Invoke-WebRequest",
            "Get-FileHash",
            "SHA256",
            ".partial",
            "Offline",
            "Copy-Item",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("pip download", text)
        self.assertNotIn("pip install", text)


if __name__ == "__main__":
    unittest.main()
