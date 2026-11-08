import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "artifacts"
    / "infinifabric-build-v1"
    / "download_minimax_h3_hygon_runtime_inputs.ps1"
)


class HygonRuntimeInputDownloadContractTests(unittest.TestCase):
    def test_script_fetches_exact_offline_inputs_without_installing(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            r"E:\projects\minimax-h3-match\相关代码文件\wheel",
            "einops-0.8.2-py3-none-any.whl",
            "pytest-9.0.3-py3-none-any.whl",
            "iniconfig-2.0.0-py3-none-any.whl",
            "packaging-24.2-py3-none-any.whl",
            "pluggy-1.5.0-py3-none-any.whl",
            "pygments-2.21.0-py3-none-any.whl",
            "triton-3.5.1+das.opt1.dtk2604.torch290",
            "flash-attn==2.6.1+das.opt1.dtk2604.torch290",
            "https://pypi.sourcefind.cn/release/dtk/",
            "vision/archive/refs/tags/v0.24.0.tar.gz",
            "Invoke-WebRequest",
            "Get-FileHash",
            "SHA256SUMS.hygon-runtime-inputs",
            ".partial",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertIn("pip download", text)
        self.assertNotIn("pip install", text)
        self.assertIn("input_count=9", text)
        self.assertNotIn("Remove-Item -Recurse", text)


if __name__ == "__main__":
    unittest.main()
