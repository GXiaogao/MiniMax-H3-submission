import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "infinifabric-build-v1"
FORMAL_SHELL = ARTIFACT_DIR / "run_minimax_h3_fl2va_native_formal.sh"
README = ARTIFACT_DIR / "README.md"


class MiniMaxH3NativeFl2vaFormalContractTests(unittest.TestCase):
    def test_formal_shell_exists_and_fixes_the_official_ten_step_workload(self):
        self.assertTrue(FORMAL_SHELL.is_file(), "formal launcher has not been created")
        raw = FORMAL_SHELL.read_bytes()
        text = raw.decode("utf-8")
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        for token in (
            "set -Eeuo pipefail",
            'FORMAL_VERSION="1.0-minimax-h3-native-formal-validation"',
            'EXPECTED_INFINILM_BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"',
            'HELPER="${SCRIPT_DIR}/run_minimax_h3_fl2va_native_smoke.py"',
            'MODEL_PATH="/public/home/aclsylqidf/shared/MiniMax-H3"',
            'FIRST_FRAME="${EFFECTIVE_WORK_HOME}/inputs/fl2va/Golde33443.jpg"',
            'OUTPUT_DIR="${EFFECTIVE_WORK_HOME}/outputs/fl2va-native-formal"',
            'RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-run-minimax-h3-native-fl2va-formal-${RUN_ID}"',
            'result_json="${RUN_DIR}/minimax-h3-native-fl2va-formal.json"',
            'INFINILM_SUCCESS_MARKER="${EFFECTIVE_WORK_HOME}/state/infinilm-extension.success"',
            'if [[ "${marker_build_version}" != "${EXPECTED_INFINILM_BUILD_VERSION}" ]]',
            'timeout --signal=TERM --kill-after=30s 25m',
            '--run-profile=formal',
            '--text="A dog falls asleep"',
            '--gpus=0,1,2,3,4,5,6,7',
            '--num-frames=124',
            '--steps=10',
            '--seed=42',
            '--height=384',
            '--width=576',
        ):
            self.assertIn(token, text)
        self.assertNotIn('"$@"', text)
        self.assertNotIn("--legacy", text)
        self.assertNotIn("pip install", text)

    def test_formal_shell_rejects_overrides_and_preserves_failure_evidence(self):
        self.assertTrue(FORMAL_SHELL.is_file(), "formal launcher has not been created")
        text = FORMAL_SHELL.read_text(encoding="utf-8")
        for token in (
            'if [[ "$#" -ne 0 ]]',
            "formal validation accepts no program arguments",
            'if [[ "${HELPER_RC}" -eq 124 ]]',
            "formal FL2VA validation exceeded the 25-minute launcher limit",
            'if [[ "${HELPER_RC}" -eq 139 ]]',
            "native process terminated with SIGSEGV",
            'log_line "infinilm_extension_sha256=${extension_sha256}"',
            'log_line "SUMMARY: ${SUMMARY}"',
        ):
            self.assertIn(token, text)
        self.assertLess(
            text.index('require_file "${INFINILM_SUCCESS_MARKER}"'),
            text.index('timeout --signal=TERM --kill-after=30s 25m'),
        )

    def test_readme_documents_formal_upload_run_success_and_scope(self):
        text = README.read_text(encoding="utf-8")
        for token in (
            "## 第十阶段：原生 FL2VA 十步正式验证",
            "run_minimax_h3_fl2va_native_formal.sh",
            "run_minimax_h3_fl2va_native_smoke.py",
            "程序参数：留空",
            "页面运行时长：28 分钟",
            "steps=10",
            "NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED",
            "$WORK_HOME/outputs/fl2va-native-formal/",
            "$WORK_HOME/logs/infinifabric-run-minimax-h3-native-fl2va-formal-*",
            "单次运行不能替代官方相似度或正式性能验收",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
