import unittest
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "build_infinilm_extension.sh"
)


class InfiniLMExtensionBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT_PATH.is_file():
            raise AssertionError(f"build script is missing: {SCRIPT_PATH}")
        cls.raw = SCRIPT_PATH.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_is_lf_only_strict_bash_without_dev_fd(self):
        self.assertTrue(self.raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("set -Eeuo pipefail", self.text)
        self.assertNotIn("/dev/fd", self.text)

    def test_requires_verified_infinicore_python_stage(self):
        self.assertIn("infinicore-python.success", self.text)
        self.assertIn('[[ -f "${INFINICORE_SUCCESS_MARKER}" ]]', self.text)

    def test_pins_and_verifies_infinilm_source_archive(self):
        self.assertIn("InfiniLM-infinifabric-v3.tar.gz", self.text)
        self.assertIn(
            "4c62340aa0aef5fec9e5a7f57ea7753eb6666e070b3a7e66aed5d722baad39e6",
            self.text,
        )
        self.assertIn("InfiniLM.SHA256SUMS", self.text)
        self.assertIn("sha256sum --check --quiet", self.text)

    def test_accepts_source_archive_from_personal_afs_packages(self):
        self.assertIn('SOURCE_ARCHIVE_NAME="InfiniLM-infinifabric-v3.tar.gz"', self.text)
        self.assertIn('PROJECT_SOURCE_ARCHIVE="${SCRIPT_DIR}/${SOURCE_ARCHIVE_NAME}"', self.text)
        self.assertIn('AFS_SOURCE_ARCHIVE="${PKG_DIR}/${SOURCE_ARCHIVE_NAME}"', self.text)
        self.assertIn('elif [[ -f "${AFS_SOURCE_ARCHIVE}" ]]', self.text)
        self.assertIn('SOURCE_ARCHIVE="${AFS_SOURCE_ARCHIVE}"', self.text)
        self.assertIn('log_line "source_archive=${SOURCE_ARCHIVE}"', self.text)

    def test_uses_job_local_build_and_xmake_state(self):
        self.assertIn('LOCAL_ROOT="${PWD}/.infinilm-extension-${UID}-$$"', self.text)
        self.assertIn('LOCAL_PROJECT_DIR="${LOCAL_ROOT}/InfiniLM"', self.text)
        self.assertIn('export XMAKE_GLOBALDIR="${LOCAL_ROOT}/global"', self.text)
        self.assertIn('export XMAKE_STORAGE_DIR="${XMAKE_GLOBALDIR}/.xmake"', self.text)

    def test_uses_verified_vendored_pybind_and_active_python_headers(self):
        self.assertIn("pybind11-v3.1.0.zip", self.text)
        self.assertIn(
            "affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536",
            self.text,
        )
        self.assertIn('-m zipfile -e "${PYBIND_ARCHIVE}"', self.text)
        self.assertIn('sysconfig.get_paths()["include"]', self.text)
        self.assertIn('${PYTHON_INCLUDE_DIR}/Python.h', self.text)
        self.assertIn('add_requires("pybind11")', self.text)
        self.assertIn('add_packages("pybind11")', self.text)
        self.assertIn('add_includedirs(\\"${LOCAL_PYBIND_INCLUDE}\\"', self.text)

    def test_matches_torch_and_infinicore_cxx11_abi(self):
        self.assertIn("torch._C._GLIBCXX_USE_CXX11_ABI", self.text)
        self.assertIn('export INFINILM_CXX11_ABI="${TORCH_CXX11_ABI}"', self.text)
        self.assertIn('--cxx11-abi="${TORCH_CXX11_ABI}"', self.text)

    def test_builds_only_infinilm_extension(self):
        self.assertIn('xmake build -j "${BUILD_JOBS}" _infinilm', self.text)
        self.assertNotIn("xmake install", self.text)
        self.assertIn("_infinilm${EXT_SUFFIX}", self.text)

    def test_patches_qwen_metadata_staging_for_hygon_only_infinicore(self):
        self.assertIn('BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"', self.text)
        self.assertIn("patch_hygon_fl2va_no_cpu_runtime()", self.text)
        self.assertIn(
            "csrc/models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp",
            self.text,
        )
        self.assertIn("csrc/models/qwen3_5/qwen3_5_vision.cpp", self.text)
        self.assertIn("#include <infinicore/context/context.hpp>", self.text)
        self.assertIn("infinicore::context::memcpyD2H", self.text)
        self.assertIn("infinicore::context::memcpyH2D", self.text)
        self.assertIn("position_ids_storage", self.text)
        self.assertIn("csrc/engine/infer_engine.cpp", self.text)
        self.assertIn("csrc/engine/rank_worker.cpp", self.text)
        self.assertIn(
            "csrc/layers/attention/backends/static_attn.cpp",
            self.text,
        )
        self.assertIn("read_i32_scalar", self.text)
        self.assertIn("patch_profile=hygon-fl2va-no-cpu-runtime", self.text)
        self.assertIn(
            'run_stage "hygon-fl2va-no-cpu-runtime-patch" patch_hygon_fl2va_no_cpu_runtime',
            self.text,
        )

        source_check = self.text.index('run_stage "source-check"')
        metadata_patch = self.text.index('run_stage "hygon-fl2va-no-cpu-runtime-patch"')
        xmake_config = self.text.index('run_stage "xmake-config"')
        self.assertLess(source_check, metadata_patch)
        self.assertLess(metadata_patch, xmake_config)

    def test_embedded_hygon_metadata_patch_applies_to_pinned_source(self):
        function_start = self.text.index("patch_hygon_fl2va_no_cpu_runtime()")
        code_marker = '    "${PYTHON}" -c \'\n'
        code_start = self.text.index(code_marker, function_start) + len(code_marker)
        code_end = self.text.index('\n\' "${LOCAL_PROJECT_DIR}"', code_start)
        embedded_code = self.text[code_start:code_end]
        self.assertNotIn("'", embedded_code)

        source_root = (
            SCRIPT_PATH.parents[1]
            / "infinifabric-migration-v3"
            / "verification-extract"
            / "InfiniLM"
        )
        relative_files = (
            Path("csrc/models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp"),
            Path("csrc/models/qwen3_5/qwen3_5_vision.cpp"),
            Path("csrc/engine/infer_engine.cpp"),
            Path("csrc/engine/rank_worker.cpp"),
            Path("csrc/layers/attention/backends/static_attn.cpp"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            for relative_file in relative_files:
                destination = temporary_root / relative_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative_file, destination)

            with mock.patch.object(
                sys,
                "argv",
                ["embedded-hygon-metadata-patch", str(temporary_root)],
            ):
                exec(compile(embedded_code, "embedded-hygon-metadata-patch", "exec"))

            qwen_vl = (temporary_root / relative_files[0]).read_text("utf-8")
            qwen_vision = (temporary_root / relative_files[1]).read_text("utf-8")
            infer_engine = (temporary_root / relative_files[2]).read_text("utf-8")
            rank_worker = (temporary_root / relative_files[3]).read_text("utf-8")
            static_attn = (temporary_root / relative_files[4]).read_text("utf-8")
            self.assertNotIn(
                "tensor->to(infinicore::Device::cpu())",
                qwen_vl,
            )
            self.assertNotIn(
                "cpu_tensor = tensor->to(infinicore::Device::cpu())",
                qwen_vision,
            )
            self.assertIn("infinicore::context::memcpyD2H", qwen_vl)
            self.assertIn("infinicore::context::memcpyD2H", qwen_vision)
            self.assertIn("infinicore::context::memcpyH2D", qwen_vision)
            self.assertIn("position_ids_storage", qwen_vision)
            self.assertNotIn(
                "cpu_offsets = cpu_offsets->to(infinicore::Device::cpu())",
                infer_engine,
            )
            self.assertIn("infinicore::context::memcpyD2H", infer_engine)
            self.assertNotIn(
                "{0}, infinicore::DataType::I64,\n                                infinicore::Device::cpu()",
                rank_worker,
            )
            self.assertIn(
                "{0}, infinicore::DataType::I64, rank_info_.device",
                rank_worker,
            )
            self.assertNotIn(
                "total_sequence_lengths.value()->to(infinicore::Device::cpu())",
                static_attn,
            )
            self.assertNotIn(
                "past_sequence_lengths->to(infinicore::Device::cpu())",
                static_attn,
            )
            self.assertIn("read_i32_scalar", static_attn)

    def test_validates_and_atomically_promotes_extension(self):
        self.assertIn('run_stage "ldd" ldd "${BUILT_EXTENSION}"', self.text)
        self.assertIn('grep -Fq "not found"', self.text)
        self.assertIn('mv -f -- "${EXTENSION_TMP}" "${DESTINATION_EXTENSION}"', self.text)
        self.assertIn("infinilm-extension.success", self.text)

    def test_requires_cross_job_readable_inputs_without_rewriting_old_modes(self):
        self.assertIn("input-readability", self.text)
        self.assertIn('verify_inputs_readable()', self.text)
        self.assertIn('[[ -r "${input_path}" ]]', self.text)
        self.assertNotIn('chmod o+r -- "$@"', self.text)

    def test_publishes_new_extension_before_promotion(self):
        self.assertIn('chmod o+r -- "${BUILT_EXTENSION}"', self.text)
        self.assertIn('[[ -r "${DESTINATION_EXTENSION}" ]]', self.text)
        self.assertIn('if [[ -r "${DESTINATION_EXTENSION}" ]]', self.text)
        self.assertNotIn("chmod 777", self.text)

    def test_smoke_imports_extension_without_full_python_dependency_stack(self):
        self.assertIn("import _infinilm", self.text)
        self.assertIn('hasattr(_infinilm, "InferEngine")', self.text)
        self.assertIn('hasattr(_infinilm, "DistConfig")', self.text)
        self.assertNotIn("import infinilm", self.text)

    def test_avoids_shared_python_mutation_and_destructive_operations(self):
        for token in ("pip install", "sudo", "/usr/local", "rm -rf", "rm -r"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.text)

    def test_records_failure_stage_and_result(self):
        self.assertIn("CURRENT_STAGE", self.text)
        self.assertIn("exit_code", self.text)
        self.assertIn("InfiniLM extension build: PASSED", self.text)
        self.assertIn("RESULT:", self.text)


if __name__ == "__main__":
    unittest.main()
