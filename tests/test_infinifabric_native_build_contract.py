import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "infinifabric-build-v1"
    / "build_infinicore_native.sh"
)


def read_script() -> tuple[bytes, str]:
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"production script is missing: {SCRIPT_PATH}")
    raw = SCRIPT_PATH.read_bytes()
    return raw, raw.decode("utf-8")


class InfiniCoreNativeBuildContractTests(unittest.TestCase):
    def test_is_lf_only_bash(self):
        raw, text = read_script()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("set -Eeuo pipefail", text)

    def test_logging_does_not_require_dev_fd_process_substitution(self):
        _, text = read_script()
        self.assertNotIn("exec > >(", text)
        self.assertNotIn("/dev/fd", text)
        self.assertIn("log_line()", text)
        self.assertIn('tee -a "${RUN_LOG}"', text)

    def test_loads_persistent_environment_and_validates_jobs(self):
        _, text = read_script()
        self.assertIn('source "${WORK_HOME}/setup_infinifabric_env.sh"', text)
        self.assertIn('INFINICORE_BUILD_JOBS:-16', text)
        self.assertRegex(text, r"\[\[.*BUILD_JOBS.*=~.*\^\[1-9\]")

    def test_verifies_archived_source_and_probes_arch_before_configuration(self):
        _, text = read_script()
        checksum = text.index("SOURCE_ARCHIVE_SHA256")
        probe = text.index("inspect_hygon_arch.py")
        configure = text.index("xmake f")
        self.assertLess(checksum, probe)
        self.assertLess(probe, configure)
        self.assertIn("sha256sum --check", text)
        self.assertIn('sha256sum "${SOURCE_ARCHIVE}"', text)
        self.assertIn("--print-arch", text)

    def test_configures_explicit_baseline_features(self):
        _, text = read_script()
        required = (
            "--hygon-dcu=y",
            '--hygon-arch="${HYGON_ARCH}"',
            '--cuda="${HYGON_CUDA_ROOT}"',
            "--cpu=n",
            "--omp=n",
            "--aten=y",
            "--ccl=y",
            "--graph=n",
            "--infiniops=n",
            "--ninetoothed=n",
            "--flash-attn=",
            "infinicore_cpp_api",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_extracts_compressed_inputs_into_local_build_tree(self):
        _, text = read_script()
        self.assertIn('LOCAL_ROOT="${PWD}/.infinicore-native-${UID}-$$"', text)
        self.assertIn('export XMAKE_GLOBALDIR="${LOCAL_ROOT}/global"', text)
        self.assertIn('export XMAKE_STORAGE_DIR="${XMAKE_GLOBALDIR}/.xmake"', text)
        self.assertIn('REPO_CACHE="${PKG_DIR}/xmake-repo-cache.tar.gz"', text)
        self.assertIn(
            'SOURCE_ARCHIVE="${SCRIPT_DIR}/InfiniCore-infinifabric-v3.tar.gz"',
            text,
        )
        self.assertIn('tar -xzf "${REPO_CACHE}" -C "${XMAKE_GLOBALDIR}"', text)
        self.assertIn('tar -xzf "${SOURCE_ARCHIVE}" -C "${LOCAL_ROOT}"', text)
        self.assertNotIn('cp -a "${PERSISTENT_XMAKE_STORAGE}/repositories"', text)
        self.assertNotIn('cp -a "${PROJECT_DIR}" "${LOCAL_PROJECT_DIR}"', text)
        self.assertIn('cd -- "${LOCAL_PROJECT_DIR}"', text)

    def test_prunes_only_unused_dependencies_in_local_copy(self):
        _, text = read_script()
        self.assertIn('patch_local_native_project()', text)
        self.assertIn('add_requires("boost", {configs = {stacktrace = true}})', text)
        self.assertIn('add_requires("pybind11")', text)
        self.assertIn('sed -i', text)
        self.assertIn('"${LOCAL_PROJECT_DIR}/xmake.lua"', text)
        self.assertIn('local hygon_file="${LOCAL_PROJECT_DIR}/xmake/hygon.lua"', text)
        self.assertIn('../src/infiniop/devices/cpu/*.cc', text)
        self.assertIn('grep -Fvx "${hygon_cpu_line}"', text)
        self.assertIn('mv -f -- "${hygon_tmp}" "${hygon_file}"', text)

    def test_guards_cpu_runtime_when_cpu_backend_is_disabled(self):
        _, text = read_script()
        self.assertIn(
            'context_file="${LOCAL_PROJECT_DIR}/src/infinicore/context/context_impl.cc"',
            text,
        )
        self.assertIn('if (device_counter[0] > 0) {', text)
        self.assertIn(
            'if (current_runtime_ == nullptr && !runtime_table_[0].empty()) {',
            text,
        )
        self.assertIn('hygon-only-context-cpu-guard', text)

    def test_records_local_extraction_progress_and_pins_source_archive(self):
        _, text = read_script()
        self.assertIn(
            'SOURCE_ARCHIVE_SHA256="37414edbc25545063462a3cbe47811ca0a672378058d78b761ec36e6fc3f8811"',
            text,
        )
        self.assertIn("repository_cache_extract=begin", text)
        self.assertIn("repository_cache_extract=complete", text)
        self.assertIn("source_archive_extract=begin", text)
        self.assertIn("source_archive_extract=complete", text)

    def test_uses_verified_dtk_cuda_compatibility_root_and_runtime_arch(self):
        _, text = read_script()
        self.assertIn('HYGON_CUDA_ROOT="/opt/dtk/cuda/cuda"', text)
        self.assertIn('export CUDA_HOME="${HYGON_CUDA_ROOT}"', text)
        self.assertIn('export CUDA_PATH="${HYGON_CUDA_ROOT}"', text)
        self.assertIn('export HYGON_CUDA_HOME="${HYGON_CUDA_ROOT}"', text)
        self.assertIn('export HYGON_ARCH', text)
        self.assertIn('"${HYGON_CUDA_ROOT}/bin/nvcc" --version', text)

    def test_persists_config_snapshot_but_installs_to_afs_infini_root(self):
        _, text = read_script()
        self.assertIn('PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"', text)
        self.assertIn('CONFIG_SNAPSHOT="${RUN_DIR}/InfiniCore_xmake.conf"', text)
        self.assertIn('run_stage "config-snapshot" cp', text)
        self.assertIn(
            '"${LOCAL_PROJECT_DIR}/.xmake/linux/x86_64/xmake.conf"', text
        )
        self.assertIn('"${CONFIG_SNAPSHOT}"', text)
        self.assertIn('${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so', text)

    def test_replaces_old_native_tree_with_a_recoverable_fresh_install(self):
        _, text = read_script()
        self.assertIn('NATIVE_BACKUP_ROOT="${WORK_HOME}/backups/infinicore-native-rebuild-${RUN_ID}"', text)
        self.assertIn('prepare_fresh_persistent_root()', text)
        self.assertIn('mv -- "${PERSISTENT_INFINI_ROOT}" "${NATIVE_BACKUP_ROOT}/previous-infini"', text)
        self.assertIn('mkdir -p "${PERSISTENT_INFINI_ROOT}"', text)
        prepare = text.index('run_stage "fresh-persistent-root"')
        install = text.index('run_stage "xmake-install"')
        self.assertLess(prepare, install)

    def test_publishes_new_native_tree_for_future_job_uids(self):
        _, text = read_script()
        self.assertIn('publish_cross_job_readable_tree()', text)
        self.assertIn('find "${PERSISTENT_INFINI_ROOT}" -type d -exec chmod o+rx -- {} +', text)
        self.assertIn('find "${PERSISTENT_INFINI_ROOT}" -type f -exec chmod o+r -- {} +', text)
        self.assertIn('run_stage "cross-job-readability" publish_cross_job_readable_tree', text)
        publish = text.index('run_stage "cross-job-readability"')
        validate = text.index('CURRENT_STAGE="artifact-validation"')
        self.assertLess(publish, validate)
        self.assertNotIn('chmod o+w', text)
        self.assertNotIn('chmod 777', text)

    def test_avoids_shared_install_and_destructive_operations(self):
        _, text = read_script()
        forbidden = (
            "pip install",
            "sudo",
            "xmake clean",
            "/usr/local",
            "rm -rf",
            "rm -r",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

    def test_validates_artifacts_before_atomic_success_marker(self):
        _, text = read_script()
        self.assertIn(
            '${PERSISTENT_INFINI_ROOT}/include/infinicore/tensor.hpp', text
        )
        self.assertIn(
            '${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so', text
        )
        self.assertIn("ldd", text)
        self.assertIn("not found", text)
        artifact_hash = text.index("artifacts.sha256")
        marker = text.index("infinicore-native.success")
        self.assertLess(artifact_hash, marker)
        self.assertIn("mv -f", text)

    def test_invalidates_stale_native_marker_before_rebuild(self):
        _, text = read_script()
        self.assertIn('CURRENT_STAGE="invalidate-stale-success-marker"', text)
        self.assertIn('previous-infinicore-native.success', text)
        invalidate = text.index('CURRENT_STAGE="invalidate-stale-success-marker"')
        build = text.index('run_stage "xmake-build"')
        self.assertLess(invalidate, build)

    def test_adds_pytorch_library_directory_before_ldd(self):
        _, text = read_script()
        self.assertIn("TORCH_LIB_DIR", text)
        self.assertIn("import torch", text)
        self.assertIn("libtorch.so", text)
        self.assertIn("libtorch_cpu.so", text)
        self.assertIn("libc10.so", text)
        self.assertIn("libc10_hip.so", text)
        torch_path = text.index('export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:')
        ldd = text.index('run_stage "ldd"')
        self.assertLess(torch_path, ldd)

    def test_records_failure_stage_and_exit_code(self):
        _, text = read_script()
        self.assertIn("trap", text)
        self.assertIn("CURRENT_STAGE", text)
        self.assertIn("exit_code", text)


if __name__ == "__main__":
    unittest.main()
