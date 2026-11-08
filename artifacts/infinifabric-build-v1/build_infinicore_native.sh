#!/usr/bin/env bash

set -Eeuo pipefail
umask 0022

readonly BUILD_VERSION="2.6-fresh-afs-cross-job-readable"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
if [[ ! -f "${WORK_HOME}/setup_infinifabric_env.sh" ]]; then
    printf 'ERROR: environment file is missing: %s\n' \
        "${WORK_HOME}/setup_infinifabric_env.sh" >&2
    exit 1
fi

source "${WORK_HOME}/setup_infinifabric_env.sh"

PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"
readonly PERSISTENT_INFINI_ROOT

BUILD_JOBS="${INFINICORE_BUILD_JOBS:-16}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: INFINICORE_BUILD_JOBS must be a positive integer.\n' >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE_ARCHIVE="${SCRIPT_DIR}/InfiniCore-infinifabric-v3.tar.gz"
readonly SOURCE_ARCHIVE_SHA256="37414edbc25545063462a3cbe47811ca0a672378058d78b761ec36e6fc3f8811"
readonly REPO_CACHE="${PKG_DIR}/xmake-repo-cache.tar.gz"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly NATIVE_BACKUP_ROOT="${WORK_HOME}/backups/infinicore-native-rebuild-${RUN_ID}"
readonly RUN_DIR="${LOG_DIR}/infinifabric-build-infinicore-native-${RUN_ID}"
readonly RUN_LOG="${RUN_DIR}/run.log"
readonly SUMMARY="${RUN_DIR}/summary.txt"
readonly HARDWARE_JSON="${RUN_DIR}/hardware.json"
readonly ARTIFACT_HASHES="${RUN_DIR}/artifacts.sha256"
readonly CONFIG_SNAPSHOT="${RUN_DIR}/InfiniCore_xmake.conf"
readonly STATE_DIR="${WORK_HOME}/state"
readonly SUCCESS_MARKER="${STATE_DIR}/infinicore-native.success"

# InfiniFabric provides a writable, per-job local workspace. Keep xmake's
# mutable state and the build tree here; AFS is used for persistent inputs,
# logs, installed artifacts and the final success marker.
readonly LOCAL_ROOT="${PWD}/.infinicore-native-${UID}-$$"
readonly LOCAL_PROJECT_DIR="${LOCAL_ROOT}/InfiniCore"
export XMAKE_GLOBALDIR="${LOCAL_ROOT}/global"
export XMAKE_STORAGE_DIR="${XMAKE_GLOBALDIR}/.xmake"
export XMAKE_TMPDIR="${LOCAL_ROOT}/tmp"
export XMAKE_RAMDIR="${LOCAL_ROOT}/tmp"
export XMAKE_STATS=false
export XMAKE_THEME=plain

readonly HYGON_CUDA_ROOT="/opt/dtk/cuda/cuda"
export CUDA_HOME="${HYGON_CUDA_ROOT}"
export CUDA_PATH="${HYGON_CUDA_ROOT}"
export HYGON_CUDA_HOME="${HYGON_CUDA_ROOT}"
export PATH="${HYGON_CUDA_ROOT}/bin:${PATH}"
export PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${STATE_DIR}" "${XMAKE_STORAGE_DIR}" "${XMAKE_TMPDIR}"
: > "${RUN_LOG}"

CURRENT_STAGE="initialization"
BUILD_PASSED=0

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_error() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}" >&2
}

on_exit() {
    local exit_code="$1"
    local result="FAILED"
    if [[ "${BUILD_PASSED}" -eq 1 && "${exit_code}" -eq 0 ]]; then
        result="PASSED"
    fi
    local summary_tmp="${SUMMARY}.tmp.$$"
    {
        printf 'build_version=%s\n' "${BUILD_VERSION}"
        printf 'result=%s\n' "${result}"
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'last_stage=%s\n' "${CURRENT_STAGE}"
        printf 'run_dir=%s\n' "${RUN_DIR}"
    } > "${summary_tmp}"
    mv -f -- "${summary_tmp}" "${SUMMARY}"
    log_line "RESULT: ${result}; exit_code=${exit_code}; last_stage=${CURRENT_STAGE}"
    log_line "SUMMARY: ${SUMMARY}"
}

trap 'on_exit "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
    local name="$1"
    shift
    CURRENT_STAGE="${name}"
    local stage_log="${RUN_DIR}/${name}.log"
    log_line ""
    log_line "=== ${name} ==="
    set +e
    "$@" 2>&1 | tee -a "${stage_log}" "${RUN_LOG}"
    local stage_rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${stage_rc}" -ne 0 ]]; then
        log_error "ERROR: stage ${name} failed with exit_code=${stage_rc}"
        return "${stage_rc}"
    fi
}

check_hygon_cuda_root() {
    test -d "${HYGON_CUDA_ROOT}/include"
    test -d "${HYGON_CUDA_ROOT}/lib64"
    test -x "${HYGON_CUDA_ROOT}/bin/nvcc"
    "${HYGON_CUDA_ROOT}/bin/nvcc" --version
}

verify_input_archives() {
    local actual_source_sha256
    local repo_list="${XMAKE_TMPDIR}/repository-cache.list"
    local source_list="${XMAKE_TMPDIR}/source-archive.list"
    actual_source_sha256="$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')"
    printf 'expected_source_archive_sha256=%s\n' "${SOURCE_ARCHIVE_SHA256}"
    printf 'actual_source_archive_sha256=%s\n' "${actual_source_sha256}"
    [[ "${actual_source_sha256}" == "${SOURCE_ARCHIVE_SHA256}" ]]
    tar -tzf "${REPO_CACHE}" > "${repo_list}"
    grep -Fxq '.xmake/repositories/xmake-repo/packages/b/boost/xmake.lua' \
        "${repo_list}"
    tar -tzf "${SOURCE_ARCHIVE}" > "${source_list}"
    grep -Fxq 'InfiniCore/xmake.lua' "${source_list}"
}

prepare_local_xmake_state() {
    printf 'repository_cache_extract=begin\n'
    tar -xzf "${REPO_CACHE}" -C "${XMAKE_GLOBALDIR}"
    printf 'repository_cache_extract=complete\n'
    printf 'source_archive_extract=begin\n'
    tar -xzf "${SOURCE_ARCHIVE}" -C "${LOCAL_ROOT}"
    printf 'source_archive_extract=complete\n'
    test -f "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/b/boost/xmake.lua"
    test -f "${LOCAL_PROJECT_DIR}/xmake.lua"
    printf 'local_xmake_storage=%s\n' "${XMAKE_STORAGE_DIR}"
    printf 'local_project=%s\n' "${LOCAL_PROJECT_DIR}"
}

verify_local_source() {
    cd -- "${LOCAL_ROOT}"
    sha256sum --check --quiet InfiniCore.SHA256SUMS
}

patch_local_native_project() {
    local xmake_file="${LOCAL_PROJECT_DIR}/xmake.lua"
    local hygon_file="${LOCAL_PROJECT_DIR}/xmake/hygon.lua"
    local context_file="${LOCAL_PROJECT_DIR}/src/infinicore/context/context_impl.cc"
    local hygon_tmp="${hygon_file}.tmp.$$"
    local boost_line='add_requires("boost", {configs = {stacktrace = true}})'
    local pybind_line='add_requires("pybind11")'
    local hygon_cpu_line='    add_files("../src/infiniop/devices/cpu/*.cc", "../src/infiniop/ops/*/cpu/*.cc", "../src/infiniop/reduce/cpu/*.cc")'
    [[ "$(grep -Fxc "${boost_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${pybind_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${hygon_cpu_line}" "${hygon_file}")" -eq 1 ]]
    printf 'source_xmake_sha256='
    sha256sum "${xmake_file}" | awk '{print $1}'
    printf 'source_hygon_sha256='
    sha256sum "${hygon_file}" | awk '{print $1}'
    printf 'source_context_sha256='
    sha256sum "${context_file}" | awk '{print $1}'
    sed -i \
        -e '/^add_requires("boost", {configs = {stacktrace = true}})$/d' \
        -e '/^add_requires("pybind11")$/d' \
        "${LOCAL_PROJECT_DIR}/xmake.lua"
    grep -Fvx "${hygon_cpu_line}" "${hygon_file}" > "${hygon_tmp}"
    mv -f -- "${hygon_tmp}" "${hygon_file}"
    "${PYTHON}" -c '
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
cpu_init_before = """    runtime_table_[0].resize(device_counter[0]);
    runtime_table_[0][0] = std::unique_ptr<Runtime>(new Runtime(Device(Device::Type::CPU, 0)));
"""
cpu_init_after = """    if (device_counter[0] > 0) {
        runtime_table_[0].resize(device_counter[0]);
        runtime_table_[0][0] = std::unique_ptr<Runtime>(new Runtime(Device(Device::Type::CPU, 0)));
    }
"""
cpu_fallback_before = """    if (current_runtime_ == nullptr) {
        current_runtime_ = runtime_table_[0][0].get();
    }
"""
cpu_fallback_after = """    if (current_runtime_ == nullptr && !runtime_table_[0].empty()) {
        current_runtime_ = runtime_table_[0][0].get();
    }
"""
if text.count(cpu_init_before) != 1:
    raise SystemExit("unexpected CPU runtime initialization block")
if text.count(cpu_fallback_before) != 1:
    raise SystemExit("unexpected CPU fallback block")
text = text.replace(cpu_init_before, cpu_init_after, 1)
text = text.replace(cpu_fallback_before, cpu_fallback_after, 1)
path.write_text(text, encoding="utf-8", newline="\n")
' "${context_file}"
    ! grep -Fq "${boost_line}" "${xmake_file}"
    ! grep -Fq "${pybind_line}" "${xmake_file}"
    ! grep -Fq "${hygon_cpu_line}" "${hygon_file}"
    grep -Fq 'if (device_counter[0] > 0) {' "${context_file}"
    grep -Fq 'if (current_runtime_ == nullptr && !runtime_table_[0].empty()) {' \
        "${context_file}"
    printf 'local_xmake_sha256='
    sha256sum "${xmake_file}" | awk '{print $1}'
    printf 'local_hygon_sha256='
    sha256sum "${hygon_file}" | awk '{print $1}'
    printf 'local_context_sha256='
    sha256sum "${context_file}" | awk '{print $1}'
    printf 'patch_profile=hygon-only-context-cpu-guard\n'
    printf 'AFS source was not modified; Hygon-only pruning and context guard applied to local copy.\n'
}

install_native_stack() {
    xmake install infinirt
    xmake install infiniop
    xmake install infiniccl
    xmake install infinicore_cpp_api
}

prepare_fresh_persistent_root() {
    if [[ -e "${PERSISTENT_INFINI_ROOT}" ]]; then
        mkdir -p "${NATIVE_BACKUP_ROOT}"
        mv -- "${PERSISTENT_INFINI_ROOT}" "${NATIVE_BACKUP_ROOT}/previous-infini"
        printf 'previous_infini_backup=%s\n' \
            "${NATIVE_BACKUP_ROOT}/previous-infini"
    fi
    mkdir -p "${PERSISTENT_INFINI_ROOT}"
    printf 'fresh_infini_root=%s\n' "${PERSISTENT_INFINI_ROOT}"
}

publish_cross_job_readable_tree() {
    # This tree was created by the current job. Publish its directories and
    # files before the job UID disappears; later jobs must only read it.
    find "${PERSISTENT_INFINI_ROOT}" -type d -exec chmod o+rx -- {} +
    find "${PERSISTENT_INFINI_ROOT}" -type f -exec chmod o+r -- {} +
    if find "${PERSISTENT_INFINI_ROOT}" -type d ! -perm -0005 -print -quit \
        | grep -q .; then
        printf 'ERROR: directory without other read/traverse permission remains.\n' >&2
        return 1
    fi
    if find "${PERSISTENT_INFINI_ROOT}" -type f ! -perm -0004 -print -quit \
        | grep -q .; then
        printf 'ERROR: file without other read permission remains.\n' >&2
        return 1
    fi
    if find "${PERSISTENT_INFINI_ROOT}" -perm -0002 -print -quit \
        | grep -q .; then
        printf 'ERROR: world-writable artifact detected.\n' >&2
        return 1
    fi
    stat -Lc '%A mode=%a uid=%u gid=%g %n' \
        "${PERSISTENT_INFINI_ROOT}" \
        "${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so" \
        "${PERSISTENT_INFINI_ROOT}/lib/libinfiniop.so" \
        "${PERSISTENT_INFINI_ROOT}/lib/libinfinirt.so" \
        "${PERSISTENT_INFINI_ROOT}/lib/libinfiniccl.so"
}

log_line "build_version=${BUILD_VERSION}"
log_line "time=$(date -Is)"
log_line "host=$(hostname)"
log_line "uid=$(id -u)"
log_line "work_home=${WORK_HOME}"
log_line "source_archive=${SOURCE_ARCHIVE}"
log_line "repository_cache=${REPO_CACHE}"
log_line "local_project=${LOCAL_PROJECT_DIR}"
log_line "persistent_infini_root=${PERSISTENT_INFINI_ROOT}"
log_line "local_xmake_storage=${XMAKE_STORAGE_DIR}"
log_line "hygon_cuda_root=${HYGON_CUDA_ROOT}"
log_line "backend_profile=hygon-only-pruned-cpu-sources-context-cpu-guard"
log_line "build_jobs=${BUILD_JOBS}"

[[ -f "${SOURCE_ARCHIVE}" ]] || {
    log_error "ERROR: source archive is missing beside this script: ${SOURCE_ARCHIVE}"
    exit 1
}
[[ -f "${REPO_CACHE}" ]] || {
    log_error "ERROR: xmake repository cache is missing: ${REPO_CACHE}"
    exit 1
}
[[ -f "${SCRIPT_DIR}/inspect_hygon_arch.py" ]] || {
    log_error "ERROR: inspect_hygon_arch.py is missing beside this script."
    exit 1
}
CURRENT_STAGE="invalidate-stale-success-marker"
if [[ -f "${SUCCESS_MARKER}" ]]; then
    mv -- "${SUCCESS_MARKER}" "${RUN_DIR}/previous-infinicore-native.success"
    log_line "previous_success_marker=${RUN_DIR}/previous-infinicore-native.success"
fi
run_stage "input-archive-check" verify_input_archives
run_stage "local-xmake-state" prepare_local_xmake_state
run_stage "source-check" verify_local_source

run_stage "hardware-probe" \
    python3 "${SCRIPT_DIR}/inspect_hygon_arch.py" \
        --json "${HARDWARE_JSON}" \
        --print-arch
HYGON_ARCH="$(tail -n 1 "${RUN_DIR}/hardware-probe.log")"
if [[ ! "${HYGON_ARCH}" =~ ^gfx(906|936)$ ]]; then
    log_error "ERROR: invalid normalized architecture from probe: ${HYGON_ARCH@Q}"
    exit 1
fi
export HYGON_ARCH
log_line "hygon_arch=${HYGON_ARCH}"

run_stage "hygon-cuda-root" check_hygon_cuda_root
run_stage "local-source-patch" patch_local_native_project
run_stage "xmake-global-config" \
    xmake g --network=private --pkg_searchdirs="${PKG_DIR}"

cd -- "${LOCAL_PROJECT_DIR}"
run_stage "xmake-config" \
    xmake f -y -m release \
        --hygon-dcu=y \
        --hygon-arch="${HYGON_ARCH}" \
        --cuda="${HYGON_CUDA_ROOT}" \
        --cpu=n \
        --omp=n \
        --aten=y \
        --ccl=y \
        --graph=n \
        --infiniops=n \
        --ninetoothed=n \
        --flash-attn=

run_stage "config-snapshot" cp \
    "${LOCAL_PROJECT_DIR}/.xmake/linux/x86_64/xmake.conf" \
    "${CONFIG_SNAPSHOT}"

run_stage "xmake-build" \
    xmake build -j "${BUILD_JOBS}" infinicore_cpp_api

run_stage "fresh-persistent-root" prepare_fresh_persistent_root
run_stage "xmake-install" install_native_stack
run_stage "cross-job-readability" publish_cross_job_readable_tree

CURRENT_STAGE="artifact-validation"
readonly HEADER="${PERSISTENT_INFINI_ROOT}/include/infinicore/tensor.hpp"
readonly LIBRARY="${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so"
[[ -f "${HEADER}" ]] || {
    log_error "ERROR: installed header is missing: ${HEADER}"
    exit 1
}
[[ -f "${LIBRARY}" ]] || {
    log_error "ERROR: installed library is missing: ${LIBRARY}"
    exit 1
}
[[ -f "${CONFIG_SNAPSHOT}" ]] || {
    log_error "ERROR: persisted xmake config snapshot is missing: ${CONFIG_SNAPSHOT}"
    exit 1
}

run_stage "file" file "${LIBRARY}"

CURRENT_STAGE="torch-runtime"
TORCH_LIB_DIR="$("${PYTHON}" -c \
    'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
readonly TORCH_LIB_DIR
for torch_library in libtorch.so libtorch_cpu.so libc10.so libc10_hip.so; do
    [[ -f "${TORCH_LIB_DIR}/${torch_library}" ]] || {
        log_error "ERROR: PyTorch runtime library is missing: ${TORCH_LIB_DIR}/${torch_library}"
        exit 1
    }
done
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${PERSISTENT_INFINI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
log_line "torch_lib_dir=${TORCH_LIB_DIR}"

run_stage "ldd" ldd "${LIBRARY}"
if grep -Fq "not found" "${RUN_DIR}/ldd.log"; then
    log_error "ERROR: unresolved dynamic library dependency detected."
    exit 1
fi

CURRENT_STAGE="artifact-hashing"
sha256sum "${HEADER}" "${LIBRARY}" "${CONFIG_SNAPSHOT}" > "${ARTIFACT_HASHES}"
cat "${ARTIFACT_HASHES}" | tee -a "${RUN_LOG}"

CURRENT_STAGE="success-marker"
SUCCESS_MARKER_TMP="${SUCCESS_MARKER}.tmp.$$"
{
    printf 'build_version=%s\n' "${BUILD_VERSION}"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'hygon_arch=%s\n' "${HYGON_ARCH}"
    printf 'cuda_root=%s\n' "${HYGON_CUDA_ROOT}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'artifacts_sha256=%s\n' "${ARTIFACT_HASHES}"
} > "${SUCCESS_MARKER_TMP}"
mv -f -- "${SUCCESS_MARKER_TMP}" "${SUCCESS_MARKER}"

BUILD_PASSED=1
CURRENT_STAGE="complete"
log_line "InfiniCore native build: PASSED"
log_line "success_marker=${SUCCESS_MARKER}"
log_line "run_dir=${RUN_DIR}"
