#!/usr/bin/env bash

set -Eeuo pipefail
umask 0022

readonly BUILD_VERSION="1.3-afs-cross-job-readable"

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

BUILD_JOBS="${INFINICORE_PYTHON_BUILD_JOBS:-16}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: INFINICORE_PYTHON_BUILD_JOBS must be a positive integer.\n' >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE_ARCHIVE="${SCRIPT_DIR}/InfiniCore-infinifabric-v3.tar.gz"
readonly SOURCE_ARCHIVE_SHA256="37414edbc25545063462a3cbe47811ca0a672378058d78b761ec36e6fc3f8811"
readonly REPO_CACHE="${PKG_DIR}/xmake-repo-cache.tar.gz"
readonly PYBIND_ARCHIVE="${PKG_DIR}/pybind11-v3.1.0.zip"
readonly PYBIND_SHA256="affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536"
readonly NATIVE_SUCCESS_MARKER="${WORK_HOME}/state/infinicore-native.success"
readonly PERSISTENT_PROJECT_DIR="${WORK_HOME}/Projects/InfiniCore"
readonly PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_DIR="${LOG_DIR}/infinifabric-build-infinicore-python-${RUN_ID}"
readonly RUN_LOG="${RUN_DIR}/run.log"
readonly SUMMARY="${RUN_DIR}/summary.txt"
readonly HARDWARE_JSON="${RUN_DIR}/hardware.json"
readonly ARTIFACT_HASHES="${RUN_DIR}/artifacts.sha256"
readonly CONFIG_SNAPSHOT="${RUN_DIR}/InfiniCore_python_xmake.conf"
readonly STATE_DIR="${WORK_HOME}/state"
readonly LOCAL_ROOT="${PWD}/.infinicore-python-${UID}-$$"
readonly LOCAL_PROJECT_DIR="${LOCAL_ROOT}/InfiniCore"
readonly LOCAL_PYBIND_ROOT="${LOCAL_ROOT}/pybind11-src/pybind11-3.1.0"
readonly LOCAL_PYBIND_INCLUDE="${LOCAL_PYBIND_ROOT}/include"

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
PYTHON_INCLUDE_DIR="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
readonly PYTHON_INCLUDE_DIR

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

verify_native_inputs_readable() {
    local native_library native_path
    for native_library in \
        libinfinicore_cpp_api.so libinfiniop.so libinfinirt.so libinfiniccl.so
    do
        native_path="${PERSISTENT_INFINI_ROOT}/lib/${native_library}"
        [[ -f "${native_path}" ]] || {
            printf 'ERROR: verified native library is missing: %s\n' \
                "${native_path}" >&2
            return 1
        }
        [[ -r "${native_path}" ]] || {
            printf 'ERROR: native library is not readable by this job UID: %s\n' \
                "${native_path}" >&2
            return 1
        }
        stat -Lc '%A mode=%a uid=%u gid=%g %n' "${native_path}"
    done
}

verify_inputs() {
    local actual_source_sha256 actual_pybind_sha256
    local repo_list="${XMAKE_TMPDIR}/repository-cache.list"
    local source_list="${XMAKE_TMPDIR}/source-archive.list"
    actual_source_sha256="$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')"
    actual_pybind_sha256="$(sha256sum "${PYBIND_ARCHIVE}" | awk '{print $1}')"
    printf 'expected_source_archive_sha256=%s\n' "${SOURCE_ARCHIVE_SHA256}"
    printf 'actual_source_archive_sha256=%s\n' "${actual_source_sha256}"
    printf 'expected_pybind_sha256=%s\n' "${PYBIND_SHA256}"
    printf 'actual_pybind_sha256=%s\n' "${actual_pybind_sha256}"
    [[ "${actual_source_sha256}" == "${SOURCE_ARCHIVE_SHA256}" ]]
    [[ "${actual_pybind_sha256}" == "${PYBIND_SHA256}" ]]
    tar -tzf "${REPO_CACHE}" > "${repo_list}"
    grep -Fxq '.xmake/repositories/xmake-repo/packages/p/pybind11/xmake.lua' \
        "${repo_list}"
    tar -tzf "${SOURCE_ARCHIVE}" > "${source_list}"
    grep -Fxq 'InfiniCore/xmake.lua' "${source_list}"
}

prepare_local_tree() {
    printf 'repository_cache_extract=begin\n'
    tar -xzf "${REPO_CACHE}" -C "${XMAKE_GLOBALDIR}"
    printf 'repository_cache_extract=complete\n'
    printf 'source_archive_extract=begin\n'
    tar -xzf "${SOURCE_ARCHIVE}" -C "${LOCAL_ROOT}"
    printf 'source_archive_extract=complete\n'
    printf 'pybind_archive_extract=begin\n'
    mkdir -p "${LOCAL_ROOT}/pybind11-src"
    "${PYTHON}" -m zipfile -e "${PYBIND_ARCHIVE}" \
        "${LOCAL_ROOT}/pybind11-src"
    printf 'pybind_archive_extract=complete\n'
    test -f "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/p/pybind11/xmake.lua"
    test -f "${LOCAL_PROJECT_DIR}/xmake.lua"
    test -f "${LOCAL_PYBIND_INCLUDE}/pybind11/pybind11.h"
}

verify_local_source() {
    cd -- "${LOCAL_ROOT}"
    sha256sum --check --quiet InfiniCore.SHA256SUMS
}

patch_binding_only_project() {
    local xmake_file="${LOCAL_PROJECT_DIR}/xmake.lua"
    local boost_line='add_requires("boost", {configs = {stacktrace = true}})'
    local pybind_require_line='add_requires("pybind11")'
    local pybind_package_line='    add_packages("pybind11")'
    local pybind_include_line="    add_includedirs(\"${LOCAL_PYBIND_INCLUDE}\", \"${PYTHON_INCLUDE_DIR}\", { public = false })"
    local dependency_line='    add_deps("infinicore_cpp_api")'
    local installed_link_line='    add_links("infinicore_cpp_api")'
    [[ "$(grep -Fxc "${boost_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${pybind_require_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${pybind_package_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${dependency_line}" "${xmake_file}")" -eq 1 ]]
    sed -i \
        -e '/^add_requires("boost", {configs = {stacktrace = true}})$/d' \
        -e '/^add_requires("pybind11")$/d' \
        -e "s#^    add_packages(\"pybind11\")\$#${pybind_include_line}#" \
        -e 's/^    add_deps("infinicore_cpp_api")$/    add_links("infinicore_cpp_api")/' \
        "${xmake_file}"
    ! grep -Fq "${boost_line}" "${xmake_file}"
    ! grep -Fq "${pybind_require_line}" "${xmake_file}"
    ! grep -Fq "${pybind_package_line}" "${xmake_file}"
    [[ "$(grep -Fxc "${pybind_include_line}" "${xmake_file}")" -eq 1 ]]
    ! grep -Fq "${dependency_line}" "${xmake_file}"
    [[ "$(grep -Fxc "${installed_link_line}" "${xmake_file}")" -eq 1 ]]
    printf 'binding_patch=link-installed-infinicore_cpp_api,vendored-pybind-headers\n'
    printf 'patched_xmake_sha256='
    sha256sum "${xmake_file}" | awk '{print $1}'
}

log_line "build_version=${BUILD_VERSION}"
log_line "time=$(date -Is)"
log_line "host=$(hostname)"
log_line "uid=$(id -u)"
log_line "work_home=${WORK_HOME}"
log_line "persistent_project=${PERSISTENT_PROJECT_DIR}"
log_line "persistent_infini_root=${PERSISTENT_INFINI_ROOT}"
log_line "local_project=${LOCAL_PROJECT_DIR}"
log_line "build_jobs=${BUILD_JOBS}"
log_line "python_include_dir=${PYTHON_INCLUDE_DIR}"

[[ -f "${NATIVE_SUCCESS_MARKER}" ]] || {
    log_error "ERROR: verified native-stage marker is missing: ${NATIVE_SUCCESS_MARKER}"
    exit 1
}
[[ -f "${SOURCE_ARCHIVE}" ]] || {
    log_error "ERROR: source archive is missing beside this script: ${SOURCE_ARCHIVE}"
    exit 1
}
[[ -f "${REPO_CACHE}" ]] || {
    log_error "ERROR: xmake repository cache is missing: ${REPO_CACHE}"
    exit 1
}
[[ -f "${PYBIND_ARCHIVE}" ]] || {
    log_error "ERROR: pybind11 archive is missing: ${PYBIND_ARCHIVE}"
    exit 1
}
[[ -f "${PYTHON_INCLUDE_DIR}/Python.h" ]] || {
    log_error "ERROR: Python development header is missing: ${PYTHON_INCLUDE_DIR}/Python.h"
    exit 1
}
[[ -f "${SCRIPT_DIR}/inspect_hygon_arch.py" ]] || {
    log_error "ERROR: inspect_hygon_arch.py is missing beside this script."
    exit 1
}
[[ -f "${PERSISTENT_PROJECT_DIR}/python/infinicore/__init__.py" ]] || {
    log_error "ERROR: persistent InfiniCore Python sources are missing."
    exit 1
}

run_stage "native-input-readability" verify_native_inputs_readable

run_stage "input-check" verify_inputs
run_stage "local-tree" prepare_local_tree
run_stage "source-check" verify_local_source
run_stage "local-source-patch" patch_binding_only_project

run_stage "hardware-probe" \
    "${PYTHON}" "${SCRIPT_DIR}/inspect_hygon_arch.py" \
        --json "${HARDWARE_JSON}" \
        --print-arch
HYGON_ARCH="$(tail -n 1 "${RUN_DIR}/hardware-probe.log")"
if [[ ! "${HYGON_ARCH}" =~ ^gfx(906|936)$ ]]; then
    log_error "ERROR: invalid normalized architecture from probe: ${HYGON_ARCH@Q}"
    exit 1
fi
export HYGON_ARCH
log_line "hygon_arch=${HYGON_ARCH}"

CURRENT_STAGE="torch-runtime"
TORCH_LIB_DIR="$("${PYTHON}" -c \
    'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
readonly TORCH_LIB_DIR
for runtime_library in \
    libtorch.so libtorch_cpu.so libtorch_python.so libtorch_hip.so \
    libc10.so libc10_hip.so
do
    [[ -f "${TORCH_LIB_DIR}/${runtime_library}" ]] || {
        log_error "ERROR: PyTorch runtime library is missing: ${TORCH_LIB_DIR}/${runtime_library}"
        exit 1
    }
done
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${PERSISTENT_INFINI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PERSISTENT_PROJECT_DIR}/python:${PYTHONPATH:-}"
log_line "torch_lib_dir=${TORCH_LIB_DIR}"
log_line "torch_cxx11_abi=$("${PYTHON}" -c 'import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)')"

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
run_stage "xmake-build" xmake build -j "${BUILD_JOBS}" _infinicore

CURRENT_STAGE="extension-path"
EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
readonly EXT_SUFFIX
readonly BUILT_EXTENSION="${LOCAL_PROJECT_DIR}/build/linux/x86_64/release/_infinicore${EXT_SUFFIX}"
readonly DESTINATION_DIR="${PERSISTENT_PROJECT_DIR}/python/infinicore/lib"
readonly DESTINATION_EXTENSION="${DESTINATION_DIR}/_infinicore${EXT_SUFFIX}"
[[ -f "${BUILT_EXTENSION}" ]] || {
    log_error "ERROR: built Python extension is missing: ${BUILT_EXTENSION}"
    exit 1
}

run_stage "file" file "${BUILT_EXTENSION}"
run_stage "ldd" ldd "${BUILT_EXTENSION}"
if grep -Fq "not found" "${RUN_DIR}/ldd.log"; then
    log_error "ERROR: unresolved dynamic library dependency detected."
    exit 1
fi
run_stage "built-extension-readability" chmod o+r -- "${BUILT_EXTENSION}"

CURRENT_STAGE="extension-promotion"
mkdir -p "${DESTINATION_DIR}"
if [[ -f "${DESTINATION_EXTENSION}" ]]; then
    if [[ -r "${DESTINATION_EXTENSION}" ]]; then
        cp -p -- "${DESTINATION_EXTENSION}" \
            "${RUN_DIR}/previous-_infinicore${EXT_SUFFIX}"
    else
        log_line "previous_extension_backup=SKIPPED_UNREADABLE"
    fi
fi
EXTENSION_TMP="${DESTINATION_EXTENSION}.tmp.$$"
cp -p -- "${BUILT_EXTENSION}" "${EXTENSION_TMP}"
[[ "$(sha256sum "${BUILT_EXTENSION}" | awk '{print $1}')" == \
   "$(sha256sum "${EXTENSION_TMP}" | awk '{print $1}')" ]]
mv -f -- "${EXTENSION_TMP}" "${DESTINATION_EXTENSION}"
[[ -r "${DESTINATION_EXTENSION}" ]] || {
    log_error "ERROR: promoted extension is not readable in the build job."
    exit 1
}

export EXPECTED_EXTENSION="${DESTINATION_EXTENSION}"
run_stage "import-smoke" "${PYTHON}" -c \
    'import os; from pathlib import Path; import torch, infinicore; from infinicore.lib import _infinicore; actual=Path(_infinicore.__file__).resolve(); expected=Path(os.environ["EXPECTED_EXTENSION"]).resolve(); assert actual == expected, (actual, expected); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print("infinicore:", infinicore.__file__); print("extension:", actual); print("device_count:", torch.cuda.device_count())'

CURRENT_STAGE="artifact-hashing"
sha256sum "${DESTINATION_EXTENSION}" "${CONFIG_SNAPSHOT}" > "${ARTIFACT_HASHES}"
cat "${ARTIFACT_HASHES}" | tee -a "${RUN_LOG}"

CURRENT_STAGE="success-marker"
readonly SUCCESS_MARKER="${STATE_DIR}/infinicore-python.success"
SUCCESS_MARKER_TMP="${SUCCESS_MARKER}.tmp.$$"
{
    printf 'build_version=%s\n' "${BUILD_VERSION}"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'hygon_arch=%s\n' "${HYGON_ARCH}"
    printf 'extension=%s\n' "${DESTINATION_EXTENSION}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'artifacts_sha256=%s\n' "${ARTIFACT_HASHES}"
} > "${SUCCESS_MARKER_TMP}"
mv -f -- "${SUCCESS_MARKER_TMP}" "${SUCCESS_MARKER}"

BUILD_PASSED=1
CURRENT_STAGE="complete"
log_line "InfiniCore Python binding: PASSED"
log_line "success_marker=${SUCCESS_MARKER}"
log_line "run_dir=${RUN_DIR}"
