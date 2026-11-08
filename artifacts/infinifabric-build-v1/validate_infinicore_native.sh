#!/usr/bin/env bash

set -Eeuo pipefail

readonly VALIDATION_VERSION="1.0-reuse-installed-torch-runtime"

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

readonly PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_DIR="${LOG_DIR}/infinifabric-validate-infinicore-native-${RUN_ID}"
readonly RUN_LOG="${RUN_DIR}/run.log"
readonly SUMMARY="${RUN_DIR}/summary.txt"
readonly ARTIFACT_HASHES="${RUN_DIR}/artifacts.sha256"
readonly STATE_DIR="${WORK_HOME}/state"
readonly HEADER="${PERSISTENT_INFINI_ROOT}/include/infinicore/tensor.hpp"
readonly LIBRARY="${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so"
readonly SUCCESS_MARKER="${STATE_DIR}/infinicore-native.success"
export PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${STATE_DIR}"
: > "${RUN_LOG}"

CURRENT_STAGE="initialization"
VALIDATION_PASSED=0

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_error() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}" >&2
}

on_exit() {
    local exit_code="$1"
    local result="FAILED"
    if [[ "${VALIDATION_PASSED}" -eq 1 && "${exit_code}" -eq 0 ]]; then
        result="PASSED"
    fi
    local summary_tmp="${SUMMARY}.tmp.$$"
    {
        printf 'validation_version=%s\n' "${VALIDATION_VERSION}"
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

log_line "validation_version=${VALIDATION_VERSION}"
log_line "time=$(date -Is)"
log_line "host=$(hostname)"
log_line "uid=$(id -u)"
log_line "work_home=${WORK_HOME}"
log_line "persistent_infini_root=${PERSISTENT_INFINI_ROOT}"

CURRENT_STAGE="installed-artifacts"
for installed_artifact in \
    "${HEADER}" \
    "${LIBRARY}" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniop.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniccl.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfinirt.so"
do
    [[ -f "${installed_artifact}" ]] || {
        log_error "ERROR: installed artifact is missing: ${installed_artifact}"
        exit 1
    }
done

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

run_stage "file" file "${LIBRARY}"
run_stage "ldd" ldd "${LIBRARY}"
if grep -Fq "not found" "${RUN_DIR}/ldd.log"; then
    log_error "ERROR: unresolved dynamic library dependency detected."
    exit 1
fi

CURRENT_STAGE="artifact-hashing"
sha256sum \
    "${HEADER}" \
    "${LIBRARY}" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniop.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniccl.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfinirt.so" \
    > "${ARTIFACT_HASHES}"
cat "${ARTIFACT_HASHES}" | tee -a "${RUN_LOG}"

CURRENT_STAGE="success-marker"
SUCCESS_MARKER_TMP="${SUCCESS_MARKER}.tmp.$$"
{
    printf 'validation_version=%s\n' "${VALIDATION_VERSION}"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'torch_lib_dir=%s\n' "${TORCH_LIB_DIR}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'artifacts_sha256=%s\n' "${ARTIFACT_HASHES}"
} > "${SUCCESS_MARKER_TMP}"
mv -f -- "${SUCCESS_MARKER_TMP}" "${SUCCESS_MARKER}"

VALIDATION_PASSED=1
CURRENT_STAGE="complete"
log_line "InfiniCore native validation: PASSED"
log_line "success_marker=${SUCCESS_MARKER}"
log_line "run_dir=${RUN_DIR}"
