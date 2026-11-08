#!/usr/bin/env bash

set -Eeuo pipefail

readonly VALIDATION_VERSION="1.0-hygon-import-recovery"

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

readonly PERSISTENT_PROJECT_DIR="${WORK_HOME}/Projects/InfiniCore"
readonly PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"
readonly NATIVE_SUCCESS_MARKER="${WORK_HOME}/state/infinicore-native.success"
readonly PYTHON_SUCCESS_MARKER="${WORK_HOME}/state/infinicore-python.success"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_DIR="${LOG_DIR}/infinifabric-validate-infinicore-python-${RUN_ID}"
readonly RUN_LOG="${RUN_DIR}/run.log"
readonly SUMMARY="${RUN_DIR}/summary.txt"
readonly ARTIFACT_HASHES="${RUN_DIR}/artifacts.sha256"
export PYTHON="${PYTHON:-$(command -v python3)}"

EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
readonly EXT_SUFFIX
readonly EXTENSION="${PERSISTENT_PROJECT_DIR}/python/infinicore/lib/_infinicore${EXT_SUFFIX}"

mkdir -p "${RUN_DIR}" "${WORK_HOME}/state"
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
log_line "extension=${EXTENSION}"

CURRENT_STAGE="installed-artifacts"
[[ -f "${NATIVE_SUCCESS_MARKER}" ]] || {
    log_error "ERROR: native success marker is missing: ${NATIVE_SUCCESS_MARKER}"
    exit 1
}
[[ -f "${EXTENSION}" ]] || {
    log_error "ERROR: built Python extension is missing: ${EXTENSION}"
    log_error "Run build_infinicore_python.sh before this validation script."
    exit 1
}

CURRENT_STAGE="invalidate-stale-success-marker"
if [[ -f "${PYTHON_SUCCESS_MARKER}" ]]; then
    mv -- "${PYTHON_SUCCESS_MARKER}" \
        "${RUN_DIR}/previous-infinicore-python.success"
fi

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
export EXPECTED_EXTENSION="${EXTENSION}"
log_line "torch_lib_dir=${TORCH_LIB_DIR}"

run_stage "file" file "${EXTENSION}"
run_stage "ldd" ldd "${EXTENSION}"
if grep -Fq "not found" "${RUN_DIR}/ldd.log"; then
    log_error "ERROR: unresolved dynamic library dependency detected."
    exit 1
fi

run_stage "import-smoke" "${PYTHON}" -c '
import os
from pathlib import Path
import torch, infinicore
from infinicore.lib import _infinicore

actual = Path(_infinicore.__file__).resolve()
expected = Path(os.environ["EXPECTED_EXTENSION"]).resolve()
assert actual == expected, (actual, expected)
hygon = _infinicore.Device.Type.HYGON
hygon_count = _infinicore.get_device_count(hygon)
current = _infinicore.get_device()
assert hygon_count == 8, hygon_count
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert current.type == hygon, current
print("infinicore:", infinicore.__file__)
print("extension:", actual)
print("current_device:", current)
print("hygon_count:", hygon_count)
print("torch_device_count:", torch.cuda.device_count())
'

CURRENT_STAGE="artifact-hashing"
sha256sum "${EXTENSION}" "${NATIVE_SUCCESS_MARKER}" > "${ARTIFACT_HASHES}"
cat "${ARTIFACT_HASHES}" | tee -a "${RUN_LOG}"

CURRENT_STAGE="success-marker"
SUCCESS_MARKER_TMP="${PYTHON_SUCCESS_MARKER}.tmp.$$"
{
    printf 'validation_version=%s\n' "${VALIDATION_VERSION}"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'extension=%s\n' "${EXTENSION}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'artifacts_sha256=%s\n' "${ARTIFACT_HASHES}"
} > "${SUCCESS_MARKER_TMP}"
mv -f -- "${SUCCESS_MARKER_TMP}" "${PYTHON_SUCCESS_MARKER}"

VALIDATION_PASSED=1
CURRENT_STAGE="complete"
log_line "InfiniCore Python validation: PASSED"
log_line "success_marker=${PYTHON_SUCCESS_MARKER}"
log_line "run_dir=${RUN_DIR}"
