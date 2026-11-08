#!/usr/bin/env bash

set -Eeuo pipefail

INSTALL_VERSION="1.6-infinilm-private-runtime-job-local-staging"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
if [[ -z "${WORK_HOME:-}" ]]; then
    WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
fi
ENV_FILE="${WORK_HOME}/setup_infinifabric_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
    printf 'ERROR: environment file is missing: %s\n' "${ENV_FILE}" >&2
    exit 1
fi
source "${WORK_HOME}/setup_infinifabric_env.sh"

EFFECTIVE_WORK_HOME="${WORK_HOME:-${IFAB_AFS_HOME}/minimax-h3}"
EFFECTIVE_LOG_DIR="${LOG_DIR:-${EFFECTIVE_WORK_HOME}/logs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HELPER="${SCRIPT_DIR}/install_infinilm_private_runtime.py"
WHEEL_DIR="${EFFECTIVE_WORK_HOME}/packages/infinilm-runtime-wheels"
INFINICORE_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniLM/python"
INFINI_ROOT="${EFFECTIVE_WORK_HOME}/.infini"
STATE_DIR="${EFFECTIVE_WORK_HOME}/state"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-install-infinilm-private-runtime-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
LOCAL_ROOT="${PWD}/.infinilm-private-runtime-${UID:-unknown}-$$"
LOCAL_TARGET="${LOCAL_ROOT}/site-packages"
PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_line "install_version=${INSTALL_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "python=${PYTHON}"
log_line "work_home=${EFFECTIVE_WORK_HOME}"
log_line "wheel_dir=${WHEEL_DIR}"
log_line "run_dir=${RUN_DIR}"

test -f "${HELPER}"
test -d "${WHEEL_DIR}"
test -d "${INFINICORE_PYTHON}"
test -d "${INFINILM_PYTHON}"
test -d "${INFINI_ROOT}/lib"

export PYTHONUNBUFFERED=1
log_line "stage=torch-lib-detection status=START"
TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
log_line "stage=torch-lib-detection status=DONE"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

set +e
"${PYTHON}" -u "${HELPER}" \
    --wheel-dir "${WHEEL_DIR}" \
    --local-target "${LOCAL_TARGET}" \
    --state-dir "${STATE_DIR}" \
    --run-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_LOG}"
HELPER_RC=${PIPESTATUS[0]}
set -e

RESULT="FAILED"
if [[ "${HELPER_RC}" -eq 0 ]]; then
    RESULT="PASSED"
fi
{
    printf 'install_version=%s\n' "${INSTALL_VERSION}"
    printf 'result=%s\n' "${RESULT}"
    printf 'helper_exit_code=%s\n' "${HELPER_RC}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'result_json=%s\n' "${RUN_DIR}/runtime-install-result.json"
} > "${SUMMARY}"

log_line "SUMMARY: ${SUMMARY}"
log_line "RESULT_JSON: ${RUN_DIR}/runtime-install-result.json"
exit "${HELPER_RC}"
