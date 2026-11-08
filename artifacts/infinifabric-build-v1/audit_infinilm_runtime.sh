#!/usr/bin/env bash

set -Eeuo pipefail

AUDIT_VERSION="1.0-infinilm-runtime-read-only"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
ENV_FILE="${WORK_HOME}/setup_infinifabric_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
    printf 'ERROR: environment file is missing: %s\n' "${ENV_FILE}" >&2
    exit 1
fi
source "${ENV_FILE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HELPER="${SCRIPT_DIR}/audit_infinilm_runtime.py"
INFINICORE_MARKER="${WORK_HOME}/state/infinicore-python.success"
INFINILM_EXTENSION_MARKER="${WORK_HOME}/state/infinilm-extension.success"
RUNTIME_MARKER="${WORK_HOME}/state/infinilm-runtime-import.success"
INFINICORE_PYTHON="${WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PROJECT="${WORK_HOME}/Projects/InfiniLM"
INFINILM_PYTHON="${INFINILM_PROJECT}/python"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${LOG_DIR}/infinifabric-audit-infinilm-runtime-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${WORK_HOME}/state"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_line "audit_version=${AUDIT_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "python=${PYTHON}"
log_line "work_home=${WORK_HOME}"
log_line "run_dir=${RUN_DIR}"

test -f "${HELPER}"
test -f "${INFINICORE_MARKER}"
test -f "${INFINILM_EXTENSION_MARKER}"
test -d "${INFINICORE_PYTHON}"
test -d "${INFINILM_PYTHON}"

TORCH_LIB="$(${PYTHON} -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
test -d "${TORCH_LIB}"
export LD_LIBRARY_PATH="${WORK_HOME}/.infini/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f "${RUNTIME_MARKER}" ]]; then
    mv -f -- "${RUNTIME_MARKER}" "${RUN_DIR}/previous-infinilm-runtime-import.success"
fi

set +e
"${PYTHON}" "${HELPER}" \
    --project-root "${INFINILM_PROJECT}" \
    --run-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_LOG}"
AUDIT_RC=${PIPESTATUS[0]}
set -e

RUNTIME_READY=0
if [[ "${AUDIT_RC}" -eq 0 ]]; then
    MARKER_TMP="${RUNTIME_MARKER}.tmp.$$"
    {
        printf 'audit_version=%s\n' "${AUDIT_VERSION}"
        printf 'time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)"
        sha256sum "${RUN_DIR}/runtime-audit.json"
    } > "${MARKER_TMP}"
    mv -f -- "${MARKER_TMP}" "${RUNTIME_MARKER}"
    RUNTIME_READY=1
fi

{
    printf 'audit_version=%s\n' "${AUDIT_VERSION}"
    printf 'diagnostic_result=COLLECTED\n'
    printf 'runtime_ready=%s\n' "${RUNTIME_READY}"
    printf 'helper_exit_code=%s\n' "${AUDIT_RC}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
} > "${SUMMARY}"

log_line "DIAGNOSTIC COLLECTED; runtime_ready=${RUNTIME_READY}; helper_exit_code=${AUDIT_RC}"
log_line "SUMMARY: ${SUMMARY}"
log_line "AUDIT_JSON: ${RUN_DIR}/runtime-audit.json"
log_line "MISSING_IMPORTS: ${RUN_DIR}/missing-imports.txt"
log_line "TRACEBACK: ${RUN_DIR}/canonical-import.stderr.log"
exit 0
