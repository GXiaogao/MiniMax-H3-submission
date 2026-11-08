#!/usr/bin/env bash

set -Eeuo pipefail

DIAGNOSTIC_VERSION="1.0-infinilm-full-runtime-read-only"

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
HELPER="${SCRIPT_DIR}/diagnose_infinilm_full_runtime.py"
WHEEL_MANIFEST="${SCRIPT_DIR}/infinilm-local-wheel-inventory.json"
INFINICORE_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PROJECT="${EFFECTIVE_WORK_HOME}/Projects/InfiniLM"
INFINILM_PYTHON="${INFINILM_PROJECT}/python"
INFINI_ROOT="${EFFECTIVE_WORK_HOME}/.infini"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-full-runtime-diagnostic-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_line "diagnostic_version=${DIAGNOSTIC_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "python=${PYTHON}"
log_line "work_home=${EFFECTIVE_WORK_HOME}"
log_line "run_dir=${RUN_DIR}"

for required in "${HELPER}" "${WHEEL_MANIFEST}"; do
    if [[ ! -f "${required}" ]]; then
        log_line "ERROR: required diagnostic input is missing: ${required}"
        exit 2
    fi
done
for required_dir in "${INFINICORE_PYTHON}" "${INFINILM_PYTHON}" "${INFINI_ROOT}/lib"; do
    if [[ ! -d "${required_dir}" ]]; then
        log_line "ERROR: required runtime directory is missing: ${required_dir}"
        exit 2
    fi
done

TORCH_LIB="$(${PYTHON} -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

set +e
"${PYTHON}" "${HELPER}" \
    --project-root "${INFINILM_PROJECT}" \
    --infinicore-python "${INFINICORE_PYTHON}" \
    --infini-root "${INFINI_ROOT}" \
    --wheel-manifest "${WHEEL_MANIFEST}" \
    --run-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_LOG}"
HELPER_RC=${PIPESTATUS[0]}
set -e

RUNTIME_READY="unknown"
if [[ -f "${RUN_DIR}/full-runtime-diagnostic.json" ]]; then
    RUNTIME_READY="$(${PYTHON} -c 'import json, pathlib, sys; print(str(json.loads(pathlib.Path(sys.argv[1]).read_text())["runtime_ready"]).lower())' "${RUN_DIR}/full-runtime-diagnostic.json")"
fi

{
    printf 'diagnostic_version=%s\n' "${DIAGNOSTIC_VERSION}"
    printf 'diagnostic_result=%s\n' "$([[ "${HELPER_RC}" -eq 0 ]] && printf COLLECTED || printf FAILED)"
    printf 'helper_exit_code=%s\n' "${HELPER_RC}"
    printf 'runtime_ready=%s\n' "${RUNTIME_READY}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
} > "${SUMMARY}"

log_line "DIAGNOSTIC COLLECTED; runtime_ready=${RUNTIME_READY}; helper_exit_code=${HELPER_RC}"
log_line "SUMMARY: ${SUMMARY}"
log_line "DIAGNOSTIC_JSON: ${RUN_DIR}/full-runtime-diagnostic.json"
log_line "MISSING_RUNTIME: ${RUN_DIR}/missing-runtime.txt"
exit 0
