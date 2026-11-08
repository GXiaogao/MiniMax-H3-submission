#!/usr/bin/env bash

set -Eeuo pipefail

PREPARE_VERSION="1.2-minimax-h3-hygon-runtime-pytest-dependency"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
if [[ -z "${WORK_HOME:-}" ]]; then
    WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
fi
if [[ ! -f "${WORK_HOME}/setup_infinifabric_env.sh" ]]; then
    printf 'ERROR: environment file is missing: %s\n' "${WORK_HOME}/setup_infinifabric_env.sh" >&2
    exit 1
fi
source "${WORK_HOME}/setup_infinifabric_env.sh"

EFFECTIVE_WORK_HOME="${WORK_HOME:-${IFAB_AFS_HOME}/minimax-h3}"
EFFECTIVE_LOG_DIR="${LOG_DIR:-${EFFECTIVE_WORK_HOME}/logs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HELPER="${SCRIPT_DIR}/prepare_minimax_h3_hygon_runtime.py"
RUNTIME_HELPER="${SCRIPT_DIR}/install_infinilm_private_runtime.py"
INPUT_DIR="${EFFECTIVE_WORK_HOME}/packages/minimax-h3-hygon-runtime-inputs"
READY_DIR="${EFFECTIVE_WORK_HOME}/packages/minimax-h3-hygon-runtime-ready"
STATE_MANIFEST="${EFFECTIVE_WORK_HOME}/state/minimax-h3-hygon-runtime.json"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-prepare-minimax-h3-hygon-runtime-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
LOCAL_ROOT="${PWD}/.minimax-h3-hygon-runtime-prepare-${UID:-unknown}-$$"
LOCAL_TARGET="${LOCAL_ROOT}/site-packages"
PYTHON="${PYTHON:-$(command -v python3)}"
BUILD_JOBS="${BUILD_JOBS:-16}"

mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}" "${READY_DIR}" "${EFFECTIVE_WORK_HOME}/state"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_line "prepare_version=${PREPARE_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "work_home=${EFFECTIVE_WORK_HOME}"
log_line "input_dir=${INPUT_DIR}"
log_line "run_dir=${RUN_DIR}"

test -f "${HELPER}"
test -f "${RUNTIME_HELPER}"
test -d "${INPUT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${EFFECTIVE_WORK_HOME}/.infini/lib:${TORCH_LIB}:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

set +e
"${PYTHON}" -u "${HELPER}" \
    --runtime-helper "${RUNTIME_HELPER}" \
    --input-dir "${INPUT_DIR}" \
    --ready-dir "${READY_DIR}" \
    --state-manifest "${STATE_MANIFEST}" \
    --work-dir "${LOCAL_ROOT}/work" \
    --local-target "${LOCAL_TARGET}" \
    --run-dir "${RUN_DIR}" \
    --jobs "${BUILD_JOBS}" 2>&1 | tee -a "${RUN_LOG}"
HELPER_RC=${PIPESTATUS[0]}
set -e

{
    printf 'prepare_version=%s\n' "${PREPARE_VERSION}"
    printf 'helper_exit_code=%s\n' "${HELPER_RC}"
    printf 'state_manifest=%s\n' "${STATE_MANIFEST}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
} > "${SUMMARY}"

log_line "SUMMARY: ${SUMMARY}"
exit "${HELPER_RC}"
