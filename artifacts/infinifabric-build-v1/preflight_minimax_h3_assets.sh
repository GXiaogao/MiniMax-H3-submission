#!/usr/bin/env bash

set -Eeuo pipefail

PREFLIGHT_VERSION="1.2-minimax-h3-assets-permission-safe-discovery"

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
HELPER="${SCRIPT_DIR}/preflight_minimax_h3_assets.py"
RUNTIME_HELPER="${SCRIPT_DIR}/install_infinilm_private_runtime.py"
STATE_MANIFEST="${EFFECTIVE_WORK_HOME}/state/infinilm-runtime-private.json"
INFINICORE_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniLM/python"
INFINI_ROOT="${EFFECTIVE_WORK_HOME}/.infini"
SHARED_ROOT="/public/home/aclsylqidf/shared"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-preflight-minimax-h3-assets-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
LOCAL_ROOT="${PWD}/.minimax-h3-assets-preflight-${UID:-unknown}-$$"
LOCAL_TARGET="${PWD}/.minimax-h3-assets-preflight-${UID:-unknown}-$$/site-packages"
PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_line "preflight_version=${PREFLIGHT_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "work_home=${EFFECTIVE_WORK_HOME}"
log_line "state_manifest=${STATE_MANIFEST}"
log_line "run_dir=${RUN_DIR}"

test -f "${HELPER}"
test -f "${RUNTIME_HELPER}"
test -f "${STATE_MANIFEST}"
test -d "${INFINICORE_PYTHON}"
test -d "${INFINILM_PYTHON}"
test -d "${INFINI_ROOT}/lib"

export PYTHONUNBUFFERED=1
TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

set +e
"${PYTHON}" -u "${HELPER}" \
    --runtime-helper "${RUNTIME_HELPER}" \
    --state-manifest "${STATE_MANIFEST}" \
    --local-target "${LOCAL_TARGET}" \
    --work-home "${EFFECTIVE_WORK_HOME}" \
    --shared-root "${SHARED_ROOT}" \
    --run-dir "${RUN_DIR}" \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
HELPER_RC=${PIPESTATUS[0]}
set -e

{
    printf 'preflight_version=%s\n' "${PREFLIGHT_VERSION}"
    printf 'helper_exit_code=%s\n' "${HELPER_RC}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'result_json=%s\n' "${RUN_DIR}/minimax-h3-assets-preflight.json"
} > "${SUMMARY}"

log_line "SUMMARY: ${SUMMARY}"
exit "${HELPER_RC}"
