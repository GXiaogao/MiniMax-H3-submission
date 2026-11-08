#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$#" -ne 0 ]]; then
    printf 'ERROR: projection build accepts no program arguments.\n' >&2
    exit 2
fi
if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
ROOT_WORK="${IFAB_AFS_HOME}/minimax-h3"
source "${ROOT_WORK}/setup_infinifabric_env.sh"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${LOG_DIR:-${ROOT_WORK}/logs}/infinifabric-prepare-h3-projection-${BUILD_ID}"
LOCAL_ROOT="${PWD}/.h3-projection-build-${UID:-unknown}-$$"
PYTHON="${PYTHON:-$(command -v python3)}"
mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
RUN_LOG="${RUN_DIR}/run.log"
: > "${RUN_LOG}"
LAST_STAGE="preflight"
finish() {
    local rc="$?"
    trap - EXIT
    printf 'exit_code=%s\nlast_stage=%s\nrun_dir=%s\nvalidation_scope=isolated-trace-build-not-inference\n' \
        "${rc}" "${LAST_STAGE}" "${RUN_DIR}" > "${RUN_DIR}/summary.txt"
    printf 'SUMMARY: %s/summary.txt\n' "${RUN_DIR}" | tee -a "${RUN_LOG}"
    exit "${rc}"
}
trap finish EXIT
for name in prepare_minimax_h3_projection_trace.py prepare_minimax_h3_attention_trace.py prepare_minimax_h3_condition_trace.py prepare_minimax_h3_boundary_trace.py h3_projection_trace.hpp h3_attention_trace.hpp h3_boundary_trace.hpp build_infinilm_extension.sh; do
    if [[ ! -f "${SCRIPT_DIR}/${name}" ]]; then
        printf 'ERROR: required helper missing: %s\n' "${name}" | tee -a "${RUN_LOG}"
        exit 1
    fi
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export H3_BOUNDARY_PATCHER="${SCRIPT_DIR}/prepare_minimax_h3_projection_trace.py"
export H3_BOUNDARY_BUILD_ID="${BUILD_ID}"
unset H3_BOUNDARY_TRACE_DIR
LAST_STAGE="prepare-isolated-builder"
"${PYTHON}" "${H3_BOUNDARY_PATCHER}" --builder \
    --source "${SCRIPT_DIR}/build_infinilm_extension.sh" \
    --destination "${LOCAL_ROOT}/build-isolated.sh" 2>&1 | tee -a "${RUN_LOG}"
LAST_STAGE="build"
if timeout --signal=TERM --kill-after=30s 25m bash "${LOCAL_ROOT}/build-isolated.sh" 2>&1 | tee -a "${RUN_LOG}"; then
    LAST_STAGE="complete"
    printf 'ISOLATED_PROJECTION_BUILD COMPLETE\nmarker=%s/state/infinilm-projection.success\n' "${ROOT_WORK}" | tee -a "${RUN_LOG}"
else
    statuses=("${PIPESTATUS[@]}")
    rc="${statuses[0]}"
    if [[ "${rc}" -eq 0 ]]; then rc="${statuses[1]}"; fi
    exit "${rc}"
fi
