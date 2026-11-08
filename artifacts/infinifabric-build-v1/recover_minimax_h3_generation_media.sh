#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$#" -ne 0 ]]; then
    printf 'ERROR: Existing media inspection accepts no program arguments.\n' >&2
    exit 2
fi
if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
ROOT_WORK="${WORK_HOME:-${IFAB_AFS_HOME}/minimax-h3}"
source "${ROOT_WORK}/setup_infinifabric_env.sh"
ROOT_WORK="${WORK_HOME:-${ROOT_WORK}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${ROOT_WORK}/logs/infinifabric-h3-existing-media-${RUN_ID}"
LOCAL_ROOT="${PWD}/.h3-media-inspection-${UID:-unknown}-$$"
PYTHON="${PYTHON:-$(command -v python3)}"
mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
RUN_LOG="${RUN_DIR}/run.log"
: > "${RUN_LOG}"
finish() {
    local rc="$?"
    trap - EXIT
    printf 'exit_code=%s\nrun_dir=%s\nnew_inference_executed=false\nquality_status=UNASSESSED\n' \
        "${rc}" "${RUN_DIR}" > "${RUN_DIR}/summary.txt"
    printf 'SUMMARY: %s/summary.txt\n' "${RUN_DIR}" | tee -a "${RUN_LOG}"
    exit "${rc}"
}
trap finish EXIT
for name in recover_minimax_h3_generation_media.py run_minimax_h3_load_sync_generation.py \
    diagnose_minimax_h3_full_forward_load_sync.py diagnose_minimax_h3_load_sync.py \
    diagnose_minimax_h3_prefix_parity.py diagnose_minimax_h3_transformer_parity.py \
    run_minimax_h3_fl2va_native_smoke.py install_infinilm_private_runtime.py; do
    if [[ ! -f "${SCRIPT_DIR}/${name}" ]]; then
        printf 'ERROR: required helper missing: %s\n' "${name}" | tee -a "${RUN_LOG}"
        exit 1
    fi
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf 'run_dir=%s\nmode=inspect-existing-video; new_inference=false\n' "${RUN_DIR}" | tee -a "${RUN_LOG}"
if timeout --signal=TERM --kill-after=10s 4m "${PYTHON}" -u "${SCRIPT_DIR}/recover_minimax_h3_generation_media.py" \
    --work-home "${ROOT_WORK}" --run-dir "${RUN_DIR}" --local-target "${LOCAL_ROOT}/site-packages" 2>&1 | tee -a "${RUN_LOG}"; then
    :
else
    statuses=("${PIPESTATUS[@]}")
    rc="${statuses[0]}"
    if [[ "${rc}" -eq 0 ]]; then rc="${statuses[1]}"; fi
    exit "${rc}"
fi
