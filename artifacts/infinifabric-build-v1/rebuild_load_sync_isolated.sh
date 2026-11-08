#!/usr/bin/env bash
set -Eeuo pipefail
printf 'REBUILD_VERSION=1.0-isolated-receipt-source-gated\n'
[[ "$#" -eq 0 ]] || { printf 'ERROR: no arguments expected\n'; exit 2; }
[[ -n "${IFAB_AFS_HOME:-}" ]] || { printf 'ERROR: IFAB_AFS_HOME missing\n'; exit 1; }
TASK_WORK="${IFAB_AFS_HOME}/minimax-h3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOCAL_STAGE="${PWD}/.load-sync-clean-rebuild-${TASK_ID}"
mkdir -p "${LOCAL_STAGE}" "${TASK_WORK}/logs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export H3_BOUNDARY_BUILD_ID="${TASK_ID}"
export H3_BOUNDARY_PATCHER="${SCRIPT_DIR}/prepare_minimax_h3_load_sync_trace.py"
export H3_REBUILD_VERIFY_HELPER="${SCRIPT_DIR}/prepare_load_sync_clean_rebuild.py"
export H3_REBUILD_EXPECTED_HASHES="${SCRIPT_DIR}/load-sync-source-hashes.json"
unset H3_BOUNDARY_TRACE_DIR
python3 -u "${H3_REBUILD_VERIFY_HELPER}" --builder "${SCRIPT_DIR}/build_infinilm_extension.sh" \
    --destination "${LOCAL_STAGE}/build-isolated.sh"
cp -- "${SCRIPT_DIR}/InfiniLM-infinifabric-v3.tar.gz" "${LOCAL_STAGE}/"
TASK_LOG="${TASK_WORK}/logs/load-sync-clean-rebuild-launch-${TASK_ID}.log"
printf 'LAUNCH_LOG=%s\nOld success markers will not be replaced.\n' "${TASK_LOG}"
timeout --signal=TERM --kill-after=30s 25m bash "${LOCAL_STAGE}/build-isolated.sh" 2>&1 | tee "${TASK_LOG}"
printf 'REBUILD_LAUNCH_COMPLETE\nRead run_dir and success_marker above for new outputs.\n'
