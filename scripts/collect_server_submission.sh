#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is required.\n'; exit 1
fi
TASK_WORK="${IFAB_AFS_HOME}/minimax-h3"
TASK_LOG_DIR="${TASK_WORK}/logs"
mkdir -p "${TASK_LOG_DIR}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
TASK_LOG="${TASK_LOG_DIR}/source-environment-launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
timeout --signal=TERM --kill-after=20s 15m python3 "${SCRIPT_DIR}/collect_server_submission.py" \
    --work-home "${TASK_WORK}" 2>&1 | tee "${TASK_LOG}"
printf 'LAUNCH_LOG=%s\n' "${TASK_LOG}"
