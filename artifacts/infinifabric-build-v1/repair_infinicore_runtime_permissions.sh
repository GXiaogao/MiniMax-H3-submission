#!/usr/bin/env bash

set -Eeuo pipefail

REPAIR_VERSION="1.1-afs-stable-nogroup-read"

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

PYTHON="${PYTHON:-$(command -v python3)}"
EXT_SUFFIX="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
INFINICORE_MARKER="${WORK_HOME}/state/infinicore-python.success"
INFINILM_MARKER="${WORK_HOME}/state/infinilm-extension.success"
REPAIR_MARKER="${WORK_HOME}/state/runtime-permissions.success"
INFINICORE_PYTHON="${WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PYTHON="${WORK_HOME}/Projects/InfiniLM/python"
INFINICORE_EXTENSION="${INFINICORE_PYTHON}/infinicore/lib/_infinicore${EXT_SUFFIX}"
INFINILM_LIB_DIR="${INFINILM_PYTHON}/infinilm/lib"
INFINILM_EXTENSION="${INFINILM_LIB_DIR}/_infinilm${EXT_SUFFIX}"
INFINI_LIB_DIR="${WORK_HOME}/.infini/lib"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${LOG_DIR}/infinifabric-repair-runtime-permissions-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"

mkdir -p "${RUN_DIR}" "${WORK_HOME}/state"
: > "${RUN_LOG}"

CURRENT_STAGE="initialization"
REPAIR_PASSED=0

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_error() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}" >&2
}

on_exit() {
    local exit_code="$1"
    local result="FAILED"
    if [[ "${REPAIR_PASSED}" -eq 1 && "${exit_code}" -eq 0 ]]; then
        result="PASSED"
    fi
    {
        printf 'repair_version=%s\n' "${REPAIR_VERSION}"
        printf 'result=%s\n' "${result}"
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'last_stage=%s\n' "${CURRENT_STAGE}"
        printf 'run_dir=%s\n' "${RUN_DIR}"
    } > "${SUMMARY}.tmp.$$"
    mv -f -- "${SUMMARY}.tmp.$$" "${SUMMARY}"
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
    log_line ""
    log_line "=== ${name} ==="
    set +e
    "$@" 2>&1 | tee -a "${RUN_DIR}/${name}.log" "${RUN_LOG}"
    local stage_rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${stage_rc}" -ne 0 ]]; then
        log_error "ERROR: stage ${name} failed with exit_code=${stage_rc}"
        return "${stage_rc}"
    fi
}

show_permissions() {
    stat -Lc '%A mode=%a uid=%u gid=%g %n' "$@"
}

repair_permissions() {
    local current_uid shared_gid file owner_uid file_gid mode group_digit
    current_uid="$(id -u)"
    shared_gid="$(getent group nogroup | awk -F: '{print $3}')"
    if [[ -z "${shared_gid}" ]]; then
        printf 'ERROR: stable group nogroup is unavailable.\n' >&2
        return 1
    fi
    if ! id -G | tr ' ' '\n' | grep -Fxq "${shared_gid}"; then
        printf 'ERROR: current identity is not a member of nogroup gid=%s.\n' \
            "${shared_gid}" >&2
        return 1
    fi

    for file in "$@"; do
        owner_uid="$(stat -Lc '%u' "${file}")"
        file_gid="$(stat -Lc '%g' "${file}")"
        if [[ "${file_gid}" != "${shared_gid}" ]]; then
            if [[ "${owner_uid}" != "${current_uid}" ]]; then
                printf 'ERROR: cannot move unowned file to stable group: %s\n' \
                    "${file}" >&2
                return 1
            fi
            chgrp nogroup -- "${file}"
            printf 'group_changed_to_nogroup=%s\n' "${file}"
        fi

        mode="$(stat -Lc '%a' "${file}")"
        group_digit=$(( (8#${mode} / 8) % 8 ))
        if (( (group_digit & 4) == 0 )); then
            if [[ "${owner_uid}" != "${current_uid}" ]]; then
                printf 'ERROR: stable-group file lacks group read and is unowned: %s\n' \
                    "${file}" >&2
                return 1
            fi
            chmod g+r -- "${file}"
            printf 'group_read_added=%s\n' "${file}"
        else
            printf 'already_shared_group_read=%s\n' "${file}"
        fi
    done
}

log_line "repair_version=${REPAIR_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "identity=$(id)"
log_line "work_home=${WORK_HOME}"

CURRENT_STAGE="preconditions"
test -f "${INFINICORE_MARKER}"
test -f "${INFINILM_MARKER}"
test -f "${INFINICORE_EXTENSION}"
test -f "${INFINILM_EXTENSION}"
test -d "${INFINI_LIB_DIR}"

shopt -s nullglob
runtime_libraries=("${INFINI_LIB_DIR}"/*.so "${INFINI_LIB_DIR}"/*.so.*)
shopt -u nullglob
if [[ "${#runtime_libraries[@]}" -eq 0 ]]; then
    log_error "ERROR: no InfiniCore runtime libraries found in ${INFINI_LIB_DIR}"
    exit 1
fi
permission_files=(
    "${INFINICORE_EXTENSION}"
    "${INFINILM_EXTENSION}"
    "${runtime_libraries[@]}"
)

run_stage "permissions-before" show_permissions "${permission_files[@]}"
run_stage "group-read-repair" repair_permissions "${permission_files[@]}"
run_stage "permissions-after" show_permissions "${permission_files[@]}"

for file in "${permission_files[@]}"; do
    if [[ ! -r "${file}" ]]; then
        log_error "ERROR: file remains unreadable after repair: ${file}"
        exit 1
    fi
done

TORCH_LIB_DIR="$(${PYTHON} -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
test -d "${TORCH_LIB_DIR}"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${INFINI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export INFINILM_LIB_DIR
export PYTHONDONTWRITEBYTECODE=1

run_stage "import-validation" "${PYTHON}" -c '
import os
import sys
import torch
import infinicore
from infinicore.lib import _infinicore

sys.path.insert(0, os.environ["INFINILM_LIB_DIR"])
import _infinilm

hygon = _infinicore.Device.Type.HYGON
hygon_count = _infinicore.get_device_count(hygon)
torch_device_count = torch.cuda.device_count()
assert hygon_count == 8, hygon_count
assert torch.cuda.is_available()
assert torch_device_count == 8, torch_device_count
assert hasattr(_infinilm, "InferEngine")
assert hasattr(_infinilm, "DistConfig")
print("infinicore:", infinicore.__file__)
print("infinicore_extension:", _infinicore.__file__)
print("infinilm_extension:", _infinilm.__file__)
print("hygon_count:", hygon_count)
print("torch_device_count:", torch_device_count)
print("InferEngine: OK")
print("DistConfig: OK")
'

CURRENT_STAGE="success-marker"
{
    printf 'repair_version=%s\n' "${REPAIR_VERSION}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    sha256sum "${INFINICORE_EXTENSION}" "${INFINILM_EXTENSION}"
} > "${REPAIR_MARKER}.tmp.$$"
mv -f -- "${REPAIR_MARKER}.tmp.$$" "${REPAIR_MARKER}"

REPAIR_PASSED=1
CURRENT_STAGE="complete"
log_line "Permission repair validation: PASSED"
log_line "success_marker=${REPAIR_MARKER}"
