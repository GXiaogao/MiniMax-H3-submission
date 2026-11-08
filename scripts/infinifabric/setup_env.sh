#!/usr/bin/env bash

# This file must be sourced so the environment remains in the caller shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[setup_env] ERROR: source this file instead of executing it:" >&2
    echo "  source /path/to/setup_env.sh" >&2
    exit 2
fi

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    echo "[setup_env] ERROR: IFAB_AFS_HOME is not set." >&2
    return 1
fi

if [[ ! -f /opt/dtk/env.sh ]]; then
    echo "[setup_env] ERROR: /opt/dtk/env.sh is missing." >&2
    return 1
fi

source /opt/dtk/env.sh

export WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
export LOG_DIR="${WORK_HOME}/logs"
export PROJECTS_DIR="${WORK_HOME}/Projects"
export PKG_DIR="${WORK_HOME}/packages"
export WHEEL_DIR="${WORK_HOME}/wheels"
export SCRIPT_DIR="${WORK_HOME}/scripts"
export INFINI_ROOT="${WORK_HOME}/.infini"
export XMAKE_GLOBALDIR="${WORK_HOME}/.xmake"
export XMAKE_ROOT=y

export PATH="${WORK_HOME}/tools/xmake:${PATH}"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
    "${LOG_DIR}" \
    "${PROJECTS_DIR}" \
    "${PKG_DIR}" \
    "${WHEEL_DIR}" \
    "${SCRIPT_DIR}" \
    "${WORK_HOME}/tools" \
    "${INFINI_ROOT}" \
    "${XMAKE_GLOBALDIR}"

umask 0002
set -o pipefail

run_logged() {
    if [[ "$#" -lt 2 ]]; then
        echo "[setup_env] ERROR: usage: run_logged <log-name> <command> [args...]" >&2
        return 2
    fi

    local log_name="$1"
    shift
    local log_path="${LOG_DIR}/${log_name}"

    echo "+ $*" | tee "${log_path}"
    "$@" 2>&1 | tee -a "${log_path}"
    local command_rc=${PIPESTATUS[0]}
    echo "exit_code=${command_rc}" | tee -a "${log_path}"
    return "${command_rc}"
}

echo "[setup_env] Ready: WORK_HOME=${WORK_HOME}, LOG_DIR=${LOG_DIR}"
