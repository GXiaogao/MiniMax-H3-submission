#!/usr/bin/env bash

# This file must be sourced so that its exported variables remain in the
# current Pod shell:
#   source /root/private_data/setup_env.sh

if [[ -z "${BASH_VERSION:-}" ]]; then
    printf '[setup_env] ERROR: this script requires Bash.\n' >&2
    return 2 2>/dev/null || exit 2
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '[setup_env] ERROR: run this script with: source %q\n' "$0" >&2
    exit 2
fi

_minimax_error()
{
    printf '[setup_env] ERROR: %s\n' "$*" >&2
}

_minimax_prepend_path()
{
    local variable_name="$1"
    local directory="$2"
    local current_value="${!variable_name:-}"

    case ":${current_value}:" in
        *":${directory}:"*) ;;
        *)
            printf -v "$variable_name" '%s%s%s' \
                "$directory" "${current_value:+:}" "$current_value"
            ;;
    esac

    export "$variable_name"
}

run_logged()
{
    if (( $# < 2 )); then
        _minimax_error 'usage: run_logged <log-file-name> <command> [args...]'
        return 2
    fi

    local log_name="$1"
    local log_path
    local command_rc
    local tee_rc
    local -a pipeline_status
    shift

    if [[ "$log_name" == */* ]]; then
        _minimax_error "log name must not contain '/': $log_name"
        return 2
    fi

    log_path="${LOG_DIR}/${log_name}"

    {
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    } | tee "$log_path" >/dev/null || {
        _minimax_error "cannot create log file: $log_path"
        return 1
    }

    "$@" 2>&1 | tee -a "$log_path"
    pipeline_status=("${PIPESTATUS[@]}")
    command_rc="${pipeline_status[0]}"
    tee_rc="${pipeline_status[1]}"

    if (( tee_rc != 0 )); then
        _minimax_error "failed while writing log: $log_path"
        return 1
    fi

    if ! printf '\nexit code: %d\n' "$command_rc" | tee -a "$log_path"; then
        _minimax_error "failed while finalizing log: $log_path"
        return 1
    fi

    return "$command_rc"
}

ensure_xmake_offline()
{
    local setup_log="${LOG_DIR}/setup_xmake_offline.log"
    local repo_root="${XMAKE_GLOBALDIR}/repositories/xmake-repo"
    local boost_recipe="${repo_root}/packages/b/boost/xmake.lua"
    local pybind11_recipe="${repo_root}/packages/p/pybind11/xmake.lua"
    local xmake_config

    if ! command -v xmake >/dev/null 2>&1; then
        _minimax_error "xmake not found; expected ${WORK_HOME}/tools/xmake/xmake"
        return 1
    fi

    if ! xmake_config="$(xmake g 2>&1)"; then
        printf '%s\n' "$xmake_config" >"$setup_log"
        _minimax_error "failed to read xmake configuration; see $setup_log"
        return 1
    fi
    printf '%s\n' "$xmake_config" >"$setup_log"

    # Persistent settings normally survive a Pod replacement. Write only when
    # the effective settings differ, instead of reconfiguring every shell.
    if ! grep -Eq 'network[[:space:]]*=[[:space:]]*private' <<<"$xmake_config" ||
       ! grep -Fq "$PKG_DIR" <<<"$xmake_config"; then
        if ! xmake g --network=private --pkg_searchdirs="$PKG_DIR" \
            >>"$setup_log" 2>&1; then
            tail -n 50 "$setup_log" >&2
            _minimax_error 'failed to configure xmake private-network mode'
            return 1
        fi
    fi

    if [[ ! -f "$boost_recipe" || ! -f "$pybind11_recipe" ]]; then
        printf '[setup_env] Restoring the offline xmake repository cache.\n'

        if [[ ! -r "$XMAKE_REPO_CACHE" ]]; then
            _minimax_error "repository cache not found: $XMAKE_REPO_CACHE"
            return 1
        fi

        if ! tar -tzf "$XMAKE_REPO_CACHE" >/dev/null 2>&1; then
            _minimax_error "repository cache is not a valid tar.gz: $XMAKE_REPO_CACHE"
            return 1
        fi

        if ! tar -xzf "$XMAKE_REPO_CACHE" \
            -C "$WORK_HOME" >>"$setup_log" 2>&1; then
            tail -n 50 "$setup_log" >&2
            _minimax_error 'failed to extract the offline xmake repository cache'
            return 1
        fi
    fi

    if [[ ! -f "$boost_recipe" ]]; then
        _minimax_error "Boost recipe is missing after recovery: $boost_recipe"
        return 1
    fi

    if [[ ! -f "$pybind11_recipe" ]]; then
        _minimax_error "pybind11 recipe is missing after recovery: $pybind11_recipe"
        return 1
    fi

    if ! xmake_config="$(xmake g 2>&1)"; then
        printf '%s\n' "$xmake_config" >&2
        _minimax_error 'failed to read xmake global configuration'
        return 1
    fi

    printf '%s\n' "$xmake_config" >>"$setup_log"

    if ! grep -Eq 'network[[:space:]]*=[[:space:]]*private' \
        <<<"$xmake_config"; then
        _minimax_error "xmake network mode is not private; see $setup_log"
        return 1
    fi

    if ! grep -Fq "$PKG_DIR" <<<"$xmake_config"; then
        _minimax_error "xmake package search directory is incorrect; see $setup_log"
        return 1
    fi
}

ensure_git_safe_directory()
{
    local repository="$1"
    local configured_directory

    if [[ ! -e "${repository}/.git" ]]; then
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        _minimax_error 'git is required but was not found in PATH'
        return 1
    fi

    while IFS= read -r configured_directory; do
        if [[ "$configured_directory" == "$repository" ]]; then
            return 0
        fi
    done < <(git config --global --get-all safe.directory 2>/dev/null || true)

    printf '[setup_env] Adding Git safe.directory: %s\n' "$repository"
    if ! git config --global --add safe.directory "$repository"; then
        _minimax_error "failed to register Git safe.directory: $repository"
        return 1
    fi
}

ensure_xmake_project_repository()
{
    local project_directory="$1"
    local project_name="${project_directory##*/}"
    local repository_name='minimax-offline'
    local repository_root="${XMAKE_GLOBALDIR}/repositories/xmake-repo"
    local setup_log="${LOG_DIR}/setup_xmake_repo_${project_name}.log"
    local repository_list

    if [[ ! -f "${project_directory}/xmake.lua" ]]; then
        return 0
    fi

    if ! repository_list="$(
        cd "$project_directory" && xmake repo --list 2>&1
    )"; then
        printf '%s\n' "$repository_list" >"$setup_log"
        _minimax_error "failed to list xmake repositories for $project_name; see $setup_log"
        return 1
    fi

    if grep -Fq "$repository_root" <<<"$repository_list"; then
        return 0
    fi

    printf '[setup_env] Registering the offline xmake repository for %s.\n' \
        "$project_name"

    if ! (
        cd "$project_directory" || exit 1
        xmake repo --remove "$repository_name" >/dev/null 2>&1 || true
        xmake repo --add "$repository_name" "$repository_root"
    ) >"$setup_log" 2>&1; then
        tail -n 50 "$setup_log" >&2
        _minimax_error "failed to register the offline xmake repository for $project_name"
        return 1
    fi
}

export DTK_ENV_FILE="${DTK_ENV_FILE:-/opt/dtk/env.sh}"

if [[ ! -r "$DTK_ENV_FILE" ]]; then
    _minimax_error "$DTK_ENV_FILE is missing or unreadable; are you inside the compute Pod?"
    return 1
fi

if ! source "$DTK_ENV_FILE"; then
    _minimax_error "failed to load $DTK_ENV_FILE"
    return 1
fi

export WORK_HOME="${WORK_HOME:-/root/private_data}"
export XMAKE_ROOT=y
export INFINI_ROOT="${INFINI_ROOT:-/root/.infini}"
export PKG_DIR="${WORK_HOME}/packages"
export WHEEL_DIR="${WORK_HOME}/wheels"
export LOG_DIR="${WORK_HOME}/logs"
export XMAKE_GLOBALDIR="${WORK_HOME}/.xmake"
export XMAKE_REPO_CACHE="${PKG_DIR}/xmake-repo-cache.tar.gz"
export MINIMAX_AUTO_RESTORE="${MINIMAX_AUTO_RESTORE:-1}"

if [[ ! -d "$WORK_HOME" ]]; then
    _minimax_error "persistent work directory does not exist: $WORK_HOME"
    return 1
fi

if [[ ! -d "$PKG_DIR" ]]; then
    _minimax_error "offline package directory does not exist: $PKG_DIR"
    return 1
fi

if ! mkdir -p "$LOG_DIR" "$XMAKE_GLOBALDIR"; then
    _minimax_error 'failed to create the log or persistent xmake directory'
    return 1
fi

umask 0002

# Prepend in reverse priority order. Re-sourcing this file does not duplicate
# entries that are already present.
_minimax_prepend_path PATH /opt/dtk/cuda/cuda/bin
_minimax_prepend_path PATH /root/.local/bin
_minimax_prepend_path PATH "${WORK_HOME}/tools/xmake"

_minimax_prepend_path LD_LIBRARY_PATH /opt/dtk/.hyhal/rocm_smi
_minimax_prepend_path LD_LIBRARY_PATH /opt/dtk/.hyhal/rocm_smi/lib
_minimax_prepend_path LD_LIBRARY_PATH /opt/hyhal/lib
_minimax_prepend_path LD_LIBRARY_PATH /usr/local/lib/python3.11/site-packages/torch/lib
_minimax_prepend_path LD_LIBRARY_PATH /opt/dtk/cuda/cuda/lib64
_minimax_prepend_path LD_LIBRARY_PATH /opt/dtk/lib
_minimax_prepend_path LD_LIBRARY_PATH "${INFINI_ROOT}/lib"

if ! ensure_xmake_offline; then
    _minimax_error 'environment initialization failed'
    return 1
fi

if ! ensure_git_safe_directory "${WORK_HOME}/Projects/InfiniCore" ||
   ! ensure_git_safe_directory "${WORK_HOME}/Projects/InfiniLM"; then
    _minimax_error 'failed to configure Git repository ownership exceptions'
    return 1
fi

if ! ensure_xmake_project_repository "${WORK_HOME}/Projects/InfiniCore" ||
   ! ensure_xmake_project_repository "${WORK_HOME}/Projects/InfiniLM"; then
    _minimax_error 'failed to configure project-local xmake repositories'
    return 1
fi

case "$MINIMAX_AUTO_RESTORE" in
    0)
        printf '[setup_env] Runtime auto-restore is disabled.\n'
        ;;
    1)
        if [[ ! -r "${WORK_HOME}/scripts/restore_runtime.sh" ]]; then
            _minimax_error \
                "runtime restore script is missing: ${WORK_HOME}/scripts/restore_runtime.sh"
            return 1
        fi

        if ! run_logged setup_runtime_restore.log \
            bash "${WORK_HOME}/scripts/restore_runtime.sh"; then
            _minimax_error \
                "runtime restoration failed; see ${LOG_DIR}/setup_runtime_restore.log"
            return 1
        fi
        ;;
    *)
        _minimax_error \
            "MINIMAX_AUTO_RESTORE must be 0 or 1, got: ${MINIMAX_AUTO_RESTORE}"
        return 2
        ;;
esac

printf '[setup_env] Ready: WORK_HOME=%s, LOG_DIR=%s\n' \
    "$WORK_HOME" "$LOG_DIR"
