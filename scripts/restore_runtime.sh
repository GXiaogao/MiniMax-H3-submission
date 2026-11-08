#!/usr/bin/env bash
# Called by setup_env.sh to restore installations from existing artifacts.
# Editable-install backend hooks may perform their own build checks.
set -euo pipefail

WORK_HOME="${WORK_HOME:-/root/private_data}"
INFINI_ROOT="${INFINI_ROOT:-/root/.infini}"
WHEEL_DIR="${WHEEL_DIR:-${WORK_HOME}/wheels}"
LOG_DIR="${LOG_DIR:-${WORK_HOME}/logs}"
REQUIREMENTS="${WORK_HOME}/scripts/infinilm_runtime_requirements.txt"
PROBE="${WORK_HOME}/scripts/runtime_probe.py"
DETAIL_LOG="${LOG_DIR}/setup_runtime_details.log"
CONSTRAINTS="${LOG_DIR}/setup_runtime_constraints.txt"

mkdir -p "$LOG_DIR"
umask 0002
printf 'time: %s\nhost: %s\n' "$(date -Is)" "$(hostname)" >"$DETAIL_LOG"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

fail()
{
    tail -n 60 "$DETAIL_LOG" >&2
    printf '[restore_runtime] ERROR: %s; details: %s\n' "$1" "$DETAIL_LOG" >&2
    exit "${2:-1}"
}

step()
{
    local label="$1" rc
    shift
    printf '[restore_runtime] %s\n' "$label"
    { printf '\n+'; printf ' %q' "$@"; printf '\n'; } >>"$DETAIL_LOG"
    if "$@" >>"$DETAIL_LOG" 2>&1; then
        return 0
    else
        rc=$?
        fail "$label failed (exit $rc)" "$rc"
    fi
}

probe()
{
    printf '\n[probe] %s\n' "$*" >>"$DETAIL_LOG"
    python "$PROBE" "$@" >>"$DETAIL_LOG" 2>&1
}

# Only exit 10 authorizes a repair. Syntax/metadata/probe errors are not
# interpreted as permission to reinstall everything.
needs_restore()
{
    local rc
    if probe "$@"; then
        return 1
    else
        rc=$?
    fi
    [[ "$rc" == 10 ]] || fail "probe $1 failed (exit $rc)" "$rc"
    return 0
}

in_project()
{
    local project="$1"
    shift
    (cd "$project" && "$@")
}

restore_project()
{
    local name="$1" project="$2" native_restored=0
    local -a native_command=(xmake install)
    [[ -f "$project/xmake.lua" ]] || fail "missing project: $project"
    if [[ "$name" == infinilm ]]; then
        native_command+=(infinicore_infer)
    elif [[ ! -f "$INFINI_ROOT/include/infinicore/tensor.hpp" ]]; then
        step 'Restoring InfiniCore native installation (header missing).' \
            in_project "$project" "${native_command[@]}"
        native_restored=1
    fi

    if needs_restore extension "$name" "$project"; then
        step "Restoring $name native Python extension." \
            in_project "$project" xmake install "_${name}"
    fi
    if needs_restore registration "$name" "$project"; then
        step "Registering $name with the current Python." \
            in_project "$project" python -m pip install --no-index \
                -e . --no-deps --no-build-isolation
    fi
    if ! probe project-import "$name" "$project"; then
        # A new Pod can lose native libraries while retaining source extensions.
        # Copy the existing build once; never repeat pip registration for this.
        if (( native_restored == 0 )); then
            step "Restoring $name native installation after failed import." \
                in_project "$project" "${native_command[@]}"
        fi
        probe project-import "$name" "$project" || fail "$name import failed"
    fi
    if [[ "$name" == infinicore && ! -f "$INFINI_ROOT/include/infinicore/tensor.hpp" ]]; then
        fail 'InfiniCore header is still missing after installation'
    fi
    printf '[restore_runtime] %s is ready.\n' "$name"
}

[[ -r "$PROBE" ]] || fail "missing probe: $PROBE"
[[ -r "$REQUIREMENTS" ]] || fail "missing requirements: $REQUIREMENTS"
if needs_restore dependencies "$REQUIREMENTS"; then
    [[ -d "$WHEEL_DIR" ]] || fail "missing offline wheels: $WHEEL_DIR"
    probe constraints "$CONSTRAINTS" || fail 'cannot protect the base Torch/NumPy versions'
    step 'Installing missing or mismatched dependencies from local wheels.' \
        python -m pip install --no-index --find-links="$WHEEL_DIR" \
            -c "$CONSTRAINTS" -r "$REQUIREMENTS"
    probe dependencies "$REQUIREMENTS" || fail 'dependency versions still do not satisfy requirements'
fi
# An ABI/import failure with satisfied metadata needs its real traceback, not
# an identical pip install repeated on every shell entry.
probe dependency-imports || fail 'Python imports failed; see the traceback above'
printf '[restore_runtime] Python dependency versions and imports are ready.\n'

restore_project infinicore "${WORK_HOME}/Projects/InfiniCore"
restore_project infinilm "${WORK_HOME}/Projects/InfiniLM"
printf '[restore_runtime] Runtime is ready. Details: %s\n' "$DETAIL_LOG"
