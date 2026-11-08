#!/usr/bin/env bash

set -Eeuo pipefail

# Defaults can be edited, or overridden with command-line arguments.
WORK_HOME="${WORK_HOME:-/root/private_data}"
TARGET_DIR="${1:-${WORK_HOME}/wheels}"
REFERENCE_PATH="${2:-${WORK_HOME}/packages/pybind11-v3.1.0.zip}"
DIR_MODE="${3:-2775}"

usage() {
    cat <<'EOF'
Usage:
  bash fix_upload_dir_permissions.sh [target-dir] [reference-path] [mode]

Examples:
  bash fix_upload_dir_permissions.sh
  bash fix_upload_dir_permissions.sh /root/private_data/outputs
  bash fix_upload_dir_permissions.sh /root/private_data/wheels /root/private_data/packages/pybind11-v3.1.0.zip 2775

The target directory receives the same owner and group as reference-path.
Only the target directory itself is changed; existing contents are untouched.
EOF
}

fail() {
    printf '[fix-permissions] ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

[[ "${EUID}" -eq 0 ]] || fail "run this script as root inside the container"
[[ -e "${REFERENCE_PATH}" ]] || fail "reference path does not exist: ${REFERENCE_PATH}"
[[ "${DIR_MODE}" =~ ^[0-7]{3,4}$ ]] || fail "invalid directory mode: ${DIR_MODE}"

mkdir -p -- "${TARGET_DIR}"
[[ ! -L "${TARGET_DIR}" ]] || fail "refusing to modify a symbolic link: ${TARGET_DIR}"
[[ -d "${TARGET_DIR}" ]] || fail "target is not a directory: ${TARGET_DIR}"

printf '[fix-permissions] reference: '
stat -c '%U:%G (%u:%g) %a %n' -- "${REFERENCE_PATH}"
printf '[fix-permissions] before:    '
stat -c '%U:%G (%u:%g) %a %n' -- "${TARGET_DIR}"

chown --reference="${REFERENCE_PATH}" -- "${TARGET_DIR}"
chmod "${DIR_MODE}" -- "${TARGET_DIR}"

printf '[fix-permissions] after:     '
stat -c '%U:%G (%u:%g) %a %n' -- "${TARGET_DIR}"
printf '[fix-permissions] OK\n'
