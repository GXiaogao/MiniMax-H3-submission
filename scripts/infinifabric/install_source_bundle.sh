#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR_LOCAL="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

ARCHIVE_NAME="infinifabric-source-v2.tar.gz"
ARCHIVE_PATH="${SCRIPT_DIR_LOCAL}/${ARCHIVE_NAME}"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    echo "ERROR: IFAB_AFS_HOME is not set." >&2
    exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "ERROR: archive not found: ${ARCHIVE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${CHECKSUM_PATH}" ]]; then
    echo "ERROR: checksum file not found: ${CHECKSUM_PATH}" >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
LOG_DIR="${WORK_HOME}/logs"
INSTALL_LOG="${LOG_DIR}/infinifabric_source_install.log"

mkdir -p "${WORK_HOME}" "${LOG_DIR}"

for project in InfiniCore InfiniLM; do
    target="${WORK_HOME}/Projects/${project}"
    if [[ -d "${target}" ]] && [[ -n "$(find "${target}" -mindepth 1 -print -quit)" ]]; then
        echo "ERROR: target is not empty: ${target}" >&2
        echo "Move or archive the existing directory before installing." >&2
        exit 1
    fi
done

{
    echo "time: $(date -Is)"
    echo "archive: ${ARCHIVE_PATH}"
    echo "destination: ${WORK_HOME}"
    echo

    cd "${SCRIPT_DIR_LOCAL}"
    # The checksum file may have been created on Windows. Remove CR bytes so
    # sha256sum does not treat them as part of the file name.
    tr -d '\r' < "${CHECKSUM_PATH}" | sha256sum -c -

    echo
    echo "Extracting source bundle..."
    tar -xzf "${ARCHIVE_PATH}" -C "${WORK_HOME}"

    echo
    echo "Checking extracted source files..."
    cd "${WORK_HOME}"
    tr -d '\r' < SOURCE_SHA256SUMS | sha256sum -c -

    chmod +x \
        "${WORK_HOME}/scripts/infinifabric/setup_env.sh" \
        "${WORK_HOME}/scripts/infinifabric/preflight.sh"

    echo
    echo "Source bundle installed."
} 2>&1 | tee "${INSTALL_LOG}"

source "${WORK_HOME}/scripts/infinifabric/setup_env.sh"
bash "${WORK_HOME}/scripts/infinifabric/preflight.sh"

echo "install log: ${INSTALL_LOG}"
echo "preflight log: ${LOG_DIR}/infinifabric_preflight.log"
