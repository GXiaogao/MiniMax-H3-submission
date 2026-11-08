#!/usr/bin/env bash

set -Eeuo pipefail

readonly INSTALLER_VERSION="2.2-resume"
readonly ARCHIVE_NAME="infinifabric-source-v2.tar.gz"
readonly EXPECTED_ARCHIVE_SHA256="5e98dc6ef33a56e8a19a817e2ce65bd2d39dd86cdfc386584f29d001be4824e6"

SCRIPT_DIR_LOCAL="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

ARCHIVE_PATH="${SCRIPT_DIR_LOCAL}/${ARCHIVE_NAME}"

echo "installer_version=${INSTALLER_VERSION}"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    echo "ERROR: IFAB_AFS_HOME is not set." >&2
    exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "ERROR: archive not found: ${ARCHIVE_PATH}" >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
LOG_DIR="${WORK_HOME}/logs"
INSTALL_LOG="${LOG_DIR}/infinifabric_source_resume_v2.log"

mkdir -p "${WORK_HOME}" "${LOG_DIR}"

{
    echo "time: $(date -Is)"
    echo "installer_version=${INSTALLER_VERSION}"
    echo "archive: ${ARCHIVE_PATH}"
    echo "destination: ${WORK_HOME}"
    echo

    actual_hash_line="$(sha256sum "${ARCHIVE_PATH}")"
    actual_hash="${actual_hash_line%% *}"
    echo "expected_sha256=${EXPECTED_ARCHIVE_SHA256}"
    echo "actual_sha256=${actual_hash}"

    if [[ "${actual_hash}" != "${EXPECTED_ARCHIVE_SHA256}" ]]; then
        echo "ERROR: archive SHA256 mismatch." >&2
        exit 1
    fi

    echo "archive checksum: OK"
    echo "Resuming extraction over the partial tree..."
    tar -xzf "${ARCHIVE_PATH}" -C "${WORK_HOME}"

    test -f "${WORK_HOME}/MIGRATION_MANIFEST.json"
    test -f "${WORK_HOME}/SOURCE_SHA256SUMS"
    test -f "${WORK_HOME}/Projects/InfiniCore/xmake.lua"
    test -f "${WORK_HOME}/Projects/InfiniLM/xmake.lua"

    echo "Verifying extracted files quietly..."
    cd "${WORK_HOME}"
    tr -d '\r' < SOURCE_SHA256SUMS | sha256sum --check --quiet -

    chmod +x \
        "${WORK_HOME}/scripts/infinifabric/setup_env.sh" \
        "${WORK_HOME}/scripts/infinifabric/preflight.sh"

    echo "source checksum: OK"
    echo "Source bundle resume completed."
} 2>&1 | tee "${INSTALL_LOG}"

source "${WORK_HOME}/scripts/infinifabric/setup_env.sh"
bash "${WORK_HOME}/scripts/infinifabric/preflight.sh"

echo "resume log: ${INSTALL_LOG}"
echo "preflight log: ${LOG_DIR}/infinifabric_preflight.log"
