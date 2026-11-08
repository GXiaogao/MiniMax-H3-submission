#!/usr/bin/env bash

set -Eeuo pipefail

readonly INSTALLER_VERSION="3.0-split-clean"
readonly PROJECT_NAME="InfiniCore"
readonly ARCHIVE_NAME="InfiniCore-infinifabric-v3.tar.gz"
readonly EXPECTED_SHA256="37414edbc25545063462a3cbe47811ca0a672378058d78b761ec36e6fc3f8811"

SCRIPT_DIR_LOCAL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${SCRIPT_DIR_LOCAL}/${ARCHIVE_NAME}"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    echo "ERROR: IFAB_AFS_HOME is not set." >&2
    exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "ERROR: archive not found: ${ARCHIVE_PATH}" >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
PROJECTS_DIR="${WORK_HOME}/Projects"
LOG_DIR="${WORK_HOME}/logs"
TARGET_DIR="${PROJECTS_DIR}/${PROJECT_NAME}"
BACKUP_DIR="${WORK_HOME}/backups/source-migration-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/install_infinicore_v3.log"

mkdir -p "${PROJECTS_DIR}" "${LOG_DIR}"

{
    echo "installer_version=${INSTALLER_VERSION}"
    echo "project=${PROJECT_NAME}"
    echo "time=$(date -Is)"

    hash_line="$(sha256sum "${ARCHIVE_PATH}")"
    actual_hash="${hash_line%% *}"
    echo "expected_sha256=${EXPECTED_SHA256}"
    echo "actual_sha256=${actual_hash}"

    if [[ "${actual_hash}" != "${EXPECTED_SHA256}" ]]; then
        echo "ERROR: archive SHA256 mismatch." >&2
        exit 1
    fi

    if [[ -d "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -print -quit)" ]]; then
        mkdir -p "${BACKUP_DIR}"
        mv "${TARGET_DIR}" "${BACKUP_DIR}/${PROJECT_NAME}.partial"
        echo "previous_partial_tree=${BACKUP_DIR}/${PROJECT_NAME}.partial"
    fi

    echo "Extracting ${PROJECT_NAME}..."
    tar -xzf "${ARCHIVE_PATH}" -C "${PROJECTS_DIR}"

    echo "Verifying ${PROJECT_NAME}..."
    cd "${PROJECTS_DIR}"
    tr -d '\r' < "${PROJECT_NAME}.SHA256SUMS" | sha256sum --check --quiet -

    if find "${TARGET_DIR}" -type f -name '*.so' -print -quit | grep -q .; then
        echo "ERROR: old shared libraries were found in the clean source tree." >&2
        exit 1
    fi

    echo "source_checksum=OK"
    echo "binary_artifact_check=OK"
    echo "${PROJECT_NAME} migration: PASSED"
} 2>&1 | tee "${LOG_FILE}"

echo "log=${LOG_FILE}"
