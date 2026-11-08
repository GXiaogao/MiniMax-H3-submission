#!/usr/bin/env bash

set -Eeuo pipefail

SMOKE_VERSION="1.3-minimax-h3-native-extension-gate"
EXPECTED_INFINILM_BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
if [[ -z "${WORK_HOME:-}" ]]; then
    WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
fi
if [[ ! -f "${WORK_HOME}/setup_infinifabric_env.sh" ]]; then
    printf 'ERROR: environment file is missing: %s\n' "${WORK_HOME}/setup_infinifabric_env.sh" >&2
    exit 1
fi
source "${WORK_HOME}/setup_infinifabric_env.sh"

EFFECTIVE_WORK_HOME="${WORK_HOME:-${IFAB_AFS_HOME}/minimax-h3}"
EFFECTIVE_LOG_DIR="${LOG_DIR:-${EFFECTIVE_WORK_HOME}/logs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HELPER="${SCRIPT_DIR}/run_minimax_h3_fl2va_native_smoke.py"
RUNTIME_HELPER="${SCRIPT_DIR}/install_infinilm_private_runtime.py"
HYGON_RUNTIME_HELPER="${SCRIPT_DIR}/prepare_minimax_h3_hygon_runtime.py"
STATE_MANIFEST="${EFFECTIVE_WORK_HOME}/state/infinilm-runtime-private.json"
HYGON_RUNTIME_MANIFEST="${EFFECTIVE_WORK_HOME}/state/minimax-h3-hygon-runtime.json"
INFINILM_SUCCESS_MARKER="${EFFECTIVE_WORK_HOME}/state/infinilm-extension.success"
INFINICORE_PYTHON="${EFFECTIVE_WORK_HOME}/Projects/InfiniCore/python"
INFINILM_PROJECT="${EFFECTIVE_WORK_HOME}/Projects/InfiniLM"
INFINILM_PYTHON="${INFINILM_PROJECT}/python"
OFFICIAL_EXAMPLE="${INFINILM_PROJECT}/examples/minimax_h3.py"
INFINI_ROOT="${EFFECTIVE_WORK_HOME}/.infini"
MODEL_PATH="/public/home/aclsylqidf/shared/MiniMax-H3"
FIRST_FRAME="${EFFECTIVE_WORK_HOME}/inputs/fl2va/Golde33443.jpg"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-run-minimax-h3-native-fl2va-smoke-${RUN_ID}"
RUN_LOG="${RUN_DIR}/run.log"
SUMMARY="${RUN_DIR}/summary.txt"
OUTPUT_DIR="${EFFECTIVE_WORK_HOME}/outputs/fl2va-native-smoke"
OUTPUT_PATH="${OUTPUT_DIR}/fl2va-native-smoke-${RUN_ID}.mp4"
LOCAL_ROOT="${PWD}/.minimax-h3-native-fl2va-smoke-${UID:-unknown}-$$"
LOCAL_TARGET="${LOCAL_ROOT}/site-packages"
PYTHON="${PYTHON:-$(command -v python3)}"

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${LOCAL_ROOT}"
: > "${RUN_LOG}"

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

write_summary() {
    local exit_code="$1"
    {
        printf 'smoke_version=%s\n' "${SMOKE_VERSION}"
        printf 'helper_exit_code=%s\n' "${exit_code}"
        printf 'run_dir=%s\n' "${RUN_DIR}"
        printf 'run_log=%s\n' "${RUN_LOG}"
        printf 'result_json=%s\n' "${RUN_DIR}/minimax-h3-native-fl2va-smoke.json"
        printf 'output_path=%s\n' "${OUTPUT_PATH}"
        if [[ -f "${OUTPUT_PATH}" ]]; then
            printf 'output_bytes=%s\n' "$(stat -c %s "${OUTPUT_PATH}")"
        else
            printf 'output_bytes=0\n'
        fi
    } > "${SUMMARY}"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        log_line "ERROR: required file is missing: $1"
        write_summary 1
        log_line "SUMMARY: ${SUMMARY}"
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        log_line "ERROR: required directory is missing: $1"
        write_summary 1
        log_line "SUMMARY: ${SUMMARY}"
        exit 1
    fi
}

read_marker_value() {
    local key="$1"
    local marker="$2"
    awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "${marker}"
}

log_line "smoke_version=${SMOKE_VERSION}"
log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
log_line "work_home=${EFFECTIVE_WORK_HOME}"
log_line "model_path=${MODEL_PATH}"
log_line "first_frame=${FIRST_FRAME}"
log_line "output_path=${OUTPUT_PATH}"
log_line "run_dir=${RUN_DIR}"

require_file "${HELPER}"
require_file "${RUNTIME_HELPER}"
require_file "${HYGON_RUNTIME_HELPER}"
require_file "${STATE_MANIFEST}"
require_file "${HYGON_RUNTIME_MANIFEST}"
require_file "${INFINILM_SUCCESS_MARKER}"
require_file "${OFFICIAL_EXAMPLE}"
require_file "${FIRST_FRAME}"
require_dir "${INFINICORE_PYTHON}"
require_dir "${INFINILM_PYTHON}"
require_dir "${INFINI_ROOT}/lib"
require_dir "${MODEL_PATH}/diffusers"

marker_build_version="$(read_marker_value build_version "${INFINILM_SUCCESS_MARKER}")"
marker_extension="$(read_marker_value extension "${INFINILM_SUCCESS_MARKER}")"
if [[ "${marker_build_version}" != "${EXPECTED_INFINILM_BUILD_VERSION}" ]]; then
    log_line "ERROR: stale or incompatible InfiniLM extension marker."
    log_line "expected_build_version=${EXPECTED_INFINILM_BUILD_VERSION}"
    log_line "actual_build_version=${marker_build_version:-MISSING}"
    log_line "Run the current build_infinilm_extension.sh successfully before this smoke test."
    write_summary 1
    log_line "SUMMARY: ${SUMMARY}"
    exit 1
fi
if [[ -z "${marker_extension}" ]]; then
    log_line "ERROR: extension path is missing from ${INFINILM_SUCCESS_MARKER}"
    write_summary 1
    log_line "SUMMARY: ${SUMMARY}"
    exit 1
fi
require_file "${marker_extension}"
extension_sha256="$(sha256sum "${marker_extension}" | awk '{print $1}')"
log_line "infinilm_build_version=${marker_build_version}"
log_line "infinilm_extension=${marker_extension}"
log_line "infinilm_extension_sha256=${extension_sha256}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${TORCH_LIB}:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

set +e
timeout --signal=TERM --kill-after=30s 25m \
    "${PYTHON}" -u "${HELPER}" \
    --runtime-helper "${RUNTIME_HELPER}" \
    --hygon-runtime-helper "${HYGON_RUNTIME_HELPER}" \
    --hygon-runtime-manifest "${HYGON_RUNTIME_MANIFEST}" \
    --state-manifest "${STATE_MANIFEST}" \
    --local-target "${LOCAL_TARGET}" \
    --run-dir "${RUN_DIR}" \
    --example "${OFFICIAL_EXAMPLE}" \
    --model-path "${MODEL_PATH}" \
    --first-frame "${FIRST_FRAME}" \
    --output "${OUTPUT_PATH}" \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
HELPER_RC=${PIPESTATUS[0]}
set -e

write_summary "${HELPER_RC}"
if [[ "${HELPER_RC}" -eq 124 ]]; then
    log_line "ERROR: native FL2VA smoke exceeded the 25-minute launcher limit."
fi
if [[ "${HELPER_RC}" -eq 139 ]]; then
    log_line "ERROR: native process terminated with SIGSEGV; inspect the last native_trace START marker and the fatal Python stack above."
fi
log_line "SUMMARY: ${SUMMARY}"
exit "${HELPER_RC}"
