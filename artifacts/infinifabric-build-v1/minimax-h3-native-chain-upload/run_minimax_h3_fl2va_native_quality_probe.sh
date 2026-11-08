#!/usr/bin/env bash

# Callable separately so the exact launch boundary can be checked without GPUs.
invoke_native_helper() {
    timeout --signal=TERM --kill-after=30s 25m \
        "${PYTHON}" -u "${HELPER}" \
        --runtime-helper "${RUNTIME_HELPER}" \
        --hygon-runtime-helper "${HYGON_RUNTIME_HELPER}" \
        --hygon-runtime-manifest "${HYGON_RUNTIME_MANIFEST}" \
        --state-manifest "${STATE_MANIFEST}" \
        --local-target "${LOCAL_TARGET}" \
        --run-dir "${RUN_DIR}" \
        --example "${PATCHED_EXAMPLE}" \
        --model-path "${MODEL_PATH}" \
        --first-frame "${FIRST_FRAME}" \
        --output "${OUTPUT_PATH}" \
        --run-profile=formal \
        --text="A dog falls asleep" \
        --gpus=0,1,2,3,4,5,6,7 \
        --num-frames=124 \
        --steps=10 \
        --seed=42 \
        --height=384 \
        --width=576
}

probe_torch_library() {
    local probe_rc=0
    if TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")' \
        2> "${RUN_DIR}/torch-preflight.log")"; then
        probe_rc=0
    else
        probe_rc="$?"
    fi
    cat "${RUN_DIR}/torch-preflight.log" | tee -a "${RUN_LOG}"
    return "${probe_rc}"
}

main() (
    set -Eeuo pipefail
    PROBE_VERSION="1.0-native-fl2va-preserve-vae-fp32"
    EXPECTED_INFINILM_BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"
    if [[ "$#" -ne 0 ]]; then
        printf 'ERROR: quality probe accepts no program arguments; its comparison workload is fixed.\n' >&2
        exit 2
    fi
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
    HELPER="${SCRIPT_DIR}/${NATIVE_PROBE_HELPER_NAME:-run_minimax_h3_fl2va_native_smoke.py}"
    PROBE_LABEL="${NATIVE_PROBE_LABEL:-quality-probe}"
    PATCH_HELPER="${SCRIPT_DIR}/prepare_minimax_h3_vae_dtype_probe.py"
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
    RUN_DIR="${EFFECTIVE_LOG_DIR}/infinifabric-run-minimax-h3-native-fl2va-${PROBE_LABEL}-${RUN_ID}"
    RUN_LOG="${RUN_DIR}/run.log"
    SUMMARY="${RUN_DIR}/summary.txt"
    RESULT_JSON="${RUN_DIR}/minimax-h3-native-fl2va-formal.json"
    PATCH_METADATA="${RUN_DIR}/vae-dtype-patch.json"
    PATCH_LOG="${RUN_DIR}/vae-dtype-patch.log"
    OUTPUT_DIR="${EFFECTIVE_WORK_HOME}/outputs/fl2va-native-${PROBE_LABEL}"
    OUTPUT_PATH="${OUTPUT_DIR}/fl2va-native-${PROBE_LABEL}-${RUN_ID}.mp4"
    LOCAL_ROOT="${PWD}/.minimax-h3-native-quality-probe-${UID:-unknown}-$$"
    LOCAL_TARGET="${LOCAL_ROOT}/site-packages"
    PATCHED_EXAMPLE="${LOCAL_ROOT}/example/minimax_h3.py"
    PYTHON="${PYTHON:-$(command -v python3)}"
    LAST_STAGE="preflight"
    mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${LOCAL_ROOT}"
    : > "${RUN_LOG}"

    log_line() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }
    finish() {
        local rc="$?"
        trap - EXIT
        {
            printf 'probe_version=%s\nexit_code=%s\nlast_stage=%s\n' "${PROBE_VERSION}" "${rc}" "${LAST_STAGE}"
            printf 'validation_scope=media-structure-only\nquality_status=UNASSESSED\n'
            printf 'run_dir=%s\nrun_log=%s\nresult_json=%s\n' "${RUN_DIR}" "${RUN_LOG}" "${RESULT_JSON}"
            printf 'patch_metadata=%s\noutput_path=%s\n' "${PATCH_METADATA}" "${OUTPUT_PATH}"
            if [[ -f "${OUTPUT_PATH}" ]]; then
                printf 'output_bytes=%s\n' "$(stat -c %s "${OUTPUT_PATH}")"
            else
                printf 'output_bytes=0\n'
            fi
        } > "${SUMMARY}"
        log_line "SUMMARY: ${SUMMARY}"
        exit "${rc}"
    }
    trap finish EXIT
    require_file() {
        if [[ ! -f "$1" ]]; then
            log_line "ERROR: required file is missing: $1"
            exit 1
        fi
    }
    require_dir() {
        if [[ ! -d "$1" ]]; then
            log_line "ERROR: required directory is missing: $1"
            exit 1
        fi
    }
    read_marker_value() {
        awk -F= -v wanted="$1" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "$2"
    }

    log_line "probe_version=${PROBE_VERSION}"
    log_line "time=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
    log_line "work_home=${EFFECTIVE_WORK_HOME}"
    log_line "model_path=${MODEL_PATH}"
    log_line "first_frame=${FIRST_FRAME}"
    log_line "output_path=${OUTPUT_PATH}"
    log_line "run_dir=${RUN_DIR}"
    log_line "diagnostic_workload=fl2va gpus=0,1,2,3,4,5,6,7 frames=124 steps=10 seed=42 resolution=576x384"
    log_line "validation_scope=media-structure-only; quality_status=UNASSESSED"

    for path in "${HELPER}" "${PATCH_HELPER}" "${RUNTIME_HELPER}" "${HYGON_RUNTIME_HELPER}" \
        "${STATE_MANIFEST}" "${HYGON_RUNTIME_MANIFEST}" "${INFINILM_SUCCESS_MARKER}" \
        "${OFFICIAL_EXAMPLE}" "${FIRST_FRAME}"; do
        require_file "${path}"
    done
    marker_build_version="$(read_marker_value build_version "${INFINILM_SUCCESS_MARKER}")"
    marker_extension="$(read_marker_value extension "${INFINILM_SUCCESS_MARKER}")"
    if [[ "${marker_build_version}" != "${EXPECTED_INFINILM_BUILD_VERSION}" ]]; then
        log_line "ERROR: stale or incompatible InfiniLM extension marker: ${marker_build_version:-MISSING}"
        log_line "expected_build_version=${EXPECTED_INFINILM_BUILD_VERSION}"
        exit 1
    fi
    require_file "${marker_extension}"
    log_line "infinilm_build_version=${marker_build_version}"
    log_line "infinilm_extension=${marker_extension}"
    extension_sha256="$(sha256sum "${marker_extension}" | awk '{print $1}')"
    log_line "infinilm_extension_sha256=${extension_sha256}"

    LAST_STAGE="prepare-vae-dtype-probe"
    log_line "stage=${LAST_STAGE} status=START"
    if timeout --signal=TERM --kill-after=5s 60s "${PYTHON}" "${PATCH_HELPER}" \
        --source "${OFFICIAL_EXAMPLE}" --destination "${PATCHED_EXAMPLE}" \
        > "${PATCH_METADATA}" 2> "${PATCH_LOG}"; then
        cat "${PATCH_METADATA}" | tee -a "${RUN_LOG}"
    else
        patch_rc="$?"
        cat "${PATCH_LOG}" | tee -a "${RUN_LOG}"
        log_line "ERROR: VAE precision probe preparation failed with exit_code=${patch_rc}"
        exit "${patch_rc}"
    fi
    # Keep the small diagnostic source alongside the hashes after local storage expires.
    cp -- "${PATCHED_EXAMPLE}" "${RUN_DIR}/minimax_h3.vae-probe.py"
    log_line "stage=${LAST_STAGE} status=DONE"

    LAST_STAGE="runtime-preflight"
    require_dir "${INFINICORE_PYTHON}"
    require_dir "${INFINILM_PYTHON}"
    require_dir "${INFINI_ROOT}/lib"
    require_dir "${MODEL_PATH}/diffusers"
    export PYTHONUNBUFFERED=1
    export PYTHONFAULTHANDLER=1
    probe_torch_library
    export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${TORCH_LIB}:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${INFINICORE_PYTHON}:${INFINILM_PYTHON}:${PYTHONPATH:-}"
    export PYTHONDONTWRITEBYTECODE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false

    LAST_STAGE="native-fl2va-quality-probe"
    if invoke_native_helper 2>&1 | tee -a "${RUN_LOG}"; then
        HELPER_RC=0
    else
        # A tee/disk failure also fails the run, even if Python returned zero.
        helper_status=("${PIPESTATUS[@]}")
        HELPER_RC="${helper_status[0]}"
        if [[ "${HELPER_RC}" -eq 0 ]]; then HELPER_RC="${helper_status[1]}"; fi
    fi
    if [[ "${HELPER_RC}" -eq 124 ]]; then
        log_line "ERROR: quality probe exceeded the 25-minute launcher limit."
    elif [[ "${HELPER_RC}" -eq 139 ]]; then
        log_line "ERROR: native process terminated with SIGSEGV; inspect the last native_trace START marker."
    elif [[ "${HELPER_RC}" -eq 0 ]]; then
        LAST_STAGE="complete"
        log_line "QUALITY_PROBE RUN_COMPLETE; quality_status=UNASSESSED; inspect output frames before judging correctness."
    fi
    exit "${HELPER_RC}"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
