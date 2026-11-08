#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0 2>/dev/null || true
unset H3_BOUNDARY_TRACE_DIR
if [[ "$#" -ne 0 ]]; then
    printf 'ERROR: Native template generation accepts no program arguments.\n' >&2
    exit 2
fi
if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi
ROOT_WORK="${WORK_HOME:-${IFAB_AFS_HOME}/minimax-h3}"
source "${ROOT_WORK}/setup_infinifabric_env.sh"
ROOT_WORK="${WORK_HOME:-${ROOT_WORK}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${ROOT_WORK}/logs/infinifabric-h3-load-sync-generation-${RUN_ID}"
LOCAL_ROOT="${PWD}/.minimax-h3-load-sync-generation-${UID:-unknown}-$$"
PYTHON="${PYTHON:-$(command -v python3)}"
mkdir -p "${RUN_DIR}" "${LOCAL_ROOT}"
RUN_LOG="${RUN_DIR}/run.log"
: > "${RUN_LOG}"
LAST_STAGE="preflight"
finish() {
    local rc="$?"
    trap - EXIT
    printf 'exit_code=%s\nlast_stage=%s\nrun_dir=%s\nquality_status=UNASSESSED\nperformance_measurement_valid=false\n' \
        "${rc}" "${LAST_STAGE}" "${RUN_DIR}" > "${RUN_DIR}/summary.txt"
    printf 'SUMMARY: %s/summary.txt\n' "${RUN_DIR}" | tee -a "${RUN_LOG}"
    exit "${rc}"
}
trap finish EXIT
for name in run_minimax_h3_load_sync_generation.py diagnose_minimax_h3_full_forward_load_sync.py \
    diagnose_minimax_h3_load_sync.py diagnose_minimax_h3_prefix_parity.py diagnose_minimax_h3_transformer_parity.py \
    run_minimax_h3_fl2va_native_smoke.py prepare_minimax_h3_vae_dtype_probe.py diagnose_minimax_h3_vae_roundtrip.py \
    install_infinilm_private_runtime.py prepare_minimax_h3_hygon_runtime.py minimax-h3-template-first-frame.png; do
    if [[ ! -f "${SCRIPT_DIR}/${name}" ]]; then
        printf 'ERROR: required package file missing: %s\n' "${name}" | tee -a "${RUN_LOG}"
        exit 1
    fi
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if TORCH_LIB="$(timeout 60s "${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")' 2> "${RUN_DIR}/torch-preflight.log")"; then
    cat "${RUN_DIR}/torch-preflight.log" | tee -a "${RUN_LOG}"
else
    rc="$?"
    cat "${RUN_DIR}/torch-preflight.log" | tee -a "${RUN_LOG}"
    exit "${rc}"
fi
export LD_LIBRARY_PATH="${ROOT_WORK}/.infini/lib:${TORCH_LIB}:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_WORK}/Projects/InfiniCore/python:${ROOT_WORK}/Projects/InfiniLM/python:${PYTHONPATH:-}"
printf 'run_dir=%s\nworkload=Native FL2VA template PNG, 50 steps, 124 frames, seed42, default canvas\n' "${RUN_DIR}" | tee -a "${RUN_LOG}"
LAST_STAGE="full-Native-generation"
if timeout --signal=TERM --kill-after=30s 25m "${PYTHON}" -u "${SCRIPT_DIR}/run_minimax_h3_load_sync_generation.py" \
    --run-dir "${RUN_DIR}" --local-target "${LOCAL_ROOT}/site-packages" \
    --model-path /public/home/aclsylqidf/shared/MiniMax-H3 \
    --work-home "${ROOT_WORK}" 2>&1 | tee -a "${RUN_LOG}"; then
    LAST_STAGE="complete"
else
    statuses=("${PIPESTATUS[@]}")
    rc="${statuses[0]}"
    if [[ "${rc}" -eq 0 ]]; then rc="${statuses[1]}"; fi
    exit "${rc}"
fi
