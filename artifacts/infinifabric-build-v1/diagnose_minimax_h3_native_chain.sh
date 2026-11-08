#!/usr/bin/env bash
# Reuse the tested preparation, fixed workload and failure-retention boundary.
set -Eeuo pipefail
CHAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${CHAIN_SCRIPT_DIR}/run_minimax_h3_fl2va_native_quality_probe.sh"
NATIVE_PROBE_HELPER_NAME="diagnose_minimax_h3_native_chain.py"
NATIVE_PROBE_LABEL="native-chain-trace"
main "$@"
