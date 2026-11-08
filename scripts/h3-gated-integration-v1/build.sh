#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
: "${INFINI_ROOT:?Source setup_env.sh first}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PACKAGE_DIR="$PWD"
export LOG_DIR="$WORK_HOME/logs/h3_gated_build_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
JOBS="${H3_BUILD_JOBS:-8}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid H3_BUILD_JOBS'; exit 2; }
step() { local name="$1"; shift; "$@" 2>&1 | tee "$LOG_DIR/$name.log"; }
sha256sum -c SHA256SUMS
step source_preflight python apply_patch.py --check
step bundle "$WORK_HOME/venvs/h3-ninetoothed-026-triton380/bin/python" -u prepare_bundle.py
step backup python backup_install.py
step source_apply python apply_patch.py
cd "$WORK_HOME/Projects/InfiniCore"
step core_config xmake f -y
step core_build xmake build -j "$JOBS" infinicore_cpp_api
step core_install xmake install infinicore_cpp_api
step core_binding_build xmake build -j "$JOBS" _infinicore
step core_binding_install xmake install _infinicore
cd "$WORK_HOME/Projects/InfiniLM"
step lm_config xmake f -y
step lm_build xmake build -j "$JOBS" _infinilm
step lm_install xmake install _infinilm
cd "$PACKAGE_DIR"
step symbols python -c 'import infinicore; from infinicore.lib import _infinicore as n; assert hasattr(infinicore,"gated_residual"); print("native counter:", n.gated_residual_hygon_launch_count()); print("New InfiniCore symbols: OK")'
cp "$LOG_DIR/bundle_path.txt" "$WORK_HOME/logs/h3_gated_active_bundle.txt"
echo "Build logs: $LOG_DIR"
echo 'Build/install completed. Operator/model correctness still requires verify_integration.py.'
