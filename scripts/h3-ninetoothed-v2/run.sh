#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
sha256sum -c SHA256SUMS
bash ./setup.sh
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
"$WORK_HOME/venvs/h3-ninetoothed-026-triton380/bin/python" -X faulthandler -u ./verify.py --logs "$WORK_HOME/logs"
