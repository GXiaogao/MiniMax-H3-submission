#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
exec "$WORK_HOME/venvs/h3-ninetoothed-026-triton380/bin/python" -X faulthandler -u verify_native.py "$@"
