#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
if [[ "${1:-}" != '--verify-only' ]]; then
    bash build.sh
else
    shift
fi
export INFINICORE_H3_GATED_BUNDLE
INFINICORE_H3_GATED_BUNDLE="$(cat "$WORK_HOME/logs/h3_gated_active_bundle.txt")"
exec python -X faulthandler -u verify_integration.py "$@"
