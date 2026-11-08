#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c SHA256SUMS
# Use the normal initialized Python. No pip, xmake, compilation or bundle switch.
python -X faulthandler -u benchmark.py "$@"
