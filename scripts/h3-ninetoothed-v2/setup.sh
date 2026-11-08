#!/usr/bin/env bash
# Source WORK_HOME/setup_env.sh first. Only this candidate venv is modified.
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$WORK_HOME/venvs/h3-ninetoothed-026-triton380"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
python "$SCRIPT_DIR/setup.py" --venv "$VENV_DIR" --wheels "$WORK_HOME/wheels"
