#!/usr/bin/env bash
# Run through run_logged after setup_env.sh. Does not activate in the parent shell.
set -euo pipefail
: "${WORK_HOME:?Source setup_env.sh first}"
VENV_DIR="$WORK_HOME/venvs/h3-ninetoothed-026"
WHEEL_DIR="$WORK_HOME/wheels"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_FILES=(
    'triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
    'ninetoothed-0.26.0-py3-none-any.whl'
    'numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
    'sympy-1.14.0-py3-none-any.whl'
    'mpmath-1.3.0-py3-none-any.whl'
)

python - "$WHEEL_DIR" "${WHEEL_FILES[@]}" <<'PY'
import hashlib, json, pathlib, sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('This wheel set requires Python 3.11.')
root = pathlib.Path(sys.argv[1])
entries = json.loads((root / 'h3_toolchain_manifest.json').read_text())
expected = set(sys.argv[2:])
if {entry['file'] for entry in entries} != expected or len(entries) != len(expected):
    raise SystemExit('Unexpected toolchain manifest contents.')
for entry in entries:
    filename = entry['file']
    if pathlib.Path(filename).name != filename:
        raise SystemExit('Manifest must contain filenames, not paths.')
    path = root / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit(f'Checksum mismatch: {path}')
    print('SHA256 OK:', filename)
PY

python - <<'PY'
import importlib.metadata as m
import torch
from packaging.version import Version
if Version(m.version('torch')).base_version != '2.9.0' or not torch.version.hip:
    raise SystemExit('This Triton wheel requires the existing HIP Torch 2.9.0 environment.')
PY
BASE_TORCH="$(python -c 'import importlib.metadata as m, torch; print(m.version("torch"), torch.__file__)')"
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    [[ ! -e "$VENV_DIR" ]] || { printf 'Refusing to reuse non-venv directory: %s\n' "$VENV_DIR" >&2; exit 1; }
    python -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/python" - "$VENV_DIR" <<'PY'
import pathlib, sys
if pathlib.Path(sys.prefix).resolve() != pathlib.Path(sys.argv[1]).resolve() or sys.prefix == sys.base_prefix:
    raise SystemExit('Expected isolated venv interpreter.')
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Existing venv has wrong Python version.')
PY

# Install the exact verified files, even if wheels/ holds other package versions.
WHEEL_PATHS=()
for wheel in "${WHEEL_FILES[@]}"; do WHEEL_PATHS+=("$WHEEL_DIR/$wheel"); done
# --no-deps prevents replacing the vendor Torch through dependency resolution.
# Ignore inherited pip redirects so the destination remains this venv.
env -u PIP_TARGET -u PIP_PREFIX -u PIP_USER -u PIP_PYTHON -u PIP_ROOT PIP_CONFIG_FILE=/dev/null \
    "$VENV_DIR/bin/python" -m pip --isolated install --no-index --no-deps "${WHEEL_PATHS[@]}"
VENV_TORCH="$("$VENV_DIR/bin/python" -c 'import importlib.metadata as m, torch; print(m.version("torch"), torch.__file__)')"
[[ "$BASE_TORCH" == "$VENV_TORCH" ]] || { printf 'Torch source/version changed: base=%s venv=%s\n' "$BASE_TORCH" "$VENV_TORCH" >&2; exit 1; }
"$VENV_DIR/bin/python" - "$WHEEL_DIR" <<'PY'
import importlib.metadata as m
from packaging.requirements import Requirement
from packaging.markers import default_environment
env = {**default_environment(), 'extra': ''}
for name in ('triton', 'ninetoothed', 'numpy', 'sympy', 'mpmath'):
    print(name, m.version(name))
    for text in m.requires(name) or ():
        req = Requirement(text)
        if req.marker and not req.marker.evaluate(env):
            continue
        version = m.version(req.name)
        if req.specifier and version not in req.specifier:
            raise SystemExit(f'Incompatible dependency: {name}: {req}; installed {version}')
print('Toolchain dependencies checked. GPU execution has NOT been tested yet.')
PY
printf '\nNext: run_logged h3_toolchain_verify.log "%s/bin/python" -u "%s/verify_h3_toolchain.py" --jit-only\n' "$VENV_DIR" "$SCRIPT_DIR"
