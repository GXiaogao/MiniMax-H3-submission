#!/usr/bin/env bash
# Run after source "$WORK_HOME/setup_env.sh". No package installs or config reset.
set -euo pipefail
: "${WORK_HOME:?source setup_env.sh first}"
: "${LOG_DIR:?source setup_env.sh first}"
: "${INFINI_ROOT:?source setup_env.sh first}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$WORK_HOME/Projects/InfiniCore"
jobs="${H3_BUILD_JOBS:-8}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'H3_BUILD_JOBS must be positive'; exit 2; }
[[ "$LOG_DIR" == "$WORK_HOME/logs" ]] || { echo 'LOG_DIR must be WORK_HOME/logs'; exit 2; }
mkdir -p "$LOG_DIR"
command -v xmake
cd "$repo"

python - <<'PY'
import os, re, sys
from pathlib import Path
root = Path(os.environ['WORK_HOME'])
config = root / 'Projects/InfiniCore/.xmake/linux/x86_64/xmake.conf'
text = config.read_text()
def option(key):
    pattern = r'(?:\["' + re.escape(key) + r'"\]|\b' + re.escape(key) + r')\s*=\s*([^,\n]+)'
    match = re.search(pattern, text)
    return match.group(1).strip().strip('"') if match else None
expected = {'hygon-dcu': 'true', 'hygon-arch': 'gfx906', 'flash-attn': 'false',
            'ninetoothed': 'false', 'infiniops': 'false', 'mode': 'release'}
for key, value in expected.items():
    actual = option(key)
    print(f'{key}: {actual}')
    if actual != value:
        raise RuntimeError(f'Expected {key}={value}; configuration is preserved, not rewritten.')
print('python:', sys.executable)
if sys.prefix != sys.base_prefix:
    raise RuntimeError('Use the normal competition Python, not the NineToothed/other venv.')
import torch, infinicore
if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError('Hygon runtime is not available')
print('torch:', torch.__version__, 'HIP:', torch.version.hip)
print('infinicore:', infinicore.__file__)
PY

python "$script_dir/apply_h3_mha_patch.py" --repo "$repo"
python "$script_dir/check_h3_mha_layout.py" --repo "$repo"
g++ -std=c++17 -DENABLE_HYGON_API -I include -I third_party/spdlog/include -fsyntax-only \
    src/infinicore/ops/multi_head_attention/mha.cc \
    src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc
xmake build -j "$jobs" infinicore_cpp_api
# Keep the old installed shared library as a concrete recovery artifact.
python - <<'PY'
import os, shutil, time
from pathlib import Path
source = Path(os.environ['INFINI_ROOT']) / 'lib/libinfinicore_cpp_api.so'
if source.is_file():
    folder = Path(os.environ['WORK_HOME']) / 'Projects/InfiniCore/.h3_mha_backups'
    folder.mkdir(exist_ok=True)
    target = folder / ('libinfinicore_cpp_api_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()) + '.so')
    shutil.copy2(source, target)
    print('Previous installed library saved:', target)
PY
xmake install infinicore_cpp_api
python -u "$script_dir/inspect_h3_mha_registration.py"
printf 'Build/install/registration completed. Numerical tests are still required.\n'
