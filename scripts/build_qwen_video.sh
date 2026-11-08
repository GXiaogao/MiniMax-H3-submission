#!/usr/bin/env bash
# Run after source "$WORK_HOME/setup_env.sh"; keep existing dependency/build config.
set -euo pipefail
: "${WORK_HOME:?source setup_env.sh first}"
: "${LOG_DIR:?source setup_env.sh first}"
: "${INFINI_ROOT:?source setup_env.sh first}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$WORK_HOME/Projects/InfiniLM"
jobs="${H3_BUILD_JOBS:-8}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'H3_BUILD_JOBS must be positive'; exit 2; }
[[ "$LOG_DIR" == "$WORK_HOME/logs" ]] || { echo 'LOG_DIR must be WORK_HOME/logs'; exit 2; }
mkdir -p "$LOG_DIR"
command -v xmake
cd "$repo"

python - <<'PY'
import hashlib, os, re, sys
from pathlib import Path
repo = Path(os.environ['WORK_HOME']) / 'Projects/InfiniLM'
config = repo / '.xmake/linux/x86_64/xmake.conf'
print('Existing xmake config (preserved):', config)
config_text = config.read_text()
print(config_text)
for key, expected in {'plat': 'linux', 'arch': 'x86_64', 'mode': 'release', 'builddir': 'build'}.items():
    match = re.search(r'\b' + key + r'\s*=\s*"([^\"]*)"', config_text)
    if not match or match.group(1) != expected:
        raise RuntimeError(f'Expected {key}={expected}; inspect the existing config before building.')
print('python:', sys.executable)
if sys.prefix != sys.base_prefix:
    raise RuntimeError('Use the competition Python, not the toolchain venv.')
import torch, infinicore, infinilm
from infinilm.lib import _infinilm
if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError('Hygon runtime is not available')
extension = Path(_infinilm.__file__).resolve()
if not extension.is_relative_to((repo / 'python/infinilm/lib').resolve()):
    raise RuntimeError(f'Unexpected extension location: {extension}')
print('torch:', torch.__version__, 'HIP:', torch.version.hip)
print('infinicore:', infinicore.__file__)
print('existing extension:', extension)
print('existing extension sha256:', hashlib.sha256(extension.read_bytes()).hexdigest())
PY

python "$script_dir/apply_qwen_video_patch.py" --repo "$repo"
python "$script_dir/check_qwen_video_layout.py" --repo "$repo"
xmake build -j "$jobs" _infinilm

# Save the installed extension before xmake replaces it. No pip reinstall needed.
python - <<'PY'
import hashlib, json, os, shutil, time
from pathlib import Path
import torch, infinicore, infinilm
from infinilm.lib import _infinilm
source = Path(_infinilm.__file__).resolve()
repo = Path(os.environ['WORK_HOME']) / 'Projects/InfiniLM'
if not source.is_relative_to((repo / 'python/infinilm/lib').resolve()):
    raise RuntimeError(f'Unexpected extension location: {source}')
folder = repo / '.qwen_video_backups' / (time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
folder.mkdir(parents=True)
target = folder / source.name
shutil.copy2(source, target)
record = {'installed_path': str(source), 'backup_path': str(target),
          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
(folder/'backup.json').write_text(json.dumps(record, indent=2))
print('Previous extension backup:', target)
PY

xmake install _infinilm
python - <<'PY'
import hashlib, os
from pathlib import Path
import torch, infinicore, infinilm
from infinilm.lib import _infinilm
repo = Path(os.environ['WORK_HOME']) / 'Projects/InfiniLM'
installed = Path(_infinilm.__file__).resolve()
if not installed.is_relative_to((repo/'python/infinilm/lib').resolve()):
    raise RuntimeError(f'Unexpected extension location: {installed}')
built = repo / 'build/linux/x86_64/release' / installed.name
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if digest(installed) != digest(built):
    raise RuntimeError('Loaded extension does not match the newly built library')
print('installed extension:', installed)
print('installed extension sha256:', digest(installed))
PY
printf 'Build/install/import completed. Run the video and image/text numerical tests next.\n'
