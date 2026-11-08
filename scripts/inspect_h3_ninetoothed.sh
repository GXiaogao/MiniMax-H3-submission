#!/usr/bin/env bash
# Read-only collection. Run after setup_env.sh through run_logged.
set -euo pipefail
: "${WORK_HOME:?Source /root/private_data/setup_env.sh first}"
CORE_DIR="${WORK_HOME}/Projects/InfiniCore"
LM_DIR="${WORK_HOME}/Projects/InfiniLM"
[[ -f "$CORE_DIR/xmake.lua" ]] || { printf 'Missing InfiniCore: %s\n' "$CORE_DIR" >&2; exit 1; }

show_file()
{
    local file="$1" first="${2:-1}" last="${3:-220}"
    printf '\n=== %s (lines %s-%s) ===\n' "$file" "$first" "$last"
    if [[ -f "$file" ]]; then
        sed -n "${first},${last}p" "$file"
    else
        printf 'NOT PRESENT\n'
    fi
}

printf 'time: %s\nhost: %s\n' "$(date -Is)" "$(hostname)"
for repo in "$CORE_DIR" "$LM_DIR"; do
    printf '\n=== Source state: %s ===\n' "$repo"
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short --branch
    git -C "$repo" diff --stat
    git -C "$repo" submodule status --recursive
done

printf '\n=== Python and installed compiler distributions ===\n'
python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import platform
import sys
import sysconfig

print("python:", sys.executable)
print("version:", sys.version)
print("platform:", sysconfig.get_platform(), platform.libc_ver())
for name in ("torch", "numpy", "sympy", "hatchling", "ninetoothed", "triton", "ntops"):
    try:
        dist = metadata.distribution(name)
        print(f"{name}: {dist.version}; location: {dist.locate_file('')}")
        if name in ("ninetoothed", "triton", "ntops"):
            for requirement in dist.requires or []:
                print("  requires:", requirement)
    except metadata.PackageNotFoundError:
        print(name + ": no distribution with this name")
    spec = importlib.util.find_spec(name)
    print("  module:", spec.origin if spec else "NOT FOUND")

print("Vendor-named compiler distributions:")
for dist in metadata.distributions():
    name = dist.metadata.get("Name", "")
    if any(word in name.lower() for word in ("triton", "ninetoothed")):
        print(" ", name, dist.version, dist.locate_file(""))
PY

for project in ntops ninetoothed; do
    directory="${WORK_HOME}/Projects/${project}"
    printf '\n=== Optional source checkout: %s ===\n' "$directory"
    if [[ -d "$directory" ]]; then
        if [[ -e "$directory/.git" ]]; then
            git -C "$directory" rev-parse HEAD
            git -C "$directory" status --short --branch
        fi
        show_file "$directory/pyproject.toml"
    else
        printf 'NOT PRESENT\n'
    fi
done

show_file "$CORE_DIR/pyproject.toml"
show_file "$CORE_DIR/requirements.txt"
show_file "$CORE_DIR/scripts/build_ntops.py"
show_file "$CORE_DIR/src/infiniop/ninetoothed/build.py"
show_file "$CORE_DIR/src/infiniop/ninetoothed/utils.h"
show_file "$CORE_DIR/src/infiniop/ops/swiglu/ninetoothed/build.py"
show_file "$CORE_DIR/src/infiniop/ops/swiglu/ninetoothed/swiglu.py"
show_file "$CORE_DIR/src/infiniop/ops/swiglu/operator.cc"
show_file "$CORE_DIR/xmake/hygon.lua" 130 185
show_file "$CORE_DIR/xmake.lua" 240 275

printf '\n=== Dependency declarations and installation notes ===\n'
cd "$CORE_DIR"
if command -v rg >/dev/null 2>&1; then
    rg -n -i 'ninetoothed|triton|九齿' --glob '*README*' --glob '*requirements*' \
        --glob 'pyproject.toml' --glob '*lock*' --glob '*hygon*' . || {
        rc=$?
        (( rc == 1 )) || exit "$rc"
    }
else
    printf 'rg not installed; inspecting top-level documentation only.\n'
    for file in README* requirements*; do
        [[ -f "$file" ]] || continue
        grep -nEi 'ninetoothed|triton|九齿' "$file" || {
            rc=$?
            (( rc == 1 )) || exit "$rc"
        }
    done
fi

printf '\n=== Compiler-related local archives and DTK entries (depth <= 4) ===\n'
for directory in "$WORK_HOME/wheels" "$WORK_HOME/packages" /opt/dtk; do
    [[ -d "$directory" ]] || continue
    printf '\nDirectory: %s\n' "$directory"
    # Follow only the starting directory if DTK is a symlink; do not traverse
    # arbitrary symlink trees. This lists paths without reading archive contents.
    find -H "$directory" -maxdepth 4 \
        \( -iname '*triton*' -o -iname '*ninetoothed*' -o -iname '*ntops*' \) \
        -print || printf 'WARNING: could not list every entry under %s\n' "$directory"
done
printf '\nCollection complete. This is not a compiler or GPU execution test.\n'
