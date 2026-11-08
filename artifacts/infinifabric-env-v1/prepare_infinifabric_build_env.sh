#!/usr/bin/env bash

set -Eeuo pipefail

readonly PREPARE_VERSION="1.1-xmake-global-root"

printf 'prepare_version=%s\n' "${PREPARE_VERSION}"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
LOG_DIR="${WORK_HOME}/logs"
ENV_FILE="${WORK_HOME}/setup_infinifabric_env.sh"
ENV_FILE_TMP="${ENV_FILE}.tmp.$$"
LOG_FILE="${LOG_DIR}/prepare_infinifabric_build_env.log"

mkdir -p "${LOG_DIR}"

cat > "${ENV_FILE_TMP}" <<'ENV_EOF'
#!/usr/bin/env bash

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf '[setup_infinifabric_env] ERROR: IFAB_AFS_HOME is not set.\n' >&2
    return 1
fi

export WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
export LOG_DIR="${WORK_HOME}/logs"
export PKG_DIR="${WORK_HOME}/packages"
# XMAKE_GLOBALDIR is the parent under which xmake creates .xmake.
# Pointing it at ${WORK_HOME}/.xmake would incorrectly produce
# ${WORK_HOME}/.xmake/.xmake.
export XMAKE_GLOBALDIR="${WORK_HOME}"
export XMAKE_STORAGE_DIR="${WORK_HOME}/.xmake"
export INFINI_ROOT="${WORK_HOME}/.infini"
export XMAKE_ROOT=y

if [[ ! -f /opt/dtk/env.sh ]]; then
    printf '[setup_infinifabric_env] ERROR: /opt/dtk/env.sh is missing.\n' >&2
    return 1
fi

# Vendor environment scripts may read unset optional variables.
set +u
source /opt/dtk/env.sh
set -u

export PATH="${WORK_HOME}/tools/xmake:${PATH}"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
    "${LOG_DIR}" \
    "${PKG_DIR}" \
    "${XMAKE_STORAGE_DIR}" \
    "${INFINI_ROOT}"

umask 0002
ENV_EOF

mv -f -- "${ENV_FILE_TMP}" "${ENV_FILE}"
chmod 0644 "${ENV_FILE}"

set +e
(
    set -Eeuo pipefail

    printf 'prepare_version=%s\n' "${PREPARE_VERSION}"
    printf 'time=%s\n' "$(date -Is)"
    printf 'host=%s\n' "$(hostname)"
    printf 'uid=%s\n' "$(id -u)"
    printf 'environment_file=%s\n' "${ENV_FILE}"

    source "${ENV_FILE}"

    printf 'WORK_HOME=%s\n' "${WORK_HOME}"
    printf 'LOG_DIR=%s\n' "${LOG_DIR}"
    printf 'PKG_DIR=%s\n' "${PKG_DIR}"
    printf 'XMAKE_GLOBALDIR=%s\n' "${XMAKE_GLOBALDIR}"
    printf 'XMAKE_STORAGE_DIR=%s\n' "${XMAKE_STORAGE_DIR}"
    printf 'INFINI_ROOT=%s\n' "${INFINI_ROOT}"

    command -v xmake
    xmake --version
    xmake l os.programdir

    xmake g \
        --network=private \
        --pkg_searchdirs="${PKG_DIR}"

    command -v hipcc
    hipcc --version

    test -f "${PKG_DIR}/boost-1.92.0-cmake.tar.xz"
    test -f "${PKG_DIR}/pybind11-v3.1.0.zip"
    test -f "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/b/boost/xmake.lua"
    test -f "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/p/pybind11/xmake.lua"

    grep -Fq \
        'boost-$(version)-cmake.tar.xz' \
        "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/b/boost/xmake.lua"
    grep -Fq \
        '9bed76128d4e46755dbe818487788c6fceb6f72b378f4daa49b7e1e600d9088d' \
        "${XMAKE_STORAGE_DIR}/repositories/xmake-repo/packages/b/boost/xmake.lua"

    python3 - <<'PY'
import os
import torch

count = torch.cuda.device_count()
print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("DCU available:", torch.cuda.is_available())
print("device count:", count)
for index in range(count):
    print(f"device {index}:", torch.cuda.get_device_name(index))

if not torch.cuda.is_available():
    raise SystemExit("DCU is not available")
if count != 8:
    raise SystemExit(f"Expected 8 devices, found {count}")
PY

    printf 'offline_packages=OK\n'
    printf 'repository_recipes=OK\n'
    printf 'eight_card_visibility=OK\n'
    printf 'environment preparation: PASSED\n'
) 2>&1 | tee "${LOG_FILE}"
prepare_rc=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "${prepare_rc}" | tee -a "${LOG_FILE}"
printf 'log=%s\n' "${LOG_FILE}"
exit "${prepare_rc}"
