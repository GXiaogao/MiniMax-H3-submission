#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR_LOCAL="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

source "${SCRIPT_DIR_LOCAL}/setup_env.sh"

LOG_FILE="${LOG_DIR}/infinifabric_preflight.log"

{
    echo "time: $(date -Is)"
    echo "host: $(hostname)"
    echo "user: $(id)"
    echo "WORK_HOME=${WORK_HOME}"
    echo "IFAB_AFS_HOME=${IFAB_AFS_HOME}"
    echo

    echo "=== tools ==="
    command -v python3 || true
    command -v torchrun || true
    command -v gcc || true
    command -v g++ || true
    command -v cmake || true
    command -v ninja || true
    command -v hipcc || true
    command -v hipconfig || true
    command -v xmake || true
    hipcc --version 2>/dev/null || true
    xmake --version 2>/dev/null || true
    echo

    echo "=== Python and devices ==="
    python3 - <<'PY'
import importlib.util
import json
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    row = {
        "index": index,
        "name": props.name,
        "total_memory": props.total_memory,
        "multi_processor_count": props.multi_processor_count,
        "gcnArchName": getattr(props, "gcnArchName", None),
    }
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))

for name in ("infinicore", "infinilm", "transformers", "diffusers", "accelerate"):
    print(f"module {name}:", importlib.util.find_spec(name))
PY
    echo

    echo "=== migrated source ==="
    for project in InfiniCore InfiniLM; do
        project_path="${PROJECTS_DIR}/${project}"
        if [[ -d "${project_path}" ]]; then
            echo "${project}: PRESENT (${project_path})"
            du -sh "${project_path}"
        else
            echo "${project}: MISSING (${project_path})"
        fi
    done
} 2>&1 | tee "${LOG_FILE}"

echo "preflight log: ${LOG_FILE}"
