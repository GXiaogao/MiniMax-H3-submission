#!/usr/bin/env bash

set -Eeuo pipefail
umask 0022

readonly BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"

if [[ -z "${IFAB_AFS_HOME:-}" ]]; then
    printf 'ERROR: IFAB_AFS_HOME is not set.\n' >&2
    exit 1
fi

WORK_HOME="${IFAB_AFS_HOME}/minimax-h3"
if [[ ! -f "${WORK_HOME}/setup_infinifabric_env.sh" ]]; then
    printf 'ERROR: environment file is missing: %s\n' \
        "${WORK_HOME}/setup_infinifabric_env.sh" >&2
    exit 1
fi
source "${WORK_HOME}/setup_infinifabric_env.sh"

BUILD_JOBS="${INFINILM_BUILD_JOBS:-16}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: INFINILM_BUILD_JOBS must be a positive integer.\n' >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE_ARCHIVE_NAME="InfiniLM-infinifabric-v3.tar.gz"
readonly PROJECT_SOURCE_ARCHIVE="${SCRIPT_DIR}/${SOURCE_ARCHIVE_NAME}"
readonly AFS_SOURCE_ARCHIVE="${PKG_DIR}/${SOURCE_ARCHIVE_NAME}"
if [[ -f "${PROJECT_SOURCE_ARCHIVE}" ]]; then
    SOURCE_ARCHIVE="${PROJECT_SOURCE_ARCHIVE}"
elif [[ -f "${AFS_SOURCE_ARCHIVE}" ]]; then
    SOURCE_ARCHIVE="${AFS_SOURCE_ARCHIVE}"
else
    SOURCE_ARCHIVE="${PROJECT_SOURCE_ARCHIVE}"
fi
readonly SOURCE_ARCHIVE
readonly SOURCE_ARCHIVE_SHA256="4c62340aa0aef5fec9e5a7f57ea7753eb6666e070b3a7e66aed5d722baad39e6"
readonly REPO_CACHE="${PKG_DIR}/xmake-repo-cache.tar.gz"
readonly PYBIND_ARCHIVE="${PKG_DIR}/pybind11-v3.1.0.zip"
readonly PYBIND_SHA256="affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536"
readonly INFINICORE_SUCCESS_MARKER="${WORK_HOME}/state/infinicore-python.success"
readonly PERSISTENT_PROJECT_DIR="${WORK_HOME}/Projects/InfiniLM"
readonly PERSISTENT_INFINI_ROOT="${INFINI_ROOT}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUN_DIR="${LOG_DIR}/infinifabric-build-infinilm-extension-${RUN_ID}"
readonly RUN_LOG="${RUN_DIR}/run.log"
readonly SUMMARY="${RUN_DIR}/summary.txt"
readonly ARTIFACT_HASHES="${RUN_DIR}/artifacts.sha256"
readonly CONFIG_SNAPSHOT="${RUN_DIR}/InfiniLM_xmake.conf"
readonly STATE_DIR="${WORK_HOME}/state"
readonly SUCCESS_MARKER="${STATE_DIR}/infinilm-extension.success"

readonly LOCAL_ROOT="${PWD}/.infinilm-extension-${UID}-$$"
readonly LOCAL_PROJECT_DIR="${LOCAL_ROOT}/InfiniLM"
readonly LOCAL_PYBIND_ROOT="${LOCAL_ROOT}/pybind11-src/pybind11-3.1.0"
readonly LOCAL_PYBIND_INCLUDE="${LOCAL_PYBIND_ROOT}/include"
export XMAKE_GLOBALDIR="${LOCAL_ROOT}/global"
export XMAKE_STORAGE_DIR="${XMAKE_GLOBALDIR}/.xmake"
export XMAKE_TMPDIR="${LOCAL_ROOT}/tmp"
export XMAKE_RAMDIR="${LOCAL_ROOT}/tmp"
export XMAKE_STATS=false
export XMAKE_THEME=plain
export PYTHON="${PYTHON:-$(command -v python3)}"

PYTHON_INCLUDE_DIR="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
readonly PYTHON_INCLUDE_DIR
TORCH_CXX11_ABI="$("${PYTHON}" -c 'import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)')"
readonly TORCH_CXX11_ABI
export INFINILM_CXX11_ABI="${TORCH_CXX11_ABI}"
EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
readonly EXT_SUFFIX
readonly INFINICORE_EXTENSION="${WORK_HOME}/Projects/InfiniCore/python/infinicore/lib/_infinicore${EXT_SUFFIX}"

mkdir -p "${RUN_DIR}" "${STATE_DIR}" "${XMAKE_STORAGE_DIR}" "${XMAKE_TMPDIR}"
: > "${RUN_LOG}"

CURRENT_STAGE="initialization"
BUILD_PASSED=0

log_line() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

log_error() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}" >&2
}

on_exit() {
    local exit_code="$1"
    local result="FAILED"
    if [[ "${BUILD_PASSED}" -eq 1 && "${exit_code}" -eq 0 ]]; then
        result="PASSED"
    fi
    local summary_tmp="${SUMMARY}.tmp.$$"
    {
        printf 'build_version=%s\n' "${BUILD_VERSION}"
        printf 'result=%s\n' "${result}"
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'last_stage=%s\n' "${CURRENT_STAGE}"
        printf 'run_dir=%s\n' "${RUN_DIR}"
    } > "${summary_tmp}"
    mv -f -- "${summary_tmp}" "${SUMMARY}"
    log_line "RESULT: ${result}; exit_code=${exit_code}; last_stage=${CURRENT_STAGE}"
    log_line "SUMMARY: ${SUMMARY}"
}

trap 'on_exit "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
    local name="$1"
    shift
    CURRENT_STAGE="${name}"
    local stage_log="${RUN_DIR}/${name}.log"
    log_line ""
    log_line "=== ${name} ==="
    set +e
    "$@" 2>&1 | tee -a "${stage_log}" "${RUN_LOG}"
    local stage_rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${stage_rc}" -ne 0 ]]; then
        log_error "ERROR: stage ${name} failed with exit_code=${stage_rc}"
        return "${stage_rc}"
    fi
}

verify_inputs_readable() {
    local input_path
    for input_path in "$@"; do
        [[ -r "${input_path}" ]] || {
            printf 'ERROR: required input is not readable by this job UID: %s\n' \
                "${input_path}" >&2
            return 1
        }
        stat -Lc '%A mode=%a uid=%u gid=%g %n' "${input_path}"
    done
}

verify_inputs() {
    local actual_source_sha256 actual_pybind_sha256
    local source_list="${XMAKE_TMPDIR}/source-archive.list"
    actual_source_sha256="$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')"
    actual_pybind_sha256="$(sha256sum "${PYBIND_ARCHIVE}" | awk '{print $1}')"
    printf 'expected_source_archive_sha256=%s\n' "${SOURCE_ARCHIVE_SHA256}"
    printf 'actual_source_archive_sha256=%s\n' "${actual_source_sha256}"
    printf 'expected_pybind_sha256=%s\n' "${PYBIND_SHA256}"
    printf 'actual_pybind_sha256=%s\n' "${actual_pybind_sha256}"
    [[ "${actual_source_sha256}" == "${SOURCE_ARCHIVE_SHA256}" ]]
    [[ "${actual_pybind_sha256}" == "${PYBIND_SHA256}" ]]
    tar -tzf "${SOURCE_ARCHIVE}" > "${source_list}"
    grep -Fxq 'InfiniLM.SHA256SUMS' "${source_list}"
    grep -Fxq 'InfiniLM/xmake.lua' "${source_list}"
}

prepare_local_tree() {
    printf 'repository_cache_extract=begin\n'
    tar -xzf "${REPO_CACHE}" -C "${XMAKE_GLOBALDIR}"
    printf 'repository_cache_extract=complete\n'
    printf 'source_archive_extract=begin\n'
    tar -xzf "${SOURCE_ARCHIVE}" -C "${LOCAL_ROOT}"
    printf 'source_archive_extract=complete\n'
    printf 'pybind_archive_extract=begin\n'
    mkdir -p "${LOCAL_ROOT}/pybind11-src"
    "${PYTHON}" -m zipfile -e "${PYBIND_ARCHIVE}" \
        "${LOCAL_ROOT}/pybind11-src"
    printf 'pybind_archive_extract=complete\n'
    test -f "${LOCAL_PROJECT_DIR}/xmake.lua"
    test -f "${LOCAL_PYBIND_INCLUDE}/pybind11/pybind11.h"
}

verify_local_source() {
    cd -- "${LOCAL_ROOT}"
    sha256sum --check --quiet InfiniLM.SHA256SUMS
}

patch_local_project() {
    local xmake_file="${LOCAL_PROJECT_DIR}/xmake.lua"
    local pybind_require_line='add_requires("pybind11")'
    local pybind_package_line='    add_packages("pybind11")'
    local pybind_include_line="    add_includedirs(\"${LOCAL_PYBIND_INCLUDE}\", \"${PYTHON_INCLUDE_DIR}\", { public = false })"
    [[ "$(grep -Fxc "${pybind_require_line}" "${xmake_file}")" -eq 1 ]]
    [[ "$(grep -Fxc "${pybind_package_line}" "${xmake_file}")" -eq 1 ]]
    sed -i \
        -e '/^add_requires("pybind11")$/d' \
        -e "s#^    add_packages(\"pybind11\")\$#${pybind_include_line}#" \
        "${xmake_file}"
    ! grep -Fq "${pybind_require_line}" "${xmake_file}"
    ! grep -Fq "${pybind_package_line}" "${xmake_file}"
    [[ "$(grep -Fxc "${pybind_include_line}" "${xmake_file}")" -eq 1 ]]
    printf 'patch_profile=vendored-pybind-active-python-headers\n'
    printf 'patched_xmake_sha256='
    sha256sum "${xmake_file}" | awk '{print $1}'
}

patch_hygon_fl2va_no_cpu_runtime() {
    "${PYTHON}" -c '
from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace_once(path, old, new, label):
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count} in {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace(old, new))
    print(f"patched={label} path={path}")


qwen_vl = root / "csrc/models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp"
qwen_vision = root / "csrc/models/qwen3_5/qwen3_5_vision.cpp"
infer_engine = root / "csrc/engine/infer_engine.cpp"
rank_worker = root / "csrc/engine/rank_worker.cpp"
static_attn = root / "csrc/layers/attention/backends/static_attn.cpp"

replace_once(
    qwen_vl,
    """#include "qwen3_vl_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp""",
    """#include "qwen3_vl_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/context/context.hpp>""",
    "qwen3-vl-context-include",
)
replace_once(
    qwen_vl,
    """#include <cstdint>
#include <stdexcept>""",
    """#include <cstdint>
#include <cstring>
#include <stdexcept>""",
    "qwen3-vl-cstring-include",
)
replace_once(
    qwen_vl,
    """std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->to(infinicore::Device::cpu());
    std::vector<int64_t> values(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        const auto *data = reinterpret_cast<const int64_t *>(cpu->data());
        values.assign(data, data + cpu->numel());
        return values;
    }
    if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *data = reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < cpu->numel(); ++i) {
            values[i] = static_cast<int64_t>(data[i]);
        }
        return values;
    }
    throw std::runtime_error(
        "Qwen3VLModel: metadata tensors must be int32 or int64");
}""",
    """std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(
            "Qwen3VLModel: metadata tensors must be contiguous");
    }
    const size_t numel = tensor->numel();
    std::vector<int64_t> values(numel);
    if (tensor->dtype() == infinicore::DataType::I64) {
        if (tensor->device().getType() == infinicore::Device::Type::CPU) {
            std::memcpy(values.data(), tensor->data(), tensor->nbytes());
        } else {
            infinicore::context::setDevice(tensor->device());
            infinicore::context::memcpyD2H(
                values.data(), tensor->data(), tensor->nbytes());
        }
        return values;
    }
    if (tensor->dtype() == infinicore::DataType::I32) {
        std::vector<int32_t> raw(numel);
        if (tensor->device().getType() == infinicore::Device::Type::CPU) {
            std::memcpy(raw.data(), tensor->data(), tensor->nbytes());
        } else {
            infinicore::context::setDevice(tensor->device());
            infinicore::context::memcpyD2H(
                raw.data(), tensor->data(), tensor->nbytes());
        }
        for (size_t i = 0; i < numel; ++i) {
            values[i] = static_cast<int64_t>(raw[i]);
        }
        return values;
    }
    throw std::runtime_error(
        "Qwen3VLModel: metadata tensors must be int32 or int64");
}""",
    "qwen3-vl-device-to-host-metadata",
)

replace_once(
    qwen_vision,
    """#include "../../utils.hpp"

#include <infinicore/ops.hpp>""",
    """#include "../../utils.hpp"

#include <infinicore/context/context.hpp>
#include <infinicore/ops.hpp>""",
    "qwen3-vision-context-include",
)
replace_once(
    qwen_vision,
    """#include <cmath>
#include <cstdint>
#include <iterator>""",
    """#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>""",
    "qwen3-vision-cstring-include",
)
replace_once(
    qwen_vision,
    """std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    auto cpu_tensor = tensor->to(infinicore::Device::cpu());
    std::vector<int64_t> values(cpu_tensor->numel());
    if (cpu_tensor->dtype() == infinicore::DataType::I64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(cpu_tensor->data());
        values.assign(ptr, ptr + cpu_tensor->numel());
        return values;
    }
    if (cpu_tensor->dtype() == infinicore::DataType::I32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(cpu_tensor->data());
        for (size_t i = 0; i < cpu_tensor->numel(); ++i) {
            values[i] = static_cast<int64_t>(ptr[i]);
        }
        return values;
    }
    throw std::runtime_error(
        "Qwen35VisionModel: grid_thw must be int32 or int64");
}""",
    """std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(
            "Qwen35VisionModel: grid_thw must be contiguous");
    }
    const size_t numel = tensor->numel();
    std::vector<int64_t> values(numel);
    if (tensor->dtype() == infinicore::DataType::I64) {
        if (tensor->device().getType() == infinicore::Device::Type::CPU) {
            std::memcpy(values.data(), tensor->data(), tensor->nbytes());
        } else {
            infinicore::context::setDevice(tensor->device());
            infinicore::context::memcpyD2H(
                values.data(), tensor->data(), tensor->nbytes());
        }
        return values;
    }
    if (tensor->dtype() == infinicore::DataType::I32) {
        std::vector<int32_t> raw(numel);
        if (tensor->device().getType() == infinicore::Device::Type::CPU) {
            std::memcpy(raw.data(), tensor->data(), tensor->nbytes());
        } else {
            infinicore::context::setDevice(tensor->device());
            infinicore::context::memcpyD2H(
                raw.data(), tensor->data(), tensor->nbytes());
        }
        for (size_t i = 0; i < numel; ++i) {
            values[i] = static_cast<int64_t>(raw[i]);
        }
        return values;
    }
    throw std::runtime_error(
        "Qwen35VisionModel: grid_thw must be int32 or int64");
}""",
    "qwen3-vision-device-to-host-metadata",
)
replace_once(
    qwen_vision,
    """    auto position_ids_cpu = infinicore::Tensor::empty(
        {2, total_tokens}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *position_ids = reinterpret_cast<int64_t *>(position_ids_cpu->data());""",
    """    std::vector<int64_t> position_ids_storage(2 * total_tokens);
    auto *position_ids = position_ids_storage.data();""",
    "qwen3-vision-host-position-ids",
)
replace_once(
    qwen_vision,
    """    return position_ids_cpu->to(image_grid_thw->device());""",
    """    auto position_ids_output = infinicore::Tensor::empty(
        {2, total_tokens}, infinicore::DataType::I64,
        image_grid_thw->device());
    if (image_grid_thw->device().getType() ==
        infinicore::Device::Type::CPU) {
        std::memcpy(position_ids_output->data(), position_ids_storage.data(),
                    position_ids_output->nbytes());
    } else {
        infinicore::context::setDevice(image_grid_thw->device());
        infinicore::context::memcpyH2D(
            position_ids_output->data(), position_ids_storage.data(),
            position_ids_output->nbytes(), false);
    }
    return position_ids_output;""",
    "qwen3-vision-host-to-device-position-ids",
)

replace_once(
    infer_engine,
    """#include <cstdint>
#include <future>""",
    """#include <cstdint>
#include <cstring>
#include <future>""",
    "infer-engine-cstring-include",
)
replace_once(
    infer_engine,
    """    auto cpu_offsets = offsets.value();
    if (cpu_offsets->device().getType() != infinicore::Device::Type::CPU) {
        cpu_offsets = cpu_offsets->to(infinicore::Device::cpu());
        infinicore::context::syncStream();
    }

    if (cpu_offsets->dtype() != infinicore::DataType::I32
        || cpu_offsets->shape().size() != 1
        || cpu_offsets->shape()[0] < 2) {
        throw std::invalid_argument(
            std::string(name) + " must be a one-dimensional int32 tensor with at least two entries");
    }

    const auto *values = reinterpret_cast<const int32_t *>(cpu_offsets->data());
    size_t max_length = 0;
    for (size_t i = 1; i < cpu_offsets->shape()[0]; ++i) {
        if (values[i] < values[i - 1]) {
            throw std::invalid_argument(std::string(name) + " must be nondecreasing");
        }
        max_length = std::max(
            max_length,
            static_cast<size_t>(values[i] - values[i - 1]));
    }
    return max_length;""",
    """    const auto &offset_tensor = offsets.value();
    if (offset_tensor->dtype() != infinicore::DataType::I32
        || offset_tensor->shape().size() != 1
        || offset_tensor->shape()[0] < 2
        || !offset_tensor->is_contiguous()) {
        throw std::invalid_argument(
            std::string(name) + " must be a contiguous one-dimensional int32 tensor with at least two entries");
    }

    std::vector<int32_t> values(offset_tensor->numel());
    if (offset_tensor->device().getType() == infinicore::Device::Type::CPU) {
        std::memcpy(values.data(), offset_tensor->data(), offset_tensor->nbytes());
    } else {
        infinicore::context::setDevice(offset_tensor->device());
        infinicore::context::memcpyD2H(
            values.data(), offset_tensor->data(), offset_tensor->nbytes());
    }

    size_t max_length = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] < values[i - 1]) {
            throw std::invalid_argument(std::string(name) + " must be nondecreasing");
        }
        max_length = std::max(
            max_length,
            static_cast<size_t>(values[i] - values[i - 1]));
    }
    return max_length;""",
    "infer-engine-host-offset-read",
)

replace_once(
    rank_worker,
    """                            auto output_ids = infinicore::Tensor::empty(
                                {0}, infinicore::DataType::I64,
                                infinicore::Device::cpu());""",
    """                            auto output_ids = infinicore::Tensor::empty(
                                {0}, infinicore::DataType::I64, rank_info_.device);""",
    "rank-worker-skip-sampling-device-output",
)

replace_once(
    static_attn,
    """#include "infinicore/ops/per_tensor_quant_i8.hpp"

namespace infinilm::layers::attention::backends {""",
    """#include "infinicore/ops/per_tensor_quant_i8.hpp"

#include <infinicore/context/context.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infinilm::layers::attention::backends {
namespace {

int32_t read_i32_scalar(const infinicore::Tensor &tensor,
                        const char *name) {
    if (tensor->dtype() != infinicore::DataType::I32
        || tensor->numel() < 1
        || !tensor->is_contiguous()) {
        throw std::invalid_argument(
            std::string(name) + " must be a contiguous nonempty int32 tensor");
    }
    int32_t value = 0;
    if (tensor->device().getType() == infinicore::Device::Type::CPU) {
        std::memcpy(&value, tensor->data(), sizeof(value));
    } else {
        infinicore::context::setDevice(tensor->device());
        infinicore::context::memcpyD2H(
            &value, tensor->data(), sizeof(value));
    }
    return value;
}

} // namespace""",
    "static-attn-host-scalar-reader",
)
replace_once(
    static_attn,
    """        size_t total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];""",
    """        size_t total_seq_len = static_cast<size_t>(
            read_i32_scalar(total_sequence_lengths.value(),
                            "total_sequence_lengths"));""",
    "static-attn-total-length-host-read",
)
replace_once(
    static_attn,
    """    size_t cache_pos = reinterpret_cast<int32_t *>(past_sequence_lengths->to(infinicore::Device::cpu())->data())[0];""",
    """    size_t cache_pos = static_cast<size_t>(
        read_i32_scalar(past_sequence_lengths, "past_sequence_lengths"));""",
    "static-attn-past-length-host-read",
)

print("patch_profile=hygon-fl2va-no-cpu-runtime")
' "${LOCAL_PROJECT_DIR}"
}

log_line "build_version=${BUILD_VERSION}"
log_line "time=$(date -Is)"
log_line "host=$(hostname)"
log_line "uid=$(id -u)"
log_line "work_home=${WORK_HOME}"
log_line "source_archive=${SOURCE_ARCHIVE}"
log_line "persistent_project=${PERSISTENT_PROJECT_DIR}"
log_line "persistent_infini_root=${PERSISTENT_INFINI_ROOT}"
log_line "local_project=${LOCAL_PROJECT_DIR}"
log_line "build_jobs=${BUILD_JOBS}"
log_line "python_include_dir=${PYTHON_INCLUDE_DIR}"
log_line "torch_cxx11_abi=${TORCH_CXX11_ABI}"

[[ -f "${INFINICORE_SUCCESS_MARKER}" ]] || {
    log_error "ERROR: verified InfiniCore Python marker is missing: ${INFINICORE_SUCCESS_MARKER}"
    exit 1
}
[[ -f "${SOURCE_ARCHIVE}" ]] || {
    log_error "ERROR: InfiniLM source archive is missing from both project input and AFS packages."
    log_error "project_archive=${PROJECT_SOURCE_ARCHIVE}"
    log_error "afs_archive=${AFS_SOURCE_ARCHIVE}"
    exit 1
}
[[ -f "${REPO_CACHE}" ]] || {
    log_error "ERROR: xmake repository cache is missing: ${REPO_CACHE}"
    exit 1
}
[[ -f "${PYBIND_ARCHIVE}" ]] || {
    log_error "ERROR: pybind11 archive is missing: ${PYBIND_ARCHIVE}"
    exit 1
}
[[ -f "${PYTHON_INCLUDE_DIR}/Python.h" ]] || {
    log_error "ERROR: Python development header is missing: ${PYTHON_INCLUDE_DIR}/Python.h"
    exit 1
}
[[ -f "${PERSISTENT_PROJECT_DIR}/python/infinilm/lib/__init__.py" ]] || {
    log_error "ERROR: persistent InfiniLM Python sources are missing."
    exit 1
}
for native_library in \
    libinfinicore_cpp_api.so libinfiniop.so libinfinirt.so libinfiniccl.so
do
    [[ -f "${PERSISTENT_INFINI_ROOT}/lib/${native_library}" ]] || {
        log_error "ERROR: verified InfiniCore library is missing: ${PERSISTENT_INFINI_ROOT}/lib/${native_library}"
        exit 1
    }
done
[[ -f "${INFINICORE_EXTENSION}" ]] || {
    log_error "ERROR: verified InfiniCore extension is missing: ${INFINICORE_EXTENSION}"
    exit 1
}

run_stage "input-readability" verify_inputs_readable \
    "${INFINICORE_EXTENSION}" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniccl.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfinicore_cpp_api.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfiniop.so" \
    "${PERSISTENT_INFINI_ROOT}/lib/libinfinirt.so"

CURRENT_STAGE="invalidate-stale-success-marker"
if [[ -f "${SUCCESS_MARKER}" ]]; then
    mv -- "${SUCCESS_MARKER}" "${RUN_DIR}/previous-infinilm-extension.success"
fi

run_stage "input-check" verify_inputs
run_stage "local-tree" prepare_local_tree
run_stage "source-check" verify_local_source
run_stage "local-source-patch" patch_local_project
run_stage "hygon-fl2va-no-cpu-runtime-patch" patch_hygon_fl2va_no_cpu_runtime

CURRENT_STAGE="torch-runtime"
TORCH_LIB_DIR="$("${PYTHON}" -c \
    'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
readonly TORCH_LIB_DIR
for runtime_library in \
    libtorch.so libtorch_cpu.so libtorch_python.so libtorch_hip.so \
    libc10.so libc10_hip.so
do
    [[ -f "${TORCH_LIB_DIR}/${runtime_library}" ]] || {
        log_error "ERROR: PyTorch runtime library is missing: ${TORCH_LIB_DIR}/${runtime_library}"
        exit 1
    }
done
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${PERSISTENT_INFINI_ROOT}/lib:${LD_LIBRARY_PATH:-}"
log_line "torch_lib_dir=${TORCH_LIB_DIR}"

run_stage "xmake-global-config" \
    xmake g --network=private --pkg_searchdirs="${PKG_DIR}"

cd -- "${LOCAL_PROJECT_DIR}"
run_stage "xmake-config" \
    xmake f -y -m release --cxx11-abi="${TORCH_CXX11_ABI}"
run_stage "config-snapshot" cp \
    "${LOCAL_PROJECT_DIR}/.xmake/linux/x86_64/xmake.conf" \
    "${CONFIG_SNAPSHOT}"
run_stage "xmake-build" xmake build -j "${BUILD_JOBS}" _infinilm

CURRENT_STAGE="extension-path"
readonly BUILT_EXTENSION="${LOCAL_PROJECT_DIR}/build/linux/x86_64/release/_infinilm${EXT_SUFFIX}"
readonly DESTINATION_DIR="${PERSISTENT_PROJECT_DIR}/python/infinilm/lib"
readonly DESTINATION_EXTENSION="${DESTINATION_DIR}/_infinilm${EXT_SUFFIX}"
[[ -f "${BUILT_EXTENSION}" ]] || {
    log_error "ERROR: built InfiniLM extension is missing: ${BUILT_EXTENSION}"
    exit 1
}

run_stage "file" file "${BUILT_EXTENSION}"
run_stage "ldd" ldd "${BUILT_EXTENSION}"
if grep -Fq "not found" "${RUN_DIR}/ldd.log"; then
    log_error "ERROR: unresolved dynamic library dependency detected."
    exit 1
fi
run_stage "built-extension-readability" chmod o+r -- "${BUILT_EXTENSION}"

CURRENT_STAGE="extension-promotion"
mkdir -p "${DESTINATION_DIR}"
if [[ -f "${DESTINATION_EXTENSION}" ]]; then
    if [[ -r "${DESTINATION_EXTENSION}" ]]; then
        cp -p -- "${DESTINATION_EXTENSION}" \
            "${RUN_DIR}/previous-_infinilm${EXT_SUFFIX}"
    else
        log_line "previous_extension_backup=SKIPPED_UNREADABLE"
    fi
fi
EXTENSION_TMP="${DESTINATION_EXTENSION}.tmp.$$"
cp -p -- "${BUILT_EXTENSION}" "${EXTENSION_TMP}"
[[ "$(sha256sum "${BUILT_EXTENSION}" | awk '{print $1}')" == \
   "$(sha256sum "${EXTENSION_TMP}" | awk '{print $1}')" ]]
mv -f -- "${EXTENSION_TMP}" "${DESTINATION_EXTENSION}"
[[ -r "${DESTINATION_EXTENSION}" ]] || {
    log_error "ERROR: promoted extension is not readable in the build job."
    exit 1
}

export EXPECTED_EXTENSION="${DESTINATION_EXTENSION}"
export INFINILM_EXTENSION_DIR="${DESTINATION_DIR}"
run_stage "extension-import-smoke" "${PYTHON}" -c '
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["INFINILM_EXTENSION_DIR"])
import _infinilm

actual = Path(_infinilm.__file__).resolve()
expected = Path(os.environ["EXPECTED_EXTENSION"]).resolve()
assert actual == expected, (actual, expected)
assert hasattr(_infinilm, "InferEngine")
assert hasattr(_infinilm, "DistConfig")
print("extension:", actual)
print("InferEngine: OK")
print("DistConfig: OK")
'

CURRENT_STAGE="artifact-hashing"
sha256sum "${DESTINATION_EXTENSION}" "${CONFIG_SNAPSHOT}" \
    "${INFINICORE_SUCCESS_MARKER}" > "${ARTIFACT_HASHES}"
cat "${ARTIFACT_HASHES}" | tee -a "${RUN_LOG}"

CURRENT_STAGE="success-marker"
SUCCESS_MARKER_TMP="${SUCCESS_MARKER}.tmp.$$"
{
    printf 'build_version=%s\n' "${BUILD_VERSION}"
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'cxx11_abi=%s\n' "${TORCH_CXX11_ABI}"
    printf 'extension=%s\n' "${DESTINATION_EXTENSION}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'artifacts_sha256=%s\n' "${ARTIFACT_HASHES}"
} > "${SUCCESS_MARKER_TMP}"
mv -f -- "${SUCCESS_MARKER_TMP}" "${SUCCESS_MARKER}"

BUILD_PASSED=1
CURRENT_STAGE="complete"
log_line "InfiniLM extension build: PASSED"
log_line "success_marker=${SUCCESS_MARKER}"
log_line "run_dir=${RUN_DIR}"
