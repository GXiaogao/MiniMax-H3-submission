#!/usr/bin/env python3
"""Compile/run the real MHA layout checks with host-only Tensor metadata.

No GPU or InfiniCore libraries are linked. This checks validation/descriptor
metadata only; numerical correctness still requires the device regression.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


PREAMBLE = r'''
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
using Stride = ptrdiff_t;
struct TensorStub {
    std::array<size_t, 4> shape;
    std::array<Stride, 4> strides;
    size_t bytes;
    size_t ndim() const { return 4; }
    size_t size(size_t i) const { return shape[i]; }
    Stride stride(size_t i) const { return strides[i]; }
    size_t element_size() const { return bytes; }
};
using Tensor = std::shared_ptr<TensorStub>;
'''

CASES = r'''
int main() {
    size_t passed = 0;
    auto accepted = [&](const char *name, std::array<size_t,4> shape,
                        std::array<Stride,4> strides, size_t span, Stride row) {
        auto t = std::make_shared<TensorStub>(TensorStub{shape, strides, 4});
        if (validate_layout(t) != span || sequence_stride(t) != row) {
            throw std::runtime_error(std::string("wrong span/row stride: ") + name);
        }
        std::cout << "PASS " << name << '\n';
        ++passed;
    };
    auto rejected = [&](const char *name, std::array<size_t,4> shape,
                        std::array<Stride,4> strides) {
        auto t = std::make_shared<TensorStub>(TensorStub{shape, strides, 4});
        try { validate_layout(t); }
        catch (const std::invalid_argument &) {
            std::cout << "PASS reject:" << name << '\n';
            ++passed;
            return;
        }
        throw std::runtime_error(std::string("accepted invalid layout: ") + name);
    };
    try {
        accepted("qwen_unsqueeze_batch", {1,24,1,64}, {0,64,64,1}, 6144, 64);
        accepted("singleton_head", {2,7,1,64}, {448,64,0,1}, 3584, 64);
        accepted("singleton_sequence", {2,1,4,64}, {256,0,64,1}, 2048, 64);
        accepted("all_singleton_outer", {1,1,1,64}, {0,0,0,1}, 256, 64);
        accepted("packed_positive", {1,5,4,64}, {3840,768,64,1}, 13312, 768);
        accepted("permuted_positive", {2,3,4,64}, {768,64,192,1}, 6144, 64);
        rejected("broadcast_batch", {2,7,1,64}, {0,64,64,1});
        rejected("broadcast_sequence", {1,7,1,64}, {448,0,64,1});
        rejected("broadcast_head", {1,7,2,64}, {896,128,0,1});
        rejected("negative_singleton", {1,7,1,64}, {-1,64,64,1});
        rejected("overlap", {1,7,2,64}, {896,64,64,1});
        rejected("last_stride", {1,7,1,64}, {896,128,128,2});
        rejected("empty", {1,0,1,64}, {0,64,64,1});
        const auto limit = std::numeric_limits<ptrdiff_t>::max();
        rejected("span_overflow", {1,2,1,64}, {0,limit,64,1});
        rejected("byte_overflow", {1,1,1,static_cast<size_t>(limit)}, {0,0,0,1});
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    std::cout << "PASSED: " << passed << " host layout checks (no GPU)\n";
}
'''


def probe_source(repo):
    source = (Path(repo) / 'src/infinicore/ops/multi_head_attention/mha_hygon_infiniop.cc').read_text(encoding='utf-8')
    # Compile the actual source functions, never a Python translation of them.
    start = source.index('void require(')
    end = source.index('void reject_output_alias(', start)
    return PREAMBLE + source[start:end] + CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='h3-mha-layout-') as folder:
        source = Path(folder) / 'layout.cc'
        binary = Path(folder) / 'layout'
        source.write_text(probe_source(args.repo), encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++17', '-Wall', '-Wextra', '-pedantic', str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
