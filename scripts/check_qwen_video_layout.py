#!/usr/bin/env python3
"""Compile actual native frame-size validation with host C++; no GPU claim."""
import argparse
from pathlib import Path
import subprocess
import tempfile

PREAMBLE = r'''
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
'''

CASES = r'''
int main() {
    int passed = 0;
    auto accept = [&](const char *name, std::vector<int64_t> grid, size_t patches, size_t expected) {
        if (vision_frame_tokens(grid, patches) != expected) {
            throw std::runtime_error(std::string("wrong frame size: ") + name);
        }
        std::cout << "PASS " << name << '\n';
        ++passed;
    };
    auto reject = [&](const char *name, std::vector<int64_t> grid, size_t patches, const char *reason) {
        try {
            vision_frame_tokens(grid, patches);
        } catch (const std::runtime_error &e) {
            if (std::string(e.what()).find(reason) == std::string::npos) throw;
            std::cout << "PASS " << name << '\n';
            ++passed;
            return;
        }
        throw std::runtime_error(std::string("unexpected acceptance: ") + name);
    };
    try {
        accept("single_image", {1,4,6}, 24, 24);
        accept("two_frames", {2,2,4}, 16, 8);
        accept("three_frames", {3,6,4}, 72, 24);
        accept("singleton_spatial", {4,1,1}, 4, 1);
        reject("empty_grid", {}, 0, "positive");
        reject("short_grid", {1,2}, 2, "positive");
        reject("multiple_grids", {1,2,4,1,2,4}, 16, "positive");
        reject("zero_temporal", {0,2,4}, 0, "positive");
        reject("negative_spatial", {1,-2,4}, 8, "positive");
        reject("zero_width", {1,2,0}, 0, "positive");
        reject("row_count_mismatch", {2,2,4}, 8, "pixel rows");
        const auto huge = std::numeric_limits<int64_t>::max();
        reject("spatial_overflow", {1,huge,3}, 1, "overflow");
        reject("temporal_overflow", {huge,2,2}, 1, "overflow");
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    std::cout << "PASSED: " << passed << " native frame-size host checks (no GPU)\n";
}
'''


def probe_source(repo):
    source = (Path(repo)/'csrc/models/qwen3_5/qwen3_5_vision.cpp').read_text(encoding='utf-8')
    start = source.index('size_t vision_frame_tokens(')
    end = source.index('size_t get_size_or_first(', start)
    return PREAMBLE + source[start:end] + CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='qwen-video-layout-') as folder:
        source = Path(folder)/'layout.cc'
        binary = Path(folder)/'layout'
        source.write_text(probe_source(args.repo), encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++17', '-Wall', '-Wextra', '-pedantic', str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
