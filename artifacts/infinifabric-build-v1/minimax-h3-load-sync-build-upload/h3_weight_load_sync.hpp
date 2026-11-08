#pragma once
#include <infinicore/context/context.hpp>
#include <infinicore/tensor.hpp>

namespace infinilm::engine::weight_load_sync {
// Correctness-first loading candidate. Existing Core slicing stays on the rank's device.
// Full-parameter staging costs temporary memory and bandwidth; not an inference optimization.
inline infinicore::Tensor stage(const infinicore::Tensor &source,
                                const infinicore::Device &destination) {
    if (source->device().getType() != infinicore::Device::Type::HYGON ||
        destination.getType() != infinicore::Device::Type::HYGON ||
        source->device() == destination) {
        return source;
    }
    infinicore::context::setDevice(source->device());
    auto ready = source->contiguous();
    infinicore::context::syncStream();
    auto staged = ready->to(destination);
    infinicore::context::setDevice(destination);
    // ready remains owned here until the target has finished reading it.
    infinicore::context::syncStream();
    return staged;
}
} // namespace infinilm::engine::weight_load_sync
