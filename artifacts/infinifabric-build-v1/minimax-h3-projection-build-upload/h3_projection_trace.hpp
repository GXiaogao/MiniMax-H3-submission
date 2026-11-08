#pragma once
#include "h3_attention_trace.hpp"
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>

namespace infinilm::models::minimax_h3::projection_trace {
inline thread_local bool active = false;
struct Scope {
    const bool previous;
    explicit Scope(bool requested) : previous(active) { active = requested && attention_trace::enabled(); }
    ~Scope() { active = previous; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
inline void inspect_serial(const infinicore::Tensor &weight, const infinicore::Tensor &original) {
    if (!active) return;
    attention_trace::save("out_weight", weight);
    attention_trace::save("out_original", original);
    attention_trace::save("out_local", original);
    attention_trace::save("out_control_sum", original); // TP1 identity control.
    attention_trace::save("out_original_after_control", original);
}
inline void inspect_control(const infinicore::Tensor &input, const infinicore::Tensor &weight,
                            const std::optional<infinicore::Tensor> &bias,
                            infinicclComm_t communicator, bool packed,
                            const infinicore::Tensor &original) {
    if (!active) return;
    if (bias) throw std::runtime_error("projection trace requires bias-free output projection");
    // Observe the original before additional GPU work. Never return these controls.
    attention_trace::save("out_weight", weight);
    attention_trace::save("out_original", original);
    auto local = packed ? infinicore::op::linear_packed(input, weight, bias)
                        : infinicore::op::linear(input, weight->contiguous(), bias);
    attention_trace::save("out_local", local);
    auto control_sum = infinicore::op::distributed::allreduce(local, INFINICCL_SUM, communicator);
    attention_trace::save("out_control_sum", control_sum);
    attention_trace::save("out_original_after_control", original);
}
} // namespace infinilm::models::minimax_h3::projection_trace
