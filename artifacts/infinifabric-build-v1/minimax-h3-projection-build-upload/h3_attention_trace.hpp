#pragma once
// First-refiner Attention snapshots, each TP rank writes a distinct filename.
#include <infinicore/context/context.hpp>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace infinilm::models::minimax_h3::attention_trace {
inline thread_local bool active = false;
inline bool enabled() {
    const auto *dir=std::getenv("H3_BOUNDARY_TRACE_DIR");
    return dir && *dir;
}
struct Scope {
    const bool previous;
    explicit Scope(bool first_refiner) : previous(active) { active = first_refiner && enabled(); }
    ~Scope() { active = previous; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
inline void save(const char *name,const infinicore::Tensor &value) {
    if(!active || !enabled())return;
    const auto &rank=infinilm::global_state::get_tensor_model_parallel_rank_info();
    if ((rank.tp_size != 1 && rank.tp_size != 8) || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size)
        throw std::runtime_error("Attention trace requires TP1/TP8 and a valid rank");
    if(infinicore::context::isGraphRecording())
        throw std::runtime_error("H3 Attention trace requires graph compilation disabled");
    if(value->dtype()!=infinicore::DataType::F32 && value->dtype()!=infinicore::DataType::BF16)
        throw std::runtime_error("H3 Attention trace requires FP32/BF16");
    auto tensor=value->is_contiguous()?value:value->contiguous();
    const size_t total=tensor->numel(), count=std::min<size_t>(65536,total);
    if(!count)throw std::runtime_error("empty trace tensor");
    const std::string stem=std::string(std::getenv("H3_BOUNDARY_TRACE_DIR"))+"/refiner0_"+name+".rank"+std::to_string(rank.tp_rank);
    if(std::ifstream(stem+".json").good() || std::ifstream(stem+".f32").good())
        throw std::runtime_error("trace boundary already exists; use a fresh worker directory");
    infinicore::context::syncStream();
    std::vector<unsigned char> raw(tensor->nbytes());
    infinicore::context::memcpyD2H(raw.data(),tensor->data(),raw.size());
    std::vector<float> samples(count);
    const bool f32=tensor->dtype()==infinicore::DataType::F32;
    for(size_t i=0;i<count;++i) {
        const size_t index=count==1?0:i*(total-1)/(count-1);
        if(f32)std::memcpy(&samples[i],raw.data()+index*4,4);
        else {
            uint16_t bits;std::memcpy(&bits,raw.data()+index*2,2);
            uint32_t wide=uint32_t(bits)<<16;std::memcpy(&samples[i],&wide,4);
        }
    }
    std::ofstream output(stem+".f32",std::ios::binary);
    output.exceptions(std::ios::failbit|std::ios::badbit);
    output.write(reinterpret_cast<const char*>(samples.data()),samples.size()*sizeof(float));output.close();
    std::ofstream meta(stem+".json");meta.exceptions(std::ios::failbit|std::ios::badbit);
    meta<<"{\"shape\":[";
    for(size_t i=0;i<tensor->ndim();++i){if(i)meta<<",";meta<<tensor->size(i);}
    meta<<"],\"dtype\":\""<<(f32?"float32":"bfloat16")<<"\",\"elements\":"<<total
        <<",\"sample_count\":"<<count<<",\"storage_dtype\":\"float32\",\"rank\":"<<rank.tp_rank<<",\"tp_size\":"<<rank.tp_size<<","
        <<"\"sampling\":\"linear-integer-indices-at-most-65536\",\"all_elements_finite_checked\":false}\n";
}
} // namespace infinilm::models::minimax_h3::attention_trace
