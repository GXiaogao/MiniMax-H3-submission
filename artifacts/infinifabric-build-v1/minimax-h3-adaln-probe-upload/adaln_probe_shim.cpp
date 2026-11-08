// Diagnostic only: calls the installed InfiniCore projection and allgather.
#include <infinicore/tensor.hpp>
#include <infinicore/context/context.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/distributed/allgather.hpp>
#include <infiniccl.h>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>
#include <cstdio>

extern "C" int adaln_probe(int world, int rows, int in_width, int out_width, int marker,
                         void **inputs, void **weights, void **biases,
                         void **locals, void **outputs, char *error, size_t error_size) {
    std::vector<infinicclComm_t> comms(world > 0 ? world : 0, nullptr);
    try {
        if ((world != 1 && world != 8) || rows < 1 || in_width < 1 || out_width % world || out_width < 1)
            throw std::runtime_error("invalid diagnostic dimensions");
        infinicore::context::setDevice(infinicore::Device(infinicore::Device::Type::HYGON,0));
        if (world > 1) {
            std::vector<int> ids(world);for(int i=0;i<world;++i)ids[i]=i;
            if (infinicclCommInitAll(INFINI_DEVICE_HYGON,comms.data(),world,ids.data()) != 0)
                throw std::runtime_error("infinicclCommInitAll failed");
        }
        std::vector<std::exception_ptr> errors(world);
        std::vector<std::thread> workers;
        for(int rank=0;rank<world;++rank)workers.emplace_back([&,rank] {
            try {
                const size_t local_width=out_width/world;
                infinicore::Device device(infinicore::Device::Type::HYGON,rank);
                infinicore::context::setDevice(device);
                auto x=infinicore::Tensor::from_blob(inputs[rank],{size_t(rows),size_t(marker?local_width:in_width)},infinicore::DataType::BF16,device);
                auto local=x;
                if (!marker) {
                    auto w=infinicore::Tensor::from_blob(weights[rank],{local_width,size_t(in_width)},infinicore::DataType::BF16,device);
                    auto b=infinicore::Tensor::from_blob(biases[rank],{local_width},infinicore::DataType::BF16,device);
                    local=infinicore::op::linear(x,w,b);
                }
                auto local_dst=infinicore::Tensor::from_blob(locals[rank],{size_t(rows),local_width},infinicore::DataType::BF16,device);
                local_dst->copy_from(local);
                auto gathered=local;
                if(world>1) {
                    auto transposed=local->permute({1,0})->contiguous();
                    gathered=infinicore::op::distributed::allgather(transposed,world,comms[rank]);
                    gathered=gathered->permute({1,0})->contiguous();
                }
                auto dst=infinicore::Tensor::from_blob(outputs[rank],{size_t(rows),size_t(out_width)},infinicore::DataType::BF16,device);
                dst->copy_from(gathered);
                infinicore::context::syncDevice();
            } catch(...) {errors[rank]=std::current_exception();}
        });
        for(auto &thread:workers)thread.join();
        for(auto &failure:errors)if(failure)std::rethrow_exception(failure);
        for(auto &comm:comms)if(comm){if(infinicclCommDestroy(comm)!=0)throw std::runtime_error("communicator destruction failed");comm=nullptr;}
        return 0;
    } catch(const std::exception &e) {
        // A failed collective can leave peers blocked; the outer job timeout
        // bounds that case. Never reinterpret a partial run as a comparison.
        std::snprintf(error,error_size,"%s",e.what());return 1;
    }
}
