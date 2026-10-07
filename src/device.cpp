#include "carbon/device.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {
int cuda_gpu_score(const sycl::device& device) {
    if (!device.is_gpu()) {
        return -1;
    }
    if (device.get_backend() != sycl::backend::ext_oneapi_cuda) {
        return -1;
    }
    // Preserve the established CUDA device ordering, including FP64/atomic64
    // capabilities used by auxiliary reductions. Dose atomics remain FP32.
    int score = 1000;
    if (device.has(sycl::aspect::fp64)) {
        score += 100;
    }
    if (device.has(sycl::aspect::atomic64)) {
        score += 100;
    }
    // Among equally capable GPUs prefer the one with the most global memory
    // (in GiB, bounded to keep the score in range), then more compute units.
    try {
        score += static_cast<int>(
            device.get_info<sycl::info::device::global_mem_size>() >> 30);
        score += static_cast<int>(
            std::min<std::size_t>(
                device.get_info<sycl::info::device::max_compute_units>(), 1024));
    } catch (const sycl::exception&) {
    }
    return score;
}

}  // namespace

sycl::queue make_sycl_queue(const std::string& device_name) {
    if (device_name != "cuda") throw std::invalid_argument("device must be cuda");
    try {
        return sycl::queue{
            cuda_gpu_score,
            [](sycl::exception_list exceptions) {
                for (const auto& exception : exceptions) std::rethrow_exception(exception);
            },
            sycl::property_list{sycl::property::queue::enable_profiling{},
                                sycl::property::queue::in_order{}}};
    } catch (const sycl::exception& ex) {
        throw std::runtime_error("No SYCL CUDA GPU found. Underlying error: " + std::string(ex.what()));
    }
}

std::string describe_sycl_device(const std::string& device_name) {
    auto queue = make_sycl_queue(device_name);
    const auto device = queue.get_device();
    std::string description = device.get_info<sycl::info::device::name>() + " | " +
        device.get_info<sycl::info::device::vendor>() + " | backend cuda | driver " +
        device.get_info<sycl::info::device::driver_version>();
    description += " | fp64=";
    description += device.has(sycl::aspect::fp64) ? "yes" : "no";
    description += " atomic64=";
    description += device.has(sycl::aspect::atomic64) ? "yes" : "no";
    return description;
}

}  // namespace carbon
