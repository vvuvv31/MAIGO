#include "carbon/device.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <exception>
#include <stdexcept>
#include <string>

namespace carbon {
namespace {

const char* backend_name(sycl::backend backend) {
    switch (backend) {
        case sycl::backend::opencl:
            return "opencl";
        case sycl::backend::ext_oneapi_level_zero:
            return "level_zero";
        case sycl::backend::ext_oneapi_cuda:
            return "cuda";
        case sycl::backend::ext_oneapi_hip:
            return "hip";
        case sycl::backend::ext_oneapi_native_cpu:
            return "native_cpu";
        default:
            return "unknown";
    }
}

sycl::property_list make_queue_properties() {
    return sycl::property_list{sycl::property::queue::enable_profiling{},
                               sycl::property::queue::in_order{}};
}

auto make_async_handler() {
    return [](sycl::exception_list exceptions) {
        for (const auto& exception : exceptions) {
            std::rethrow_exception(exception);
        }
    };
}

// Prefer a GPU whose SYCL backend matches `wanted`. Returns negative score
// when the device is not usable for that preference.
int backend_gpu_score(const sycl::device& device, sycl::backend wanted) {
    if (!device.is_gpu()) {
        return -1;
    }
    if (device.get_backend() != wanted) {
        return -1;
    }
    // Prefer devices that can run the accurate double-atomic scorer.
    int score = 1000;
    if (device.has(sycl::aspect::fp64)) {
        score += 100;
    }
    if (device.has(sycl::aspect::atomic64)) {
        score += 100;
    }
    return score;
}

sycl::queue make_backend_gpu_queue(sycl::backend wanted, const std::string& label) {
    try {
        return sycl::queue{
            [=](const sycl::device& device) { return backend_gpu_score(device, wanted); },
            make_async_handler(), make_queue_properties()};
    } catch (const sycl::exception& ex) {
        throw std::runtime_error("No SYCL " + label +
                                 " GPU found (set ONEAPI_DEVICE_SELECTOR or install the "
                                 "matching runtime). Underlying error: " +
                                 std::string(ex.what()));
    }
}

}  // namespace

sycl::queue make_sycl_queue(const std::string& device_name) {
    const auto async_handler = make_async_handler();
    const auto properties = make_queue_properties();

    if (device_name == "gpu") {
        // Uses the default GPU selector. Pin the vendor at runtime with
        // ONEAPI_DEVICE_SELECTOR=level_zero:0 (Intel Arc) or cuda:gpu (NVIDIA).
        return sycl::queue{sycl::gpu_selector_v, async_handler, properties};
    }
    if (device_name == "cpu") {
        return sycl::queue{sycl::cpu_selector_v, async_handler, properties};
    }
    if (device_name == "default") {
        return sycl::queue{sycl::default_selector_v, async_handler, properties};
    }
    // Explicit vendor/backends so multi-GPU hosts can choose without env vars.
    if (device_name == "cuda" || device_name == "nvidia") {
        return make_backend_gpu_queue(sycl::backend::ext_oneapi_cuda, "CUDA/NVIDIA");
    }
    if (device_name == "level_zero" || device_name == "intel" || device_name == "arc") {
        return make_backend_gpu_queue(sycl::backend::ext_oneapi_level_zero,
                                      "Level Zero/Intel");
    }
    if (device_name == "opencl") {
        return make_backend_gpu_queue(sycl::backend::opencl, "OpenCL");
    }
    throw std::invalid_argument(
        "Unknown SYCL device selector: " + device_name +
        " (expected serial path uses host; SYCL: gpu|cpu|default|cuda|nvidia|"
        "level_zero|intel|arc|opencl)");
}

std::string describe_sycl_device(const std::string& device_name) {
    auto queue = make_sycl_queue(device_name);
    const auto& device = queue.get_device();
    std::string description = device.get_info<sycl::info::device::name>() + " | " +
                              device.get_info<sycl::info::device::vendor>() + " | backend " +
                              backend_name(device.get_backend()) + " | driver " +
                              device.get_info<sycl::info::device::driver_version>();
    description += " | fp64=";
    description += device.has(sycl::aspect::fp64) ? "yes" : "no";
    description += " atomic64=";
    description += device.has(sycl::aspect::atomic64) ? "yes" : "no";
    return description;
}

}  // namespace carbon

#endif
