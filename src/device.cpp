#include "carbon/device.hpp"
#include "carbon/transport.hpp"

#ifdef CARBON_HAS_SYCL

#include <exception>
#include <stdexcept>

namespace carbon {

sycl::queue make_sycl_queue(const std::string& device_name) {
    const auto async_handler = [](sycl::exception_list exceptions) {
        for (const auto& exception : exceptions) {
            std::rethrow_exception(exception);
        }
    };
    const sycl::property_list properties{sycl::property::queue::enable_profiling{},
                                         sycl::property::queue::in_order{}};

    if (device_name == "gpu") {
        return sycl::queue{sycl::gpu_selector_v, async_handler, properties};
    }
    if (device_name == "cpu") {
        return sycl::queue{sycl::cpu_selector_v, async_handler, properties};
    }
    if (device_name == "default") {
        return sycl::queue{sycl::default_selector_v, async_handler, properties};
    }
    throw std::invalid_argument("Unknown SYCL device selector: " + device_name);
}

std::string describe_sycl_device(const std::string& device_name) {
    auto queue = make_sycl_queue(device_name);
    const auto& device = queue.get_device();
    return device.get_info<sycl::info::device::name>() + " | " +
           device.get_info<sycl::info::device::vendor>() + " | driver " +
           device.get_info<sycl::info::device::driver_version>();
}

}  // namespace carbon

#endif
