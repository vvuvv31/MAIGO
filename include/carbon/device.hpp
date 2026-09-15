#pragma once

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace carbon {

sycl::queue make_sycl_queue(const std::string& device_name);

// Read-only capabilities of the SYCL device that `device_name` selects. When
// several devices match, the one with the largest global memory (then the most
// compute units) is reported. No queue is created.
struct DeviceCapabilities {
    std::string name{};
    std::string vendor{};
    std::string backend{};
    std::size_t device_count{0};
    std::size_t global_mem_bytes{0};
    std::size_t max_alloc_bytes{0};
    std::size_t max_compute_units{0};
    std::size_t max_work_group_size{0};
    std::size_t sub_group_size{0};
    bool fp64{false};
    bool atomic64{false};
};

[[nodiscard]] DeviceCapabilities probe_sycl_device(const std::string& device_name);

}  // namespace carbon

#endif
