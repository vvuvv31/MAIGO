#pragma once
#include <sycl/sycl.hpp>
#include <string>
namespace carbon {
// Explicit CUDA or Intel Level Zero GPU selection; no CPU/backend fallback.
sycl::queue make_sycl_queue(const std::string& device_name);
}  // namespace carbon
