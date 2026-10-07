#pragma once
#include <sycl/sycl.hpp>
#include <string>
namespace carbon {
// Only the CUDA GPU selector is supported; no vendor aliases or fallback.
sycl::queue make_sycl_queue(const std::string& device_name);
}  // namespace carbon
