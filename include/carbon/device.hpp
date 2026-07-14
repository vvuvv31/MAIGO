#pragma once

#ifdef CARBON_HAS_SYCL

#include <sycl/sycl.hpp>

#include <string>

namespace carbon {

sycl::queue make_sycl_queue(const std::string& device_name);

}  // namespace carbon

#endif

