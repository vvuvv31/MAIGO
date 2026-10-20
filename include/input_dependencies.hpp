#pragma once
#include "immutable_input_cache.hpp"
#include "ct_grid.hpp"
#include <cctype>
#include <cmath>
#include <fstream>
#include <string_view>

namespace carbon {
// Packed CCTG v3 only; slice conversion is outside this runtime.
inline std::string ct_input_key(const std::filesystem::path& path) {
    return input_cache_key(path);
}
}
