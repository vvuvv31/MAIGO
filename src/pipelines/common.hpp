#pragma once

#include "carbon/secondary_schedule.hpp"
#include <cstddef>
#include <type_traits>

namespace carbon::pipelines {
// Shared scheduling formulas; each pipeline owns the choice of whether to use them.
inline constexpr std::size_t broadbeam_history_chunk(std::size_t histories) {
    return histories > 1000000 ? 34816U : 17408U;
}
inline constexpr std::size_t minibeam_history_chunk(std::size_t histories) {
    return histories > 1000000 ? 34816U : 4096U;
}
} // namespace carbon::pipelines
