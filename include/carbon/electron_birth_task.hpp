#pragma once
#include <cstdint>
#include <type_traits>
namespace carbon {
// Independent birth state: retain exact CDF bits until the electron worker samples.
struct QueuedElectronBirth {
    double energy{};
    std::uint64_t cdf_bits{};
    std::uint32_t source{};
};
static_assert(std::is_trivially_copyable_v<QueuedElectronBirth>);
} // namespace carbon
