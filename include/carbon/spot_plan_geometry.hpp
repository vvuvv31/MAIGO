#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>

namespace carbon {

struct WeightedEntrancePoint {
    double x_mm{0.0};
    double y_mm{0.0};
    std::size_t histories{0};
};

// History-weighted entrance-plane centroid used as the plan-level lateral
// affine pivot. Returns nullopt when the plan has no positive history weight.
[[nodiscard]] std::optional<std::pair<double, double>>
history_weighted_entrance_pivot(
    std::span<const WeightedEntrancePoint> points) noexcept;

}  // namespace carbon
