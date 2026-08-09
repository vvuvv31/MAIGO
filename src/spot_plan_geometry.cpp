#include "carbon/spot_plan_geometry.hpp"

namespace carbon {

std::optional<std::pair<double, double>> history_weighted_entrance_pivot(
    const std::span<const WeightedEntrancePoint> points) noexcept {
    double sum_weight = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const auto& point : points) {
        const auto weight = static_cast<double>(point.histories);
        sum_weight += weight;
        sum_x += weight * point.x_mm;
        sum_y += weight * point.y_mm;
    }
    if (sum_weight <= 0.0) {
        return std::nullopt;
    }
    return std::pair{sum_x / sum_weight, sum_y / sum_weight};
}

}  // namespace carbon
